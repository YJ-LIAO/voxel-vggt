"""Train TokenScorer from counterfactual retention oracle shards."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM, TokenScorer
from ovggt.training.token_oracle_dataset import (
    CounterfactualOracleDataset,
    collate_oracle_pairs,
    token_oracle_ranking_loss,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML config; CLI options override YAML values")
    parser.add_argument("--oracle-shards", nargs="+", help="Path(s) to .pt oracle shards")
    parser.add_argument("--output", help="Checkpoint output path")
    parser.add_argument("--score-state-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--regression-weight", type=float)
    parser.add_argument("--min-loss-gap", type=float)
    parser.add_argument("--device")
    parser.add_argument(
        "--score-state-proj-checkpoint",
        help="Optional OVGGT checkpoint containing aggregator.score_state_projs.* weights",
    )
    parser.add_argument("--val-fraction", type=float, help="Fraction of samples for held-out validation")
    parser.add_argument(
        "--split-key",
        choices=["event_id_hash", "sequence_id"],
        help="Strategy for train/val split: event_id_hash or sequence_id",
    )
    parser.add_argument("--split-seed", type=int, help="Seed for deterministic train/val split")
    parser.add_argument(
        "--stress-profile-weights",
        type=str,
        default=None,
        help="JSON mapping event type to sampling weight, e.g. '{\"dedup\":2,\"eviction\":1}'",
    )
    args = parser.parse_args(argv)

    defaults = {
        "oracle_shards": None,
        "output": None,
        "score_state_dim": 128,
        "hidden_dim": 256,
        "num_layers": 24,
        "batch_size": 64,
        "epochs": 1,
        "lr": 1e-4,
        "regression_weight": 0.1,
        "min_loss_gap": 0.01,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "score_state_proj_checkpoint": None,
        "val_fraction": 0.1,
        "split_key": "event_id_hash",
        "split_seed": 0,
        "stress_profile_weights": None,
    }
    values = dict(defaults)
    if args.config:
        config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        if not isinstance(config, dict):
            raise ValueError(f"Expected mapping config in {args.config}")
        values.update({k: v for k, v in config.items() if k in values})

    cli_values = vars(args)
    for key, value in cli_values.items():
        if key == "config" or value is None:
            continue
        values[key] = value

    if not values["oracle_shards"]:
        parser.error("--oracle-shards is required, either in YAML or CLI")
    if not values["output"]:
        parser.error("--output is required, either in YAML or CLI")
    if isinstance(values["oracle_shards"], (str, Path)):
        values["oracle_shards"] = [str(values["oracle_shards"])]
    else:
        values["oracle_shards"] = [str(path) for path in values["oracle_shards"]]

    return argparse.Namespace(config=args.config, **values)


def main() -> None:
    args = parse_args()
    dataset = CounterfactualOracleDataset(args.oracle_shards, min_loss_gap=args.min_loss_gap)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No pairwise samples found in oracle shards with min_loss_gap={args.min_loss_gap}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_oracle_pairs,
    )
    device = torch.device(args.device)
    scorer = TokenScorer(
        score_state_dim=args.score_state_dim,
        metadata_dim=TOKEN_METADATA_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        epoch_metrics = _empty_metric_sums()
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            logits = scorer(batch["score_state"], batch["metadata_features"], batch["layer_id"])
            loss, details = token_oracle_ranking_loss(
                logits,
                batch,
                regression_weight=args.regression_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            _add_metric_sums(epoch_metrics, float(loss.detach().cpu()), details)
            if step % 50 == 0:
                print(
                    format_training_log_line(
                        epoch=epoch,
                        step=step,
                        loss=float(loss.detach().cpu()),
                        details=details,
                    ),
                    flush=True,
                )
            step += 1
        print(format_epoch_summary_line(epoch=epoch, metrics=epoch_metrics), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    projection_state = load_score_state_projection_state(args.score_state_proj_checkpoint)
    if not projection_state:
        projection_state = load_score_state_projection_state_from_oracle_shards(args.oracle_shards)
    deploy_state = build_ovggt_token_scorer_state_dict(
        scorer_state=scorer.state_dict(),
        num_layers=args.num_layers,
        score_state_projection_state=projection_state,
    )
    torch.save(
        {
            "token_scorer": scorer.state_dict(),
            "model": deploy_state,
            "score_state_dim": args.score_state_dim,
            "metadata_dim": TOKEN_METADATA_FEATURE_DIM,
            "num_layers": args.num_layers,
            "score_state_projection_checkpoint": args.score_state_proj_checkpoint,
            "hidden_dim": args.hidden_dim,
        },
        output,
    )
    print(f"saved_checkpoint={output} total_steps={step}", flush=True)


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def format_training_log_line(epoch: int, step: int, loss: float, details: dict) -> str:
    return (
        f"epoch={epoch} step={step} loss={float(loss):.6f} "
        f"pairwise={float(details['pairwise']):.6f} "
        f"regression={float(details['regression']):.6f} "
        f"rank_acc={float(details['rank_acc']):.4f} "
        f"mean_score_diff={float(details['mean_score_diff']):.6f}"
    )


def format_epoch_summary_line(epoch: int, metrics: dict) -> str:
    count = max(int(metrics["count"]), 1)
    details = {
        "pairwise": metrics["pairwise"] / count,
        "regression": metrics["regression"] / count,
        "rank_acc": metrics["rank_acc"] / count,
        "mean_score_diff": metrics["mean_score_diff"] / count,
    }
    return (
        f"epoch_summary={epoch} batches={count} "
        f"loss={metrics['loss'] / count:.6f} "
        f"pairwise={details['pairwise']:.6f} "
        f"regression={details['regression']:.6f} "
        f"rank_acc={details['rank_acc']:.4f} "
        f"mean_score_diff={details['mean_score_diff']:.6f}"
    )


def _empty_metric_sums() -> dict:
    return {
        "count": 0,
        "loss": 0.0,
        "pairwise": 0.0,
        "regression": 0.0,
        "rank_acc": 0.0,
        "mean_score_diff": 0.0,
    }


def _add_metric_sums(metrics: dict, loss: float, details: dict) -> None:
    metrics["count"] += 1
    metrics["loss"] += float(loss)
    for key in ("pairwise", "regression", "rank_acc", "mean_score_diff"):
        metrics[key] += float(details[key])


def build_ovggt_token_scorer_state_dict(
    scorer_state: dict,
    num_layers: int,
    score_state_projection_state: dict | None = None,
) -> dict:
    deploy_state = {}
    for layer_idx in range(int(num_layers)):
        for key, value in scorer_state.items():
            deploy_state[f"aggregator.token_scorers.{layer_idx}.{key}"] = value.detach().cpu().clone()
    if score_state_projection_state:
        for key, value in score_state_projection_state.items():
            if key.startswith("aggregator.score_state_projs."):
                deploy_state[key] = value.detach().cpu().clone()
    return deploy_state


def load_score_state_projection_state(checkpoint_path: str | None) -> dict:
    if not checkpoint_path:
        return {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dict at {checkpoint_path}")
    return {
        key: value
        for key, value in checkpoint.items()
        if key.startswith("aggregator.score_state_projs.")
    }


def load_score_state_projection_state_from_oracle_shards(shard_paths: Sequence[str | Path]) -> dict:
    for shard_path in shard_paths:
        shard = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
        if isinstance(shard, dict):
            state = shard.get("score_state_projection_state", {})
            if state:
                return {
                    key: value
                    for key, value in state.items()
                    if key.startswith("aggregator.score_state_projs.")
                }
    return {}


def _stable_bucket(value: str, seed: int = 0, buckets: int = 10000) -> int:
    digest = hashlib.md5(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


def split_oracle_pair_samples(
    samples: list[dict],
    val_fraction: float = 0.1,
    split_key: str = "event_id_hash",
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Split pair samples before DataLoader construction.

    Early calibration uses split_key="event_id_hash" so pair samples from the
    same oracle event never appear in both splits. Full production uses
    split_key="sequence_id" so scenes/sequences do not leak.
    """
    threshold = int(float(val_fraction) * 10000)
    train, val = [], []
    for sample in samples:
        if split_key == "sequence_id":
            prov = sample.get("sequence_provenance") or {}
            key = str(prov.get("sequence_id") or sample.get("event_id") or "")
        elif split_key == "event_id_hash":
            key = str(sample.get("event_id") or "")
        else:
            raise ValueError(f"Unsupported split_key={split_key}")
        target = val if _stable_bucket(key, seed=seed) < threshold else train
        target.append(sample)
    return train, val


def summarize_metrics_by_event_type(rows: list[dict]) -> dict[str, dict]:
    """Summarize per-event-type validation metrics.

    Each row must have "event_type" (str) and "rank_correct" (bool or bool tensor).
    Returns a dict mapping event_type to {"count": int, "rank_acc": float}.
    """
    buckets: dict[str, list[bool]] = {}
    for row in rows:
        et = row.get("event_type", "unknown")
        rc = row.get("rank_correct", False)
        if isinstance(rc, torch.Tensor):
            rc = bool(rc.item())
        buckets.setdefault(et, []).append(bool(rc))
    result = {}
    for et, flags in sorted(buckets.items()):
        count = len(flags)
        rank_acc = sum(flags) / max(count, 1)
        result[et] = {"count": count, "rank_acc": rank_acc}
    return result


if __name__ == "__main__":
    main()
