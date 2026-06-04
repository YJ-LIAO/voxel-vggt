"""Joint training of token-ranking and FIFO-count retention policy.

Trains a single ``JointRetentionPolicy`` (shared encoder + token ranking head
+ FIFO count classifier) from counterfactual oracle shards.  The shared
encoder is updated by both token-ranking and count-classification losses so
that it learns representations useful for both tasks.

Checkpoint output includes the native joint state, exported TokenScorer state,
exported FifoCountHead (shared_encoder_v2) state, and a full deploy state dict
with ``aggregator.token_scorers.*`` and ``aggregator.count_head.*`` keys ready
for direct loading into an OVGGT model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ovggt.layers.retention_policy import (
    JointRetentionPolicy,
    build_count_head_state_from_joint,
    build_ovggt_joint_retention_state_dict,
    build_token_scorer_state_from_joint,
)
from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
from ovggt.training.token_oracle_dataset import (
    CounterfactualOracleDataset,
    FifoCountDataset,
    collate_fifo_count_samples,
    collate_oracle_pairs,
    load_oracle_events,
    split_oracle_events,
    token_oracle_ranking_loss,
)
from train_token_scorer_oracle import (
    load_score_state_projection_state,
    load_score_state_projection_state_from_oracle_shards,
)


# --------------------------------------------------------------------------- #
# Config / CLI (three-layer override: defaults < YAML < CLI)
# --------------------------------------------------------------------------- #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML config; CLI options override YAML values")
    parser.add_argument("--oracle-shards", nargs="+", help="Path(s) to .pt oracle shards")
    parser.add_argument("--output", help="Checkpoint output path")
    parser.add_argument("--score-state-proj-checkpoint",
                        help="Optional OVGGT checkpoint with aggregator.score_state_projs.* weights")
    parser.add_argument("--score-state-dim", type=int)
    parser.add_argument("--metadata-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--count-candidates", nargs="+", type=int, help="Candidate count values")
    parser.add_argument("--count-head-arch", help="Count head architecture identifier")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--regression-weight", type=float)
    parser.add_argument("--min-loss-gap", type=float)
    parser.add_argument("--count-loss-weight", type=float)
    parser.add_argument("--count-label-reduction", choices=["min", "mean"],
                        help="Label reduction strategy for count head")
    parser.add_argument("--count-repeat-factor", type=float,
                        help="Repeat count loader this many times per epoch (0 disables count)")
    parser.add_argument("--val-fraction", type=float, help="Fraction of events for validation")
    parser.add_argument("--split-key", choices=["event_id_hash", "sequence_id"],
                        help="Strategy for train/val split")
    parser.add_argument("--split-seed", type=int, help="Seed for deterministic train/val split")
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    defaults = {
        "oracle_shards": None,
        "output": None,
        "score_state_proj_checkpoint": None,
        "score_state_dim": 128,
        "metadata_dim": TOKEN_METADATA_FEATURE_DIM,
        "hidden_dim": 256,
        "num_layers": 24,
        "count_candidates": [0, 8, 16, 32, 64, 128],
        "count_head_arch": "shared_encoder_v2",
        "batch_size": 64,
        "epochs": 1,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "regression_weight": 0.1,
        "min_loss_gap": 0.01,
        "count_loss_weight": 1.0,
        "count_label_reduction": "min",
        "count_repeat_factor": 1.0,
        "val_fraction": 0.1,
        "split_key": "event_id_hash",
        "split_seed": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def _empty_metric_sums() -> dict:
    return {
        "count": 0,
        "loss": 0.0,
        # Token metrics
        "pairwise": 0.0,
        "regression": 0.0,
        "rank_acc": 0.0,
        "mean_score_diff": 0.0,
        # Count metrics
        "count_loss": 0.0,
        "count_correct": 0,
        "count_total": 0,
        "count_abs_error": 0.0,
        # Gradient norms
        "shared_encoder_grad_norm": 0.0,
        "token_head_grad_norm": 0.0,
        "count_head_grad_norm": 0.0,
    }


def _add_metric_sums(
    metrics: dict,
    loss: float,
    token_details: dict,
    count_loss: float | None = None,
    count_correct: int = 0,
    count_total: int = 0,
    count_abs_error: float = 0.0,
    shared_encoder_grad_norm: float = 0.0,
    token_head_grad_norm: float = 0.0,
    count_head_grad_norm: float = 0.0,
) -> None:
    metrics["count"] += 1
    metrics["loss"] += float(loss)
    for key in ("pairwise", "regression", "rank_acc", "mean_score_diff"):
        metrics[key] += float(token_details[key])
    if count_loss is not None:
        metrics["count_loss"] += float(count_loss)
    metrics["count_correct"] += count_correct
    metrics["count_total"] += count_total
    metrics["count_abs_error"] += count_abs_error
    metrics["shared_encoder_grad_norm"] += shared_encoder_grad_norm
    metrics["token_head_grad_norm"] += token_head_grad_norm
    metrics["count_head_grad_norm"] += count_head_grad_norm


def _param_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


# --------------------------------------------------------------------------- #
# Core training function
# --------------------------------------------------------------------------- #

def train_joint_retention(
    oracle_shards: list[str],
    output: str,
    score_state_proj_checkpoint: str | None = None,
    score_state_dim: int = 128,
    metadata_dim: int = TOKEN_METADATA_FEATURE_DIM,
    hidden_dim: int = 256,
    num_layers: int = 24,
    count_candidates: Sequence[int] = (0, 8, 16, 32, 64, 128),
    count_head_arch: str = "shared_encoder_v2",
    batch_size: int = 64,
    epochs: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    regression_weight: float = 0.1,
    min_loss_gap: float = 0.01,
    count_loss_weight: float = 1.0,
    count_label_reduction: str = "min",
    count_repeat_factor: float = 1.0,
    val_fraction: float = 0.1,
    split_key: str = "event_id_hash",
    split_seed: int = 0,
    device: str = "cpu",
) -> None:
    """Train JointRetentionPolicy and save checkpoint.

    This function contains the core training loop so it can be called
    from tests without going through argparse.
    """
    count_candidates = list(count_candidates)
    torch_device = torch.device(device)

    # Load events and split
    all_events = load_oracle_events(oracle_shards)
    train_events, _ = split_oracle_events(
        all_events,
        val_fraction=val_fraction,
        split_key=split_key,
        seed=split_seed,
    )

    # Build token-ranking dataset from train events
    token_dataset = CounterfactualOracleDataset.from_events(
        train_events, min_loss_gap=min_loss_gap,
    )
    if len(token_dataset) == 0:
        raise RuntimeError(
            f"No pairwise token samples found in oracle shards with min_loss_gap={min_loss_gap}"
        )
    token_loader = DataLoader(
        token_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_oracle_pairs,
    )

    # Build FIFO count dataset from train events
    count_dataset = FifoCountDataset.from_events(
        train_events,
        count_candidates=count_candidates,
        label_reduction=count_label_reduction,
    )
    has_count_samples = len(count_dataset) > 0

    if has_count_samples:
        count_loader = DataLoader(
            count_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fifo_count_samples,
        )
    else:
        count_loader = None

    count_head_trained = False

    # Build joint model
    joint = JointRetentionPolicy(
        score_state_dim=score_state_dim,
        metadata_dim=metadata_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        count_candidates=count_candidates,
    ).to(torch_device)

    optimizer = torch.optim.AdamW(
        joint.parameters(), lr=lr, weight_decay=weight_decay,
    )

    step = 0
    for epoch in range(epochs):
        epoch_metrics = _empty_metric_sums()

        # Count loader iteration with repeat factor
        if has_count_samples and count_repeat_factor > 0:
            count_iter = iter(count_loader)
            count_steps_this_epoch = 0
            max_count_steps = len(count_loader) * max(1, int(count_repeat_factor))

        for token_batch in token_loader:
            token_batch = _move_batch_to_device(token_batch, torch_device)

            # --- Token loss ---
            token_logits = joint.forward_token(
                token_batch["score_state"],
                token_batch["metadata_features"],
                layer_id=token_batch["layer_id"],
            )
            token_loss, token_details = token_oracle_ranking_loss(
                token_logits,
                token_batch,
                regression_weight=regression_weight,
            )

            # --- Count loss (multi-task batching with count_repeat_factor) ---
            count_loss_val = None
            count_correct = 0
            count_total = 0
            count_abs_error = 0.0

            if has_count_samples and count_repeat_factor > 0 and count_steps_this_epoch < max_count_steps:
                count_batch = None
                try:
                    count_batch = next(count_iter)
                except StopIteration:
                    if count_repeat_factor > 1:
                        count_iter = iter(count_loader)
                        count_batch = next(count_iter)
                    else:
                        count_batch = None

                if count_batch is not None:
                    count_batch = _move_batch_to_device(count_batch, torch_device)
                    count_logits = joint.forward_count(
                        count_batch["score_state"],
                        count_batch["metadata_features"],
                        layer_id=count_batch["layer_id"],
                        token_mask=count_batch["token_mask"],
                    )
                    count_loss_tensor = F.cross_entropy(count_logits, count_batch["target"])
                    count_loss_val = float(count_loss_tensor.detach().cpu().item())
                    count_steps_this_epoch += 1

                    # Count metrics
                    preds = count_logits.argmax(dim=-1)
                    batch_count_total = count_batch["target"].shape[0]
                    count_correct = int((preds == count_batch["target"]).sum().detach().cpu().item())
                    count_total = batch_count_total

                    candidates_tensor = joint.count_head.candidates
                    pred_counts = candidates_tensor[preds].float()
                    target_counts = candidates_tensor[count_batch["target"]].float()
                    count_abs_error = float(
                        (pred_counts - target_counts).abs().sum().detach().cpu().item()
                    )

            # --- Total loss ---
            total_loss = token_loss
            if count_loss_val is not None:
                total_loss = total_loss + count_loss_weight * count_loss_tensor

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # Gradient norms
            enc_grad = _param_grad_norm(joint.encoder)
            tok_grad = _param_grad_norm(joint.token_head)
            cnt_grad = _param_grad_norm(joint.count_head)

            _add_metric_sums(
                epoch_metrics,
                loss=float(total_loss.detach().cpu()),
                token_details=token_details,
                count_loss=count_loss_val,
                count_correct=count_correct,
                count_total=count_total,
                count_abs_error=count_abs_error,
                shared_encoder_grad_norm=enc_grad,
                token_head_grad_norm=tok_grad,
                count_head_grad_norm=cnt_grad,
            )

            if count_loss_val is not None:
                count_head_trained = True

            if step % 50 == 0:
                parts = [
                    f"epoch={epoch} step={step}",
                    f"loss={float(total_loss.detach().cpu()):.6f}",
                    f"pairwise={float(token_details['pairwise']):.6f}",
                    f"regression={float(token_details['regression']):.6f}",
                    f"rank_acc={float(token_details['rank_acc']):.4f}",
                ]
                if count_loss_val is not None:
                    parts.append(f"count_loss={count_loss_val:.6f}")
                print(" ".join(parts), flush=True)
            step += 1

        # Epoch summary
        n = max(int(epoch_metrics["count"]), 1)
        has_count = epoch_metrics["count_total"] > 0
        summary_parts = [
            f"epoch_summary={epoch} batches={n}",
            f"loss={epoch_metrics['loss'] / n:.6f}",
            f"pairwise={epoch_metrics['pairwise'] / n:.6f}",
            f"regression={epoch_metrics['regression'] / n:.6f}",
            f"rank_acc={epoch_metrics['rank_acc'] / n:.4f}",
            f"mean_score_diff={epoch_metrics['mean_score_diff'] / n:.6f}",
        ]
        if has_count:
            cn = max(epoch_metrics["count_total"], 1)
            summary_parts.extend([
                f"count_acc={epoch_metrics['count_correct'] / cn:.4f}",
                f"mean_abs_count_error={epoch_metrics['count_abs_error'] / cn:.2f}",
            ])
        summary_parts.extend([
            f"shared_enc_grad={epoch_metrics['shared_encoder_grad_norm'] / n:.4f}",
            f"token_head_grad={epoch_metrics['token_head_grad_norm'] / n:.4f}",
            f"count_head_grad={epoch_metrics['count_head_grad_norm'] / n:.4f}",
        ])
        print(" ".join(summary_parts), flush=True)

    # --- Save checkpoint ---
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Projection state
    projection_state = load_score_state_projection_state(score_state_proj_checkpoint)
    if not projection_state:
        projection_state = load_score_state_projection_state_from_oracle_shards(oracle_shards)

    # Export sub-components
    token_scorer_state = build_token_scorer_state_from_joint(joint)
    count_head_state = build_count_head_state_from_joint(joint) if count_head_trained else None
    deploy_state = build_ovggt_joint_retention_state_dict(
        token_scorer_state=token_scorer_state,
        count_head_state=count_head_state,
        num_layers=num_layers,
        score_state_projection_state=projection_state,
    )

    torch.save(
        {
            "joint_arch": "shared_token_encoder_v1",
            "count_head_arch": count_head_arch,
            "joint_policy": joint.state_dict(),
            "token_scorer": token_scorer_state,
            "count_head": count_head_state,
            "model": deploy_state,
            "score_state_dim": score_state_dim,
            "metadata_dim": metadata_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "count_candidates": count_candidates,
            "count_head_trained": bool(count_head_trained),
            "score_state_projection_checkpoint": score_state_proj_checkpoint,
        },
        output_path,
    )
    print(f"saved_checkpoint={output_path} total_steps={step}", flush=True)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    train_joint_retention(
        oracle_shards=args.oracle_shards,
        output=args.output,
        score_state_proj_checkpoint=args.score_state_proj_checkpoint,
        score_state_dim=args.score_state_dim,
        metadata_dim=args.metadata_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        count_candidates=args.count_candidates,
        count_head_arch=args.count_head_arch,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        regression_weight=args.regression_weight,
        min_loss_gap=args.min_loss_gap,
        count_loss_weight=args.count_loss_weight,
        count_label_reduction=args.count_label_reduction,
        count_repeat_factor=args.count_repeat_factor,
        val_fraction=args.val_fraction,
        split_key=args.split_key,
        split_seed=args.split_seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
