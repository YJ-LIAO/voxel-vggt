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
import copy
import json
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
    summarize_fifo_count_samples,
    summarize_oracle_pair_samples,
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
    parser.add_argument("--fifo-token-pair-mode", choices=["any", "same_keep_count"])
    parser.add_argument("--token-score-mode", choices=["set_mean", "delta_mean"])
    parser.add_argument("--pair-sampling-seed", type=int)
    parser.add_argument("--max-pairs-per-event", type=int)
    parser.add_argument("--min-loss-gap-by-event-type",
                        help="JSON dict mapping event_type to min_loss_gap, e.g. '{\"eviction\": 0.02}'")
    parser.add_argument("--max-loss-gap", type=float)
    parser.add_argument("--min-count-loss-gap", type=float)
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction)
    parser.add_argument("--best-metric", choices=[
        "token.rank_acc",
        "token.eviction_rank_acc",
        "token.fifo_topk_rank_acc",
        "token.eviction_fifo_mean_rank_acc",
        "count.accuracy",
    ])
    parser.add_argument("--best-output")
    parser.add_argument("--early-stop-patience", type=int)
    parser.add_argument("--early-stop-min-delta", type=float)
    parser.add_argument("--deploy-count-head", choices=["auto", "always", "never"])
    parser.add_argument("--deploy-count-head-min-delta", type=float)
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
        "fifo_token_pair_mode": "same_keep_count",
        "token_score_mode": "delta_mean",
        "pair_sampling_seed": 0,
        "max_pairs_per_event": 64,
        "min_loss_gap_by_event_type": None,
        "max_loss_gap": None,
        "min_count_loss_gap": 0.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_best": False,
        "best_metric": "token.rank_acc",
        "best_output": None,
        "early_stop_patience": None,
        "early_stop_min_delta": 0.0,
        "deploy_count_head": "auto",
        "deploy_count_head_min_delta": 0.0,
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
        if key == "min_loss_gap_by_event_type" and isinstance(value, str):
            values[key] = json.loads(value)
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


def _select_best_metric(
    token_metrics: dict,
    count_metrics: dict,
    metric_name: str,
) -> float:
    """Extract a scalar metric value for best-checkpoint selection.

    Returns float("-inf") with a warning if the metric key is missing.
    """
    if metric_name == "token.rank_acc":
        val = token_metrics.get("rank_acc")
    elif metric_name == "token.eviction_rank_acc":
        per_et = token_metrics.get("per_event_type", {})
        eviction = per_et.get("eviction", {})
        val = eviction.get("rank_acc")
    elif metric_name == "token.fifo_topk_rank_acc":
        per_et = token_metrics.get("per_event_type", {})
        fifo = per_et.get("fifo_topk", {})
        val = fifo.get("rank_acc")
    elif metric_name == "token.eviction_fifo_mean_rank_acc":
        per_et = token_metrics.get("per_event_type", {})
        vals = []
        for key in ("eviction", "fifo_topk"):
            if key in per_et and "rank_acc" in per_et[key]:
                vals.append(float(per_et[key]["rank_acc"]))
        val = sum(vals) / len(vals) if vals else None
    elif metric_name == "count.accuracy":
        val = count_metrics.get("accuracy")
    else:
        print(f"WARNING: unknown best_metric '{metric_name}'", flush=True)
        return float("-inf")

    if val is None:
        print(f"WARNING: best_metric '{metric_name}' not found in validation metrics", flush=True)
        return float("-inf")
    return float(val)


def _should_deploy_count_head(
    count_head_trained: bool,
    count_metrics: dict,
    deploy_count_head: str,
    min_delta: float,
) -> bool:
    """Decide whether to include count head in the deploy state dict.

    Rules:
      - Not trained => False
      - "always" => True (if trained)
      - "never" => False
      - "auto" => accuracy >= majority_accuracy + min_delta
    """
    if not count_head_trained:
        return False
    if deploy_count_head == "always":
        return True
    if deploy_count_head == "never":
        return False
    # auto mode
    acc = count_metrics.get("accuracy")
    majority = count_metrics.get("majority_accuracy")
    if acc is None or majority is None:
        return False
    return float(acc) >= float(majority) + float(min_delta)


def _save_joint_checkpoint(
    joint: JointRetentionPolicy,
    output_path: Path,
    *,
    checkpoint_role: str,
    count_head_trained: bool,
    count_head_arch: str,
    num_layers: int,
    count_candidates: list[int],
    score_state_dim: int,
    metadata_dim: int,
    hidden_dim: int,
    score_state_proj_checkpoint: str | None,
    projection_state: dict | None,
    dataset_stats: dict,
    training_options: dict,
    validation_metrics: dict,
    best_info: dict,
    deploy_count_head: str,
    deploy_count_head_min_delta: float,
) -> None:
    """Build and save a checkpoint with deploy gating applied."""
    token_scorer_state = build_token_scorer_state_from_joint(joint)
    count_head_state = build_count_head_state_from_joint(joint) if count_head_trained else None

    # Deploy gating: decide whether to include count head in deploy state
    # No validation data => auto mode cannot make an informed decision => disable
    has_val = best_info.get("has_validation", True)
    count_metrics_for_gating = validation_metrics.get("count", {})
    if not has_val and deploy_count_head == "auto":
        deploy_count = False
    else:
        deploy_count = _should_deploy_count_head(
            count_head_trained=count_head_trained,
            count_metrics=count_metrics_for_gating,
            deploy_count_head=deploy_count_head,
            min_delta=deploy_count_head_min_delta,
        )
    deploy_count_head_state = count_head_state if deploy_count else None
    deploy_state = build_ovggt_joint_retention_state_dict(
        token_scorer_state=token_scorer_state,
        count_head_state=deploy_count_head_state,
        num_layers=num_layers,
        score_state_projection_state=projection_state,
    )

    checkpoint = {
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
        "count_head_deploy_enabled": deploy_count,
        "score_state_projection_checkpoint": score_state_proj_checkpoint,
        "validation_metrics": validation_metrics,
        "dataset_stats": dataset_stats,
        "training_options": training_options,
        # Checkpoint role and best-info
        "checkpoint_role": checkpoint_role,
        "best_metric": best_info.get("best_metric"),
        "best_metric_value": best_info.get("best_metric_value"),
        "best_epoch": best_info.get("best_epoch"),
        "final_epoch": best_info.get("final_epoch"),
        "has_validation": best_info.get("has_validation"),
        "best_selection_reason": best_info.get("best_selection_reason"),
        "best_validation_metrics": best_info.get("best_validation_metrics"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"saved_checkpoint={output_path} role={checkpoint_role}", flush=True)


# --------------------------------------------------------------------------- #
# Validation evaluation helpers
# --------------------------------------------------------------------------- #

def _evaluate_token_ranking(
    joint: JointRetentionPolicy,
    dataset: CounterfactualOracleDataset,
    batch_size: int,
    device: torch.device,
    score_mode: str = "set_mean",
) -> dict:
    """Evaluate token ranking on a dataset (no gradients).

    Returns dict with: count, loss, pairwise, regression, rank_acc,
    mean_score_diff, per_event_type.
    """
    if dataset is None or len(dataset) == 0:
        return {
            "count": 0,
            "loss": 0.0,
            "pairwise": 0.0,
            "regression": 0.0,
            "rank_acc": 0.0,
            "mean_score_diff": 0.0,
            "per_event_type": {},
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_oracle_pairs,
    )

    total_loss = 0.0
    total_pairwise = 0.0
    total_regression = 0.0
    total_rank_acc = 0.0
    total_mean_score_diff = 0.0
    total_samples = 0
    per_event_type_correct: dict[str, int] = {}
    per_event_type_total: dict[str, int] = {}

    joint.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            logits = joint.forward_token(
                batch["score_state"],
                batch["metadata_features"],
                layer_id=batch["layer_id"],
            )
            loss_tensor, details = token_oracle_ranking_loss(
                logits, batch,
                regression_weight=0.1,
                score_mode=score_mode,
            )
            n = logits.shape[0]
            total_loss += float(loss_tensor.detach().cpu()) * n
            total_pairwise += details["pairwise"] * n
            total_regression += details["regression"] * n
            total_rank_acc += details["rank_acc"] * n
            total_mean_score_diff += details["mean_score_diff"] * n
            total_samples += n

            # Per-event-type accuracy
            event_types = batch.get("event_type")
            if event_types is not None:
                better_mask = batch["better_mask"].to(device=device)
                worse_mask = batch["worse_mask"].to(device=device)
                token_mask = batch.get("token_mask")
                if token_mask is not None:
                    token_mask = token_mask.to(device=device)

                # Compute score diff per sample for correctness
                if score_mode == "delta_mean":
                    better_only = better_mask & ~worse_mask
                    worse_only = worse_mask & ~better_mask
                    from ovggt.training.token_oracle_dataset import _subset_score
                    better_score = _subset_score(logits, better_only, token_mask, reduction="mean")
                    worse_score = _subset_score(logits, worse_only, token_mask, reduction="mean")
                    better_only_count = better_only.sum(dim=1)
                    worse_only_count = worse_only.sum(dim=1)
                    fallback = (better_only_count == 0) | (worse_only_count == 0)
                    if fallback.any():
                        set_better = _subset_score(logits, better_mask, token_mask, reduction="mean")
                        set_worse = _subset_score(logits, worse_mask, token_mask, reduction="mean")
                        better_score = torch.where(fallback, set_better, better_score)
                        worse_score = torch.where(fallback, set_worse, worse_score)
                else:
                    from ovggt.training.token_oracle_dataset import _subset_score
                    better_score = _subset_score(logits, better_mask, token_mask, reduction="mean")
                    worse_score = _subset_score(logits, worse_mask, token_mask, reduction="mean")

                correct = (better_score > worse_score).cpu().tolist()

                for i, et in enumerate(event_types):
                    if et not in per_event_type_correct:
                        per_event_type_correct[et] = 0
                        per_event_type_total[et] = 0
                    per_event_type_total[et] += 1
                    if correct[i]:
                        per_event_type_correct[et] += 1

    joint.train()

    per_event_type_acc = {}
    for et in per_event_type_total:
        t = per_event_type_total[et]
        per_event_type_acc[et] = per_event_type_correct[et] / t if t > 0 else 0.0

    n = max(total_samples, 1)
    return {
        "count": total_samples,
        "loss": total_loss / n,
        "pairwise": total_pairwise / n,
        "regression": total_regression / n,
        "rank_acc": total_rank_acc / n,
        "mean_score_diff": total_mean_score_diff / n,
        "per_event_type": per_event_type_acc,
    }


def _evaluate_count_head(
    joint: JointRetentionPolicy,
    dataset: FifoCountDataset,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Evaluate count head on a dataset (no gradients).

    Returns dict with: count, loss, accuracy, mean_abs_count_error,
    majority_accuracy, target_distribution, prediction_distribution.
    """
    if dataset is None or len(dataset) == 0:
        return {
            "count": 0,
            "loss": 0.0,
            "accuracy": 0.0,
            "mean_abs_count_error": 0.0,
            "majority_accuracy": 0.0,
            "target_distribution": {},
            "prediction_distribution": {},
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fifo_count_samples,
    )

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_abs_error = 0.0
    target_dist: dict[int, int] = {}
    pred_dist: dict[int, int] = {}
    candidates_tensor = joint.count_head.candidates

    # Compute majority class from dataset
    target_counts: dict[int, int] = {}
    for sample in dataset.samples:
        tkc = int(sample.get("target_keep_count", 0))
        target_counts[tkc] = target_counts.get(tkc, 0) + 1
    if target_counts:
        majority_class = max(target_counts, key=target_counts.get)
        majority_total = target_counts[majority_class]
    else:
        majority_class = 0
        majority_total = 0

    joint.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            logits = joint.forward_count(
                batch["score_state"],
                batch["metadata_features"],
                layer_id=batch["layer_id"],
                token_mask=batch["token_mask"],
            )
            loss_tensor = F.cross_entropy(logits, batch["target"])
            n = batch["target"].shape[0]
            total_loss += float(loss_tensor.detach().cpu()) * n
            total_samples += n

            preds = logits.argmax(dim=-1)
            total_correct += int((preds == batch["target"]).sum().cpu().item())

            pred_counts = candidates_tensor[preds].float()
            target_c = candidates_tensor[batch["target"]].float()
            total_abs_error += float((pred_counts - target_c).abs().sum().cpu().item())

            for p in preds.cpu().tolist():
                pred_dist[p] = pred_dist.get(p, 0) + 1
            for t in batch["target"].cpu().tolist():
                target_dist[t] = target_dist.get(t, 0) + 1

    joint.train()

    n = max(total_samples, 1)
    majority_correct = sum(
        1 for t in target_dist
        if candidates_tensor[t].item() == candidates_tensor[majority_class].item()
        for _ in range(target_dist[t])
    )
    # Simpler: count how many targets are the majority class
    majority_correct = target_dist.get(
        next((i for i, c in enumerate(candidates_tensor.tolist()) if c == candidates_tensor[majority_class].item()), 0),
        0,
    )

    return {
        "count": total_samples,
        "loss": total_loss / n,
        "accuracy": total_correct / n,
        "mean_abs_count_error": total_abs_error / n,
        "majority_accuracy": majority_total / n if n > 0 else 0.0,
        "target_distribution": {str(k): v for k, v in target_dist.items()},
        "prediction_distribution": {str(k): v for k, v in pred_dist.items()},
    }


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
    fifo_token_pair_mode: str = "same_keep_count",
    token_score_mode: str = "delta_mean",
    pair_sampling_seed: int = 0,
    max_pairs_per_event: int = 64,
    min_loss_gap_by_event_type: dict[str, float] | None = None,
    max_loss_gap: float | None = None,
    min_count_loss_gap: float = 0.0,
    device: str = "cpu",
    save_best: bool = False,
    best_metric: str = "token.rank_acc",
    best_output: str | None = None,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
    deploy_count_head: str = "auto",
    deploy_count_head_min_delta: float = 0.0,
) -> None:
    """Train JointRetentionPolicy and save checkpoint.

    This function contains the core training loop so it can be called
    from tests without going through argparse.
    """
    count_candidates = list(count_candidates)
    torch_device = torch.device(device)

    # Load events and split
    all_events = load_oracle_events(oracle_shards)
    train_events, val_events = split_oracle_events(
        all_events,
        val_fraction=val_fraction,
        split_key=split_key,
        seed=split_seed,
    )

    # Build token-ranking dataset from train events
    token_dataset = CounterfactualOracleDataset.from_events(
        train_events,
        min_loss_gap=min_loss_gap,
        fifo_token_pair_mode=fifo_token_pair_mode,
        max_pairs_per_event=max_pairs_per_event,
        pair_sampling_seed=pair_sampling_seed,
        min_loss_gap_by_event_type=min_loss_gap_by_event_type,
        max_loss_gap=max_loss_gap,
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
        min_count_loss_gap=min_count_loss_gap,
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

    # Build validation datasets
    val_token_dataset = CounterfactualOracleDataset.from_events(
        val_events,
        min_loss_gap=min_loss_gap,
        fifo_token_pair_mode=fifo_token_pair_mode,
        max_pairs_per_event=max_pairs_per_event,
        pair_sampling_seed=pair_sampling_seed,
        min_loss_gap_by_event_type=min_loss_gap_by_event_type,
        max_loss_gap=max_loss_gap,
    ) if val_events else None

    val_count_dataset = FifoCountDataset.from_events(
        val_events,
        count_candidates=count_candidates,
        label_reduction=count_label_reduction,
        min_count_loss_gap=min_count_loss_gap,
    ) if val_events else None

    # Print dataset summaries
    train_token_summary = summarize_oracle_pair_samples(token_dataset.samples)
    print(f"token_dataset train {train_token_summary}", flush=True)
    if val_token_dataset and len(val_token_dataset) > 0:
        val_token_summary = summarize_oracle_pair_samples(val_token_dataset.samples)
        print(f"token_dataset val {val_token_summary}", flush=True)
    else:
        val_token_summary = summarize_oracle_pair_samples([])
        print(f"token_dataset val {val_token_summary}", flush=True)

    train_count_summary = summarize_fifo_count_samples(count_dataset.samples)
    print(f"count_dataset train {train_count_summary}", flush=True)
    if val_count_dataset and len(val_count_dataset) > 0:
        val_count_summary = summarize_fifo_count_samples(val_count_dataset.samples)
        print(f"count_dataset val {val_count_summary}", flush=True)
    else:
        val_count_summary = summarize_fifo_count_samples([])
        print(f"count_dataset val {val_count_summary}", flush=True)

    dataset_stats = {
        "train_token": train_token_summary,
        "val_token": val_token_summary,
        "train_count": train_count_summary,
        "val_count": val_count_summary,
    }

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
    has_validation = val_fraction > 0.0 and val_events is not None and len(val_events) > 0

    # Best-checkpoint tracking
    best_value = float("-inf")
    best_epoch = None
    best_validation_metrics: dict = {"token": {}, "count": {}}
    epochs_without_improvement = 0
    final_epoch = 0

    # Projection state (loaded once)
    projection_state = load_score_state_projection_state(score_state_proj_checkpoint)
    if not projection_state:
        projection_state = load_score_state_projection_state_from_oracle_shards(oracle_shards)

    training_options = {
        "oracle_shards": oracle_shards,
        "output": output,
        "score_state_proj_checkpoint": score_state_proj_checkpoint,
        "score_state_dim": score_state_dim,
        "metadata_dim": metadata_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "count_candidates": count_candidates,
        "count_head_arch": count_head_arch,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "regression_weight": regression_weight,
        "min_loss_gap": min_loss_gap,
        "count_loss_weight": count_loss_weight,
        "count_label_reduction": count_label_reduction,
        "count_repeat_factor": count_repeat_factor,
        "val_fraction": val_fraction,
        "split_key": split_key,
        "split_seed": split_seed,
        "fifo_token_pair_mode": fifo_token_pair_mode,
        "token_score_mode": token_score_mode,
        "pair_sampling_seed": pair_sampling_seed,
        "max_pairs_per_event": max_pairs_per_event,
        "min_loss_gap_by_event_type": min_loss_gap_by_event_type,
        "max_loss_gap": max_loss_gap,
        "min_count_loss_gap": min_count_loss_gap,
        "device": device,
        "save_best": save_best,
        "best_metric": best_metric,
        "best_output": best_output,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "deploy_count_head": deploy_count_head,
        "deploy_count_head_min_delta": deploy_count_head_min_delta,
    }

    # Determine effective early stopping (disabled when no validation)
    effective_early_stop_patience = early_stop_patience if has_validation else None

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
                score_mode=token_score_mode,
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

        final_epoch = epoch

        # --- Per-epoch validation (only if we have validation data) ---
        if has_validation:
            token_validation_metrics = _evaluate_token_ranking(
                joint, val_token_dataset, batch_size, torch_device,
                score_mode=token_score_mode,
            )
            count_validation_metrics = _evaluate_count_head(
                joint, val_count_dataset, batch_size, torch_device,
            )

            print(
                f"validation token: count={token_validation_metrics['count']} "
                f"loss={token_validation_metrics['loss']:.6f} "
                f"rank_acc={token_validation_metrics['rank_acc']:.4f}",
                flush=True,
            )
            print(
                f"validation count: count={count_validation_metrics['count']} "
                f"loss={count_validation_metrics['loss']:.6f} "
                f"accuracy={count_validation_metrics['accuracy']:.4f} "
                f"majority_acc={count_validation_metrics['majority_accuracy']:.4f}",
                flush=True,
            )

            current_validation_metrics = {
                "token": token_validation_metrics,
                "count": count_validation_metrics,
            }

            # Check for improvement
            current_value = _select_best_metric(
                token_validation_metrics, count_validation_metrics, best_metric,
            )
            if current_value > best_value + early_stop_min_delta:
                best_value = current_value
                best_epoch = epoch
                best_validation_metrics = copy.deepcopy(current_validation_metrics)
                epochs_without_improvement = 0
                print(
                    f"best_improved epoch={epoch} {best_metric}={best_value:.6f}",
                    flush=True,
                )

                # Save .best.pt on improvement
                if save_best:
                    best_path = Path(best_output) if best_output else Path(output).with_suffix(".best.pt")
                    best_info = {
                        "best_metric": best_metric,
                        "best_metric_value": best_value,
                        "best_epoch": best_epoch,
                        "final_epoch": final_epoch,
                        "has_validation": has_validation,
                        "best_selection_reason": "metric_improved",
                        "best_validation_metrics": copy.deepcopy(best_validation_metrics),
                    }
                    _save_joint_checkpoint(
                        joint,
                        best_path,
                        checkpoint_role="best",
                        count_head_trained=count_head_trained,
                        count_head_arch=count_head_arch,
                        num_layers=num_layers,
                        count_candidates=count_candidates,
                        score_state_dim=score_state_dim,
                        metadata_dim=metadata_dim,
                        hidden_dim=hidden_dim,
                        score_state_proj_checkpoint=score_state_proj_checkpoint,
                        projection_state=projection_state,
                        dataset_stats=dataset_stats,
                        training_options=training_options,
                        validation_metrics=copy.deepcopy(current_validation_metrics),
                        best_info=best_info,
                        deploy_count_head=deploy_count_head,
                        deploy_count_head_min_delta=deploy_count_head_min_delta,
                    )
            else:
                epochs_without_improvement += 1

            # Early stopping
            if effective_early_stop_patience is not None and epochs_without_improvement >= effective_early_stop_patience:
                print(
                    f"early_stop epoch={epoch} best_epoch={best_epoch} best_metric_value={best_value:.6f}",
                    flush=True,
                )
                break

    # --- Final validation (if not already done per-epoch, or no validation) ---
    if not has_validation:
        token_validation_metrics = _evaluate_token_ranking(
            joint, val_token_dataset, batch_size, torch_device,
            score_mode=token_score_mode,
        )
        count_validation_metrics = _evaluate_count_head(
            joint, val_count_dataset, batch_size, torch_device,
        )

    current_validation_metrics = {
        "token": token_validation_metrics,
        "count": count_validation_metrics,
    }

    # Build best_info for final checkpoint
    if has_validation:
        best_selection_reason = "metric_improved" if best_epoch is not None else "no_improvement"
    else:
        best_selection_reason = "no_validation"
        best_value = None
        # For no-validation: save .best.pt at the final state
        best_epoch = final_epoch
        best_validation_metrics = copy.deepcopy(current_validation_metrics)

    best_info = {
        "best_metric": best_metric,
        "best_metric_value": best_value,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "has_validation": has_validation,
        "best_selection_reason": best_selection_reason,
        "best_validation_metrics": copy.deepcopy(best_validation_metrics),
    }

    # Save no-validation .best.pt if needed
    if save_best and not has_validation:
        best_path = Path(best_output) if best_output else Path(output).with_suffix(".best.pt")
        _save_joint_checkpoint(
            joint,
            best_path,
            checkpoint_role="best",
            count_head_trained=count_head_trained,
            count_head_arch=count_head_arch,
            num_layers=num_layers,
            count_candidates=count_candidates,
            score_state_dim=score_state_dim,
            metadata_dim=metadata_dim,
            hidden_dim=hidden_dim,
            score_state_proj_checkpoint=score_state_proj_checkpoint,
            projection_state=projection_state,
            dataset_stats=dataset_stats,
            training_options=training_options,
            validation_metrics=copy.deepcopy(current_validation_metrics),
            best_info={
                "best_metric": best_metric,
                "best_metric_value": None,
                "best_epoch": final_epoch,
                "final_epoch": final_epoch,
                "has_validation": False,
                "best_selection_reason": "no_validation",
                "best_validation_metrics": {"token": {}, "count": {}},
            },
            deploy_count_head=deploy_count_head,
            deploy_count_head_min_delta=deploy_count_head_min_delta,
        )

    # --- Save final checkpoint ---
    output_path = Path(output)
    _save_joint_checkpoint(
        joint,
        output_path,
        checkpoint_role="final",
        count_head_trained=count_head_trained,
        count_head_arch=count_head_arch,
        num_layers=num_layers,
        count_candidates=count_candidates,
        score_state_dim=score_state_dim,
        metadata_dim=metadata_dim,
        hidden_dim=hidden_dim,
        score_state_proj_checkpoint=score_state_proj_checkpoint,
        projection_state=projection_state,
        dataset_stats=dataset_stats,
        training_options=training_options,
        validation_metrics=current_validation_metrics,
        best_info=best_info,
        deploy_count_head=deploy_count_head,
        deploy_count_head_min_delta=deploy_count_head_min_delta,
    )
    print(f"total_steps={step}", flush=True)


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
        fifo_token_pair_mode=args.fifo_token_pair_mode,
        token_score_mode=args.token_score_mode,
        pair_sampling_seed=args.pair_sampling_seed,
        max_pairs_per_event=args.max_pairs_per_event,
        min_loss_gap_by_event_type=args.min_loss_gap_by_event_type,
        max_loss_gap=args.max_loss_gap,
        min_count_loss_gap=args.min_count_loss_gap,
        device=args.device,
        save_best=args.save_best,
        best_metric=args.best_metric,
        best_output=args.best_output,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        deploy_count_head=args.deploy_count_head,
        deploy_count_head_min_delta=args.deploy_count_head_min_delta,
    )


if __name__ == "__main__":
    main()
