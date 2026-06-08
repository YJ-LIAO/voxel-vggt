"""Small-data overfit sanity check for JointRetentionPolicy.

Selects a small number of high-margin token samples and count samples from
oracle shards, then overfits a tiny JointRetentionPolicy on them.  This
verifies that model capacity is not the bottleneck -- if the model cannot
memorize a handful of easy examples, the architecture itself is broken.

Output markers (for CI / grep-ability):
    TOKEN_DATA   -- token sample selection summary
    TOKEN_OVERFIT -- per-step training progress
    TOKEN_DONE   -- final token overfit accuracy
    COUNT_DATA   -- count sample selection summary
    COUNT_OVERFIT -- per-step training progress
    COUNT_DONE   -- final count overfit accuracy
    SANITY_RESULT token_pass=<bool> count_pass=<bool>

Exit codes:
    0 -- both token and count pass their thresholds
    2 -- either one fails

Usage:
    env PYTHONPATH=src python tools/overfit_joint_retention_sanity.py \\
        --oracle-shards <paths...> \\
        --token-samples 128 \\
        --count-samples-per-class 8 \\
        --min-loss-gap 0.03 \\
        --device cuda \\
        --max-token-steps 150 \\
        --max-count-steps 400
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Ensure src/ is importable when run as a standalone script
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
from ovggt.layers.retention_policy import JointRetentionPolicy
from ovggt.training.token_oracle_dataset import (
    CounterfactualOracleDataset,
    FifoCountDataset,
    collate_fifo_count_samples,
    collate_oracle_pairs,
    load_oracle_events,
    token_oracle_ranking_loss,
)


# ===================================================================== #
# Public helper (imported by tests)
# ===================================================================== #

def select_high_margin_token_samples(
    samples: list[dict],
    n: int,
) -> list[dict]:
    """Return the *n* samples with the highest ``target_margin``, sorted
    in descending order of margin.

    Parameters
    ----------
    samples : list[dict]
        Each dict must contain a ``"target_margin"`` key with a float value.
    n : int
        Maximum number of samples to return.  If *n* exceeds the number of
        available samples, all are returned (still sorted descending).

    Returns
    -------
    list[dict]
        Subset of *samples* with highest margins, descending.
    """
    if not samples:
        return []
    sorted_samples = sorted(
        samples,
        key=lambda s: float(s["target_margin"]),
        reverse=True,
    )
    return sorted_samples[:n]


# ===================================================================== #
# Internal helpers
# ===================================================================== #

def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def _overfit_token(
    joint: JointRetentionPolicy,
    token_samples: list[dict],
    *,
    max_steps: int,
    lr: float,
    device: torch.device,
    score_mode: str = "delta_mean",
    pass_threshold: float = 0.98,
) -> bool:
    """Overfit the token-ranking head on a tiny dataset.

    Returns True if final rank accuracy >= *pass_threshold*.
    """
    if not token_samples:
        print("TOKEN_DATA n=0 (no token samples)", flush=True)
        print("TOKEN_DONE rank_acc=N/A", flush=True)
        return False

    print(f"TOKEN_DATA n={len(token_samples)}", flush=True)

    # Build a minimal dataset wrapper from the selected samples
    class _TinyTokenDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    dataset = _TinyTokenDataset(token_samples)
    loader = DataLoader(
        dataset,
        batch_size=min(32, len(dataset)),
        shuffle=True,
        collate_fn=collate_oracle_pairs,
    )

    optimizer = torch.optim.AdamW(joint.parameters(), lr=lr)

    final_acc = 0.0
    for step in range(1, max_steps + 1):
        joint.train()
        epoch_acc = 0.0
        epoch_count = 0
        for batch in loader:
            batch = _move_batch(batch, device)
            logits = joint.forward_token(
                batch["score_state"],
                batch["metadata_features"],
                layer_id=batch["layer_id"],
            )
            loss, details = token_oracle_ranking_loss(
                logits, batch,
                regression_weight=0.1,
                score_mode=score_mode,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_acc += details["rank_acc"] * logits.shape[0]
            epoch_count += logits.shape[0]

        final_acc = epoch_acc / max(epoch_count, 1)
        if step % 25 == 0 or step == max_steps:
            print(f"TOKEN_OVERFIT step={step} rank_acc={final_acc:.4f}", flush=True)

    passed = final_acc >= pass_threshold
    print(f"TOKEN_DONE rank_acc={final_acc:.4f} pass={passed}", flush=True)
    return passed


def _overfit_count(
    joint: JointRetentionPolicy,
    count_samples: list[dict],
    *,
    max_steps: int,
    lr: float,
    device: torch.device,
    pass_threshold: float = 0.90,
) -> bool:
    """Overfit the count head on a tiny dataset.

    Returns True if final accuracy >= *pass_threshold*.
    """
    if not count_samples:
        print("COUNT_DATA n=0 (no count samples)", flush=True)
        print("COUNT_DONE accuracy=N/A", flush=True)
        return False

    print(f"COUNT_DATA n={len(count_samples)}", flush=True)

    class _TinyCountDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    dataset = _TinyCountDataset(count_samples)
    loader = DataLoader(
        dataset,
        batch_size=min(32, len(dataset)),
        shuffle=True,
        collate_fn=collate_fifo_count_samples,
    )

    optimizer = torch.optim.AdamW(joint.parameters(), lr=lr)

    final_acc = 0.0
    for step in range(1, max_steps + 1):
        joint.train()
        epoch_correct = 0
        epoch_total = 0
        for batch in loader:
            batch = _move_batch(batch, device)
            logits = joint.forward_count(
                batch["score_state"],
                batch["metadata_features"],
                layer_id=batch["layer_id"],
                token_mask=batch["token_mask"],
            )
            loss = F.cross_entropy(logits, batch["target"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=-1)
            epoch_correct += int((preds == batch["target"]).sum().cpu().item())
            epoch_total += batch["target"].shape[0]

        final_acc = epoch_correct / max(epoch_total, 1)
        if step % 50 == 0 or step == max_steps:
            print(f"COUNT_OVERFIT step={step} accuracy={final_acc:.4f}", flush=True)

    passed = final_acc >= pass_threshold
    print(f"COUNT_DONE accuracy={final_acc:.4f} pass={passed}", flush=True)
    return passed


# ===================================================================== #
# Sample selection from oracle shards
# ===================================================================== #

def _select_count_samples(
    events: list[dict],
    count_candidates: list[int],
    samples_per_class: int,
    label_reduction: str,
    min_count_loss_gap: float,
) -> list[dict]:
    """Select up to *samples_per_class* count samples per target class.

    Prefers samples with the highest count_loss_gap (most confident).
    """
    dataset = FifoCountDataset.from_events(
        events,
        count_candidates=count_candidates,
        label_reduction=label_reduction,
        min_count_loss_gap=min_count_loss_gap,
    )

    # Group by target class
    by_target: dict[int, list[dict]] = {}
    for sample in dataset.samples:
        t = int(sample["target"])
        by_target.setdefault(t, []).append(sample)

    selected: list[dict] = []
    for target_idx in sorted(by_target):
        group = sorted(
            by_target[target_idx],
            key=lambda s: float(s.get("count_loss_gap", 0.0)),
            reverse=True,
        )
        selected.extend(group[:samples_per_class])

    return selected


# ===================================================================== #
# CLI / main
# ===================================================================== #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-shards", nargs="+", required=True,
                        help="Path(s) to .pt oracle shard files")
    parser.add_argument("--token-samples", type=int, default=64,
                        help="Number of high-margin token samples to select")
    parser.add_argument("--count-samples-per-class", type=int, default=8,
                        help="Number of count samples per target class")
    parser.add_argument("--min-loss-gap", type=float, default=0.03,
                        help="Minimum loss gap for token pair filtering")
    parser.add_argument("--device", default="cpu",
                        help="Torch device (cpu or cuda)")
    parser.add_argument("--max-token-steps", type=int, default=300,
                        help="Maximum overfit steps for token head")
    parser.add_argument("--max-count-steps", type=int, default=400,
                        help="Maximum overfit steps for count head")
    parser.add_argument("--token-pass-threshold", type=float, default=0.98,
                        help="Token rank accuracy threshold to pass")
    parser.add_argument("--count-pass-threshold", type=float, default=0.90,
                        help="Count accuracy threshold to pass")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for overfit")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Tiny hidden dim for sanity model")
    parser.add_argument("--num-layers", type=int, default=8,
                        help="Number of layers for sanity model")
    parser.add_argument("--count-candidates", nargs="+", type=int,
                        default=[0, 8, 16, 32, 64, 128],
                        help="Count candidate values")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the overfit sanity check and return exit code."""
    args = parse_args(argv)

    device = torch.device(args.device)

    # Load all events from shards
    all_events = load_oracle_events(args.oracle_shards)
    if not all_events:
        print("ERROR: No events loaded from oracle shards", file=sys.stderr, flush=True)
        return 2

    # Detect dimensions from first event
    first_event = all_events[0]
    score_state = torch.as_tensor(first_event["score_state"])
    if score_state.dim() == 3 and score_state.shape[0] == 1:
        score_state = score_state[0]
    score_state_dim = score_state.shape[1]
    metadata_dim = TOKEN_METADATA_FEATURE_DIM

    print(f"Loaded {len(all_events)} events from {len(args.oracle_shards)} shard(s)", flush=True)
    print(f"score_state_dim={score_state_dim}  metadata_dim={metadata_dim}", flush=True)

    # ------------------------------------------------------------------
    # Build tiny joint model
    # ------------------------------------------------------------------
    joint = JointRetentionPolicy(
        score_state_dim=score_state_dim,
        metadata_dim=metadata_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        count_candidates=args.count_candidates,
    ).to(device)

    total_params = sum(p.numel() for p in joint.parameters())
    print(f"Model params: {total_params:,}  (hidden_dim={args.hidden_dim}, num_layers={args.num_layers})", flush=True)

    # ------------------------------------------------------------------
    # Token overfit
    # ------------------------------------------------------------------
    token_dataset = CounterfactualOracleDataset.from_events(
        all_events,
        min_loss_gap=args.min_loss_gap,
    )
    token_samples = select_high_margin_token_samples(
        token_dataset.samples, args.token_samples
    )
    token_pass = _overfit_token(
        joint,
        token_samples,
        max_steps=args.max_token_steps,
        lr=args.lr,
        device=device,
        score_mode="delta_mean",
        pass_threshold=args.token_pass_threshold,
    )

    # ------------------------------------------------------------------
    # Count overfit
    # ------------------------------------------------------------------
    count_samples = _select_count_samples(
        all_events,
        count_candidates=args.count_candidates,
        samples_per_class=args.count_samples_per_class,
        label_reduction="min",
        min_count_loss_gap=0.0,  # no filtering for sanity
    )
    count_pass = _overfit_count(
        joint,
        count_samples,
        max_steps=args.max_count_steps,
        lr=args.lr,
        device=device,
        pass_threshold=args.count_pass_threshold,
    )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    print(f"SANITY_RESULT token_pass={token_pass} count_pass={count_pass}", flush=True)

    if token_pass and count_pass:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
