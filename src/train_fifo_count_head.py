"""Train FifoCountHead from counterfactual retention oracle shards.

Classification head that predicts the best FIFO flush count from pooled
token representations.  Uses FifoCountDataset and FifoCountHead to train
a discrete classifier over a fixed candidate set (e.g. 0, 8, 16, 32, 64, 128).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ovggt.layers.count_head import FifoCountHead
from ovggt.layers.token_scorer import TOKEN_METADATA_FEATURE_DIM
from ovggt.training.token_oracle_dataset import (
    FifoCountDataset,
    collate_fifo_count_samples,
    load_oracle_events,
    split_oracle_events,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML config; CLI options override YAML values")
    parser.add_argument("--oracle-shards", nargs="+", help="Path(s) to .pt oracle shards")
    parser.add_argument("--output", help="Checkpoint output path")
    parser.add_argument("--count-candidates", nargs="+", type=int, help="Candidate count values")
    parser.add_argument("--score-state-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--label-reduction", choices=["min", "mean"], help="Label reduction strategy")
    parser.add_argument("--val-fraction", type=float, help="Fraction of samples for validation")
    parser.add_argument("--split-seed", type=int, help="Seed for train/val split")
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    defaults = {
        "oracle_shards": None,
        "output": None,
        "count_candidates": [0, 8, 16, 32, 64, 128],
        "score_state_dim": 128,
        "hidden_dim": 128,
        "num_layers": 24,
        "batch_size": 64,
        "epochs": 10,
        "lr": 1e-4,
        "label_reduction": "min",
        "val_fraction": 0.1,
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


def build_ovggt_count_head_state_dict(count_head_state: dict) -> dict:
    """Map FifoCountHead state dict keys to the deploy format.

    Prepends ``aggregator.count_head.`` to each key so the checkpoint can be
    loaded directly into the full OVGGT model.
    """
    return {
        f"aggregator.count_head.{key}": value.detach().cpu().clone()
        for key, value in count_head_state.items()
    }


def train_fifo_count_head(
    oracle_shards: list[str],
    output: str,
    count_candidates: Sequence[int] = (0, 8, 16, 32, 64, 128),
    score_state_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 24,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-4,
    label_reduction: str = "min",
    val_fraction: float = 0.0,
    split_seed: int = 0,
    device: str = "cpu",
) -> None:
    """Train FifoCountHead and save checkpoint.

    This function contains the core training loop so it can be called
    from tests without going through argparse.
    """
    count_candidates = list(count_candidates)
    all_events = load_oracle_events(oracle_shards)
    if float(val_fraction) > 0.0:
        train_events, val_events = split_oracle_events(
            all_events,
            val_fraction=val_fraction,
            split_key="event_id_hash",
            seed=split_seed,
        )
        if not train_events:
            train_events, val_events = all_events, []
    else:
        train_events, val_events = all_events, []

    train_dataset = FifoCountDataset.from_events(
        train_events,
        count_candidates=count_candidates,
        label_reduction=label_reduction,
    )
    val_dataset = FifoCountDataset.from_events(
        val_events,
        count_candidates=count_candidates,
        label_reduction=label_reduction,
    ) if val_events else None
    if len(train_dataset) == 0:
        raise RuntimeError(f"No fifo_topk samples found in oracle shards")

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fifo_count_samples,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fifo_count_samples,
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )
    torch_device = torch.device(device)
    count_head = FifoCountHead(
        score_state_dim=score_state_dim,
        metadata_dim=TOKEN_METADATA_FEATURE_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        candidates=count_candidates,
    ).to(torch_device)

    optimizer = torch.optim.AdamW(count_head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    num_candidates = len(count_candidates)
    step = 0
    validation_metrics = {"count": 0}
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_abs_error = 0.0

        # Track class histograms
        label_hist = [0] * num_candidates
        pred_hist = [0] * num_candidates

        for batch in loader:
            batch = _move_batch_to_device(batch, torch_device)
            logits = count_head(
                batch["score_state"],
                batch["metadata_features"],
                layer_id=batch["layer_id"],
                token_mask=batch["token_mask"],
            )  # [B, num_candidates]

            target = batch["target"]  # [B]
            loss = loss_fn(logits, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Metrics
            batch_size_actual = target.shape[0]
            preds = logits.argmax(dim=-1)  # [B]

            epoch_loss += float(loss.detach().cpu()) * batch_size_actual
            epoch_correct += int((preds == target).sum().detach().cpu().item())
            epoch_total += batch_size_actual

            # Mean absolute count error
            pred_counts = count_head.candidates[preds].float()
            target_counts = count_head.candidates[target].float()
            epoch_abs_error += float((pred_counts - target_counts).abs().sum().detach().cpu().item())

            # Class histograms
            for t in target.tolist():
                label_hist[int(t)] += 1
            for p in preds.tolist():
                pred_hist[int(p)] += 1

            if step % 50 == 0:
                print(
                    f"epoch={epoch} step={step} loss={float(loss.detach().cpu()):.6f}",
                    flush=True,
                )
            step += 1

        avg_loss = epoch_loss / max(epoch_total, 1)
        accuracy = epoch_correct / max(epoch_total, 1)
        mace = epoch_abs_error / max(epoch_total, 1)
        print(
            f"epoch_summary={epoch} batches={step} "
            f"loss={avg_loss:.6f} accuracy={accuracy:.4f} "
            f"mean_abs_count_error={mace:.2f}",
            flush=True,
        )
        print(
            f"  label_hist={label_hist} pred_hist={pred_hist}",
            flush=True,
        )
        if val_loader is not None:
            validation_metrics = _evaluate_count_head(
                count_head=count_head,
                loader=val_loader,
                loss_fn=loss_fn,
                device=torch_device,
            )
            print(
                f"validation_summary={epoch} samples={validation_metrics['count']} "
                f"loss={validation_metrics['loss']:.6f} "
                f"accuracy={validation_metrics['accuracy']:.4f} "
                f"mean_abs_count_error={validation_metrics['mean_abs_count_error']:.2f}",
                flush=True,
            )

    # Save checkpoint
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "count_head": count_head.state_dict(),
            "model": build_ovggt_count_head_state_dict(count_head.state_dict()),
            "score_state_dim": score_state_dim,
            "metadata_dim": TOKEN_METADATA_FEATURE_DIM,
            "num_layers": num_layers,
            "count_candidates": count_candidates,
            "hidden_dim": hidden_dim,
            "count_head_arch": "pooled_v1",
            "train_sample_count": len(train_dataset),
            "val_sample_count": 0 if val_dataset is None else len(val_dataset),
            "validation_metrics": validation_metrics,
        },
        output_path,
    )
    print(f"saved_checkpoint={output_path} total_steps={step}", flush=True)


@torch.no_grad()
def _evaluate_count_head(
    count_head: FifoCountHead,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
) -> dict:
    count_head.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_abs_error = 0.0
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        logits = count_head(
            batch["score_state"],
            batch["metadata_features"],
            layer_id=batch["layer_id"],
            token_mask=batch["token_mask"],
        )
        target = batch["target"]
        loss = loss_fn(logits, target)
        batch_count = int(target.shape[0])
        preds = logits.argmax(dim=-1)
        pred_counts = count_head.candidates[preds].float()
        target_counts = count_head.candidates[target].float()
        total_loss += float(loss.detach().cpu()) * batch_count
        total_correct += int((preds == target).sum().detach().cpu().item())
        total_abs_error += float((pred_counts - target_counts).abs().sum().detach().cpu().item())
        total_count += batch_count
    count_head.train()
    return {
        "count": total_count,
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "mean_abs_count_error": total_abs_error / max(total_count, 1),
    }


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def main() -> None:
    args = parse_args()
    train_fifo_count_head(
        oracle_shards=args.oracle_shards,
        output=args.output,
        count_candidates=args.count_candidates,
        score_state_dim=args.score_state_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        label_reduction=args.label_reduction,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
