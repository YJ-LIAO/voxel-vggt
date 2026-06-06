#!/usr/bin/env python
"""Resume shard 000 oracle collection from batch 8 onwards, merging with existing events."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.training.frontend_oracle_collector import (
    FrontendOracleCollectorConfig,
    collect_oracle_events_from_loader,
    save_oracle_shard,
    build_frozen_frontend_model_from_config,
    build_frozen_teacher_from_config,
    build_frontend_oracle_dataloader,
    extract_score_state_projection_state,
    OracleShardFlusher,
    TASK_WEIGHTS,
    load_frontend_oracle_config,
)
from ovggt.training.frontend_oracle_collector import default_oracle_log


def main():
    existing_path = "/Train/lyj/workspace/OVGGT/checkpoints/token_oracle_budget200k_v96_fixed/oracle_shard_000_backup.pt"
    output_path = "/Train/lyj/workspace/OVGGT/checkpoints/token_oracle_budget200k_v96_fixed/oracle_shard_000.pt"
    log_fn = default_oracle_log

    # Load existing events
    log_fn(f"loading existing shard from {existing_path}")
    existing_shard = torch.load(existing_path, map_location="cpu")
    existing_events = existing_shard.get("events", [])
    skip_batches = 8  # 7 collected + 1 stuck batch skipped
    log_fn(f"loaded {len(existing_events)} existing events from {skip_batches} batches")

    # Build config (same as original shard 000)
    cfg_path = "config/train_frontend_finetune.yaml"
    cfg = load_frontend_oracle_config(cfg_path, num_views=96)
    log_fn("config loaded")

    log_fn("building frozen student model")
    model = build_frozen_frontend_model_from_config(
        cfg, device="cuda", checkpoint_path=None, high_budget=False, log_fn=log_fn,
    )
    log_fn("student model ready")
    teacher = build_frozen_teacher_from_config(
        cfg, device="cuda", checkpoint_path=None, log_fn=log_fn,
    )
    log_fn("teacher model ready")

    num_layers = int(getattr(getattr(model, "aggregator", None), "depth", 0) or 0)

    # Build the SAME dataloader with same seed (seed=0)
    data_loader = build_frontend_oracle_dataloader(
        cfg,
        dataset_key="train_dataset",
        batch_size=1,
        num_workers=0,
        drop_last=False,
        seed=0,  # same as original
        log_fn=log_fn,
    )

    # Skip the first `skip_batches` batches by consuming them
    log_fn(f"skipping first {skip_batches} batches to resume...")
    loader_iter = iter(data_loader)
    skipped = 0
    fetch_errors = 0
    while skipped < skip_batches:
        try:
            _ = next(loader_iter)
            skipped += 1
        except StopIteration:
            log_fn(f"dataloader exhausted after skipping {skipped} batches")
            break
        except Exception as exc:
            fetch_errors += 1
            log_fn(f"skip fetch error {fetch_errors}: {type(exc).__name__}: {exc}")
            continue
    log_fn(f"skipped {skipped} batches ({fetch_errors} fetch errors during skip)")

    # Build flusher that appends to existing events
    remaining_batches = 128 - skip_batches  # 121 batches left
    remaining_events = 4096 - len(existing_events)
    log_fn(
        f"resuming collection: remaining_batches={remaining_batches} "
        f"existing_events={len(existing_events)} remaining_events={remaining_events}"
    )

    base_shard = {
        "format": "ovggt_counterfactual_oracle_v1",
        "task_weights": TASK_WEIGHTS,
        "source_config": str(cfg_path),
        "dataset_key": "train_dataset",
        "score_state_projection_state": extract_score_state_projection_state(model),
    }

    flusher = OracleShardFlusher(
        base_shard=base_shard,
        output_path=output_path,
        flush_every_events=4,
        flush_every_batches=1,
        log_fn=log_fn,
    )

    # Now collect remaining batches manually (mirroring collect_oracle_events_from_loader
    # but starting from the current loader_iter position)
    events = list(existing_events)
    batch_idx = skip_batches
    collected_batch_events: list[dict] = []
    max_events_int = 4096
    max_fetch_errors_int = 256

    log_fn(f"starting collection from batch {skip_batches + 1}/128")

    while batch_idx < 128 and len(events) < max_events_int:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        except Exception as exc:
            fetch_errors += 1
            log_fn(
                f"skipping dataloader batch after fetch error "
                f"{fetch_errors}/{max_fetch_errors_int}: {type(exc).__name__}: {exc}"
            )
            if fetch_errors > max_fetch_errors_int:
                raise RuntimeError(
                    f"Exceeded max_fetch_errors={max_fetch_errors_int}"
                ) from exc
            continue

        log_fn(f"batch {batch_idx + 1}/128 start: events_so_far={len(events)}")

        from ovggt.training.frontend_oracle_collector import (
            build_sequence_provenance,
            format_provenance_log_summary,
            move_batch_to_device,
            normalize_batch_images,
            collect_oracle_events_from_sequence,
        )
        import copy

        sequence_provenance = build_sequence_provenance(
            batch, batch_index=batch_idx, dataset_key="train_dataset",
        )
        log_fn(
            f"batch {batch_idx + 1}/128 provenance: "
            f"{format_provenance_log_summary(sequence_provenance)}"
        )
        batch = move_batch_to_device(
            normalize_batch_images(copy.deepcopy(batch)), torch.device("cuda")
        )

        remaining_ev = max_events_int - len(events)
        max_eps = 64
        if max_eps > 0:
            remaining_ev = min(remaining_ev, max_eps)

        collected_batch_events = []

        def on_event_collected(event: dict) -> None:
            collected_batch_events.append(event)
            events.append(event)
            flusher.maybe_flush(
                events,
                batch_idx=None,
                force=False,
                reason=f"batch {batch_idx + 1}/128 event {len(collected_batch_events)} collected",
            )

        batch_events = collect_oracle_events_from_sequence(
            model=model,
            frames=batch,
            device=torch.device("cuda"),
            max_events=remaining_ev,
            num_samples=8,
            oracle_window=4,
            seed=0 + batch_idx * 1009,
            teacher=teacher,
            event_prefix=f"batch{batch_idx}",
            sequence_provenance=sequence_provenance,
            log_fn=log_fn,
            log_every_subsets=2,
            subset_replay_batch_size=1,
            layers_per_frame=4,
            num_layers=num_layers,
            on_event_collected=on_event_collected,
        )

        batch_idx += 1
        log_fn(
            f"batch {batch_idx}/128 done: "
            f"batch_events={len(collected_batch_events)} total_events={len(events)}"
        )
        flusher.maybe_flush(
            events,
            batch_idx=batch_idx,
            force=False,
            reason=f"batch {batch_idx}/128 done",
        )

    # Final save
    shard = dict(base_shard)
    shard["events"] = events
    shard["partial"] = False
    shard["num_events"] = len(events)
    shard["collector_config"] = asdict(FrontendOracleCollectorConfig(
        config=cfg_path,
        output=output_path,
        dataset_key="train_dataset",
        batch_size=1,
        num_workers=0,
        max_batches=128,
        max_events=4096,
        num_samples=8,
        oracle_window=4,
        device="cuda",
        seed=0,
        num_views=96,
        subset_replay_batch_size=1,
        layers_per_frame=4,
        max_events_per_sequence=64,
        flush_every_events=4,
        flush_every_batches=1,
        log_every_subsets=2,
        max_fetch_errors=256,
    ))
    flusher.maybe_flush(events, batch_idx=None, force=True, reason="final")
    log_fn(f"RESUME DONE: total_events={len(events)} batches_processed={batch_idx - skip_batches}")


if __name__ == "__main__":
    main()
