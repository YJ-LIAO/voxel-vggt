#!/usr/bin/env python
"""Resume shard 007 oracle collection from batch 26 onwards, merging with existing events."""

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
    collect_oracle_events_from_sequence,
    save_oracle_shard,
    build_frozen_frontend_model_from_config,
    build_frozen_teacher_from_config,
    build_frontend_oracle_dataloader,
    extract_score_state_projection_state,
    OracleShardFlusher,
    TASK_WEIGHTS,
    load_frontend_oracle_config,
    build_sequence_provenance,
    format_provenance_log_summary,
    move_batch_to_device,
    normalize_batch_images,
    default_oracle_log,
)
import copy


def main():
    existing_path = "/Train/lyj/workspace/OVGGT/checkpoints/token_oracle_budget200k_v96_fixed/oracle_shard_007_backup.pt"
    output_path = "/Train/lyj/workspace/OVGGT/checkpoints/token_oracle_budget200k_v96_fixed/oracle_shard_007.pt"
    log_fn = default_oracle_log

    SEED = 7
    SKIP_BATCHES = 25  # 24 collected + 1 stuck batch skipped
    MAX_BATCHES = 128
    MAX_EVENTS = 4096
    MAX_FETCH_ERRORS = 256

    # Load existing events
    log_fn(f"loading existing shard from {existing_path}")
    existing_shard = torch.load(existing_path, map_location="cpu", weights_only=False)
    existing_events = existing_shard.get("events", [])
    log_fn(f"loaded {len(existing_events)} existing events from {SKIP_BATCHES - 1} batches")

    # Build config
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

    # Build the SAME dataloader with same seed
    data_loader = build_frontend_oracle_dataloader(
        cfg,
        dataset_key="train_dataset",
        batch_size=1,
        num_workers=0,
        drop_last=False,
        seed=SEED,
        log_fn=log_fn,
    )

    # Skip the first SKIP_BATCHES batches
    log_fn(f"skipping first {SKIP_BATCHES} batches to resume...")
    loader_iter = iter(data_loader)
    skipped = 0
    fetch_errors = 0
    while skipped < SKIP_BATCHES:
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

    # Build flusher
    remaining_batches = MAX_BATCHES - SKIP_BATCHES
    remaining_events = MAX_EVENTS - len(existing_events)
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

    # Collect remaining batches
    events = list(existing_events)
    batch_idx = SKIP_BATCHES
    max_events_int = MAX_EVENTS
    max_fetch_errors_int = MAX_FETCH_ERRORS

    log_fn(f"starting collection from batch {SKIP_BATCHES + 1}/{MAX_BATCHES}")

    while batch_idx < MAX_BATCHES and len(events) < max_events_int:
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

        log_fn(f"batch {batch_idx + 1}/{MAX_BATCHES} start: events_so_far={len(events)}")

        sequence_provenance = build_sequence_provenance(
            batch, batch_index=batch_idx, dataset_key="train_dataset",
        )
        log_fn(
            f"batch {batch_idx + 1}/{MAX_BATCHES} provenance: "
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

        def on_event_collected(event: dict, _cb=collected_batch_events, _ev=events) -> None:
            _cb.append(event)
            _ev.append(event)
            flusher.maybe_flush(
                _ev,
                batch_idx=None,
                force=False,
                reason=f"batch {batch_idx + 1}/{MAX_BATCHES} event {len(_cb)} collected",
            )

        batch_events = collect_oracle_events_from_sequence(
            model=model,
            frames=batch,
            device=torch.device("cuda"),
            max_events=remaining_ev,
            num_samples=8,
            oracle_window=4,
            seed=SEED + batch_idx * 1009,
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
            f"batch {batch_idx}/{MAX_BATCHES} done: "
            f"batch_events={len(collected_batch_events)} total_events={len(events)}"
        )
        flusher.maybe_flush(
            events,
            batch_idx=batch_idx,
            force=False,
            reason=f"batch {batch_idx}/{MAX_BATCHES} done",
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
        max_batches=MAX_BATCHES,
        max_events=MAX_EVENTS,
        num_samples=8,
        oracle_window=4,
        device="cuda",
        seed=SEED,
        num_views=96,
        subset_replay_batch_size=1,
        layers_per_frame=4,
        max_events_per_sequence=64,
        flush_every_events=4,
        flush_every_batches=1,
        log_every_subsets=2,
        max_fetch_errors=MAX_FETCH_ERRORS,
    ))
    flusher.maybe_flush(events, batch_idx=None, force=True, reason="final")
    log_fn(f"RESUME DONE: total_events={len(events)} batches_processed={batch_idx - SKIP_BATCHES}")


if __name__ == "__main__":
    main()
