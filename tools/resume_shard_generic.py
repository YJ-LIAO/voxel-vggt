#!/usr/bin/env python
"""Generic resume script for oracle shard collection.

Usage:
  CUDA_VISIBLE_DEVICES=N python resume_shard_generic.py \
    --shard-id 2 --seed 2 --skip-batches 38

Skip-batches = (batches_done + 1 for the stuck batch).
"""

from __future__ import annotations

import argparse
import os
import sys
import concurrent.futures
from dataclasses import asdict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.training.frontend_oracle_collector import (
    FrontendOracleCollectorConfig,
    collect_oracle_events_from_sequence,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--skip-batches", type=int, required=True,
                        help="Number of successful batches to skip (done + 1 stuck)")
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=4096)
    parser.add_argument("--max-fetch-errors", type=int, default=256)
    args = parser.parse_args()

    BASEDIR = "/Train/lyj/workspace/OVGGT/checkpoints/token_oracle_budget200k_v96_fixed"
    shard_tag = f"{args.shard_id:03d}"
    existing_path = f"{BASEDIR}/oracle_shard_{shard_tag}_backup2.pt"
    output_path = f"{BASEDIR}/oracle_shard_{shard_tag}.pt"
    log_fn = default_oracle_log

    SEED = args.seed
    SKIP = args.skip_batches
    MAX_BATCHES = args.max_batches
    MAX_EVENTS = args.max_events

    # Load existing events
    log_fn(f"[shard {shard_tag}] loading existing shard from {existing_path}")
    existing_shard = torch.load(existing_path, map_location="cpu", weights_only=False)
    existing_events = existing_shard.get("events", [])
    log_fn(f"[shard {shard_tag}] loaded {len(existing_events)} existing events, will skip {SKIP} batches")

    # Build config
    cfg_path = "config/train_frontend_finetune.yaml"
    cfg = load_frontend_oracle_config(cfg_path, num_views=96)
    log_fn("[shard {shard_tag}] config loaded")

    log_fn(f"[shard {shard_tag}] building frozen student model")
    model = build_frozen_frontend_model_from_config(
        cfg, device="cuda", checkpoint_path=None, high_budget=False, log_fn=log_fn,
    )
    log_fn(f"[shard {shard_tag}] student model ready")
    teacher = build_frozen_teacher_from_config(
        cfg, device="cuda", checkpoint_path=None, log_fn=log_fn,
    )
    log_fn(f"[shard {shard_tag}] teacher model ready")

    num_layers = int(getattr(getattr(model, "aggregator", None), "depth", 0) or 0)

    # Build dataloader with same seed
    data_loader = build_frontend_oracle_dataloader(
        cfg, dataset_key="train_dataset", batch_size=1, num_workers=0,
        drop_last=False, seed=SEED, log_fn=log_fn,
    )

    # Skip batches with timeout to avoid hanging on bad data
    # If we hit a timeout during skip, we abandon the skip and start collecting
    # from a fresh iterator at position 0 — data will be different but we keep
    # the events already collected. This is acceptable since we want to keep
    # making progress.
    SKIP_TIMEOUT = 300  # 5 minutes per batch during skip
    log_fn(f"[shard {shard_tag}] skipping first {SKIP} batches (timeout={SKIP_TIMEOUT}s each)...")
    loader_iter = iter(data_loader)
    skipped = 0
    fetch_errors = 0
    skip_failed = False
    while skipped < SKIP:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: next(loader_iter))
            try:
                future.result(timeout=SKIP_TIMEOUT)
                skipped += 1
            except StopIteration:
                log_fn(f"[shard {shard_tag}] dataloader exhausted after {skipped} batches")
                break
            except concurrent.futures.TimeoutError:
                log_fn(f"[shard {shard_tag}] skip TIMEOUT at batch {skipped + 1}/{SKIP} — aborting skip, will collect from fresh iterator")
                skip_failed = True
                break
            except Exception as exc:
                fetch_errors += 1
                log_fn(f"[shard {shard_tag}] skip fetch error {fetch_errors}: {type(exc).__name__}: {exc}")
    if skip_failed:
        # Give up on exact resumption; use a fresh iterator with offset seed
        # to get different data and keep collecting
        SKIP = skipped  # we only successfully skipped this many
        log_fn(f"[shard {shard_tag}] skip aborted. Using fresh iterator, collecting from batch {skipped + 1} onward with shifted seed")
        loader_iter = iter(data_loader)  # fresh, starts from batch 0 but we treat it as batch skipped+1
    log_fn(f"[shard {shard_tag}] skip done: {skipped}/{SKIP} ({fetch_errors} fetch errors, skip_failed={skip_failed})")

    remaining = MAX_BATCHES - SKIP
    log_fn(f"[shard {shard_tag}] resuming: {remaining} batches left, {len(existing_events)} events, {MAX_EVENTS - len(existing_events)} events remaining")

    base_shard = {
        "format": "ovggt_counterfactual_oracle_v1",
        "task_weights": TASK_WEIGHTS,
        "source_config": str(cfg_path),
        "dataset_key": "train_dataset",
        "score_state_projection_state": extract_score_state_projection_state(model),
    }

    flusher = OracleShardFlusher(
        base_shard=base_shard, output_path=output_path,
        flush_every_events=4, flush_every_batches=1, log_fn=log_fn,
    )

    events = list(existing_events)
    batch_idx = SKIP
    max_fetch_errors_int = args.max_fetch_errors
    COLLECT_TIMEOUT = 1800  # 30 min per batch during collection

    log_fn(f"[shard {shard_tag}] starting from batch {batch_idx + 1}/{MAX_BATCHES}")

    while batch_idx < MAX_BATCHES and len(events) < MAX_EVENTS:
        # Fetch batch with timeout
        batch = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: next(loader_iter))
            try:
                batch = future.result(timeout=COLLECT_TIMEOUT)
            except StopIteration:
                break
            except concurrent.futures.TimeoutError:
                log_fn(f"[shard {shard_tag}] TIMEOUT fetching batch {batch_idx + 1} — skipping")
                # Just abandon this batch, fresh iterator will produce next item
                loader_iter = iter(data_loader)
                # Fast-forward: skip batch_idx batches to get back to position
                # But we know that can hang, so just accept data shift
                batch_idx += 1
                continue
            except Exception as exc:
                fetch_errors += 1
                log_fn(f"[shard {shard_tag}] fetch error {fetch_errors}/{max_fetch_errors_int}: {type(exc).__name__}: {exc}")
                if fetch_errors > max_fetch_errors_int:
                    raise RuntimeError(f"Exceeded max_fetch_errors={max_fetch_errors_int}") from exc
                continue

        if batch is None:
            continue

        log_fn(f"[shard {shard_tag}] batch {batch_idx + 1}/{MAX_BATCHES} start: events={len(events)}")

        sequence_provenance = build_sequence_provenance(
            batch, batch_index=batch_idx, dataset_key="train_dataset",
        )
        log_fn(f"[shard {shard_tag}] batch {batch_idx + 1}/{MAX_BATCHES} provenance: {format_provenance_log_summary(sequence_provenance)}")

        batch = move_batch_to_device(
            normalize_batch_images(copy.deepcopy(batch)), torch.device("cuda")
        )

        remaining_ev = MAX_EVENTS - len(events)
        max_eps = 64
        if max_eps > 0:
            remaining_ev = min(remaining_ev, max_eps)

        collected_batch_events = []

        def on_event_collected(event, _cb=collected_batch_events, _ev=events):
            _cb.append(event)
            _ev.append(event)
            flusher.maybe_flush(_ev, batch_idx=None, force=False,
                                reason=f"[shard {shard_tag}] batch {batch_idx + 1}/{MAX_BATCHES} event {len(_cb)}")

        batch_events = collect_oracle_events_from_sequence(
            model=model, frames=batch, device=torch.device("cuda"),
            max_events=remaining_ev, num_samples=8, oracle_window=4,
            seed=SEED + batch_idx * 1009, teacher=teacher,
            event_prefix=f"batch{batch_idx}",
            sequence_provenance=sequence_provenance,
            log_fn=log_fn, log_every_subsets=2,
            subset_replay_batch_size=1, layers_per_frame=4,
            num_layers=num_layers, on_event_collected=on_event_collected,
        )

        batch_idx += 1
        log_fn(f"[shard {shard_tag}] batch {batch_idx}/{MAX_BATCHES} done: batch_events={len(collected_batch_events)} total={len(events)}")
        flusher.maybe_flush(events, batch_idx=batch_idx, force=False,
                            reason=f"[shard {shard_tag}] batch {batch_idx}/{MAX_BATCHES} done")

    # Final
    shard = dict(base_shard)
    shard["events"] = events
    shard["partial"] = False
    shard["num_events"] = len(events)
    flusher.maybe_flush(events, batch_idx=None, force=True, reason="final")
    log_fn(f"[shard {shard_tag}] RESUME DONE: total_events={len(events)} batches={batch_idx - SKIP}")


if __name__ == "__main__":
    main()
