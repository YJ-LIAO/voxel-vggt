#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Dict, List, Sequence, Tuple

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ovggt.models import ovggt as ovggt_model_mod
from ovggt.heads.camera_head import CameraHead
from ovggt.models.aggregator import Aggregator
from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig, LayerCacheState
from ovggt.utils.frontend_keyframe import FrontendKeyframeManager

from eval_blendmvs_frontend import build_frames, load_model, parse_sequence_args, prepare_sequence_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile frontend inference hotspots on BlendMVS1.")
    parser.add_argument(
        "--dataset-root",
        default="/mnt/lyj/workspace/StreamVGGT/data/train/processed_blendedmvs",
        help="BlendMVS root. Supports both raw BlendMVS1 layout and processed_blendedmvs layout.",
    )
    parser.add_argument(
        "--weights",
        default="/mnt/lyj/workspace/StreamVGGT/ckpt/checkpoints.pth",
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--scene",
        default="588c989a90414422fbe86d96",
        help="Scene id under dataset root.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=["0,3,4"],
        help="Comma-separated frame indices, e.g. 0,3,4. Can be repeated.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


class TimerStore:
    def __init__(self, device: torch.device):
        self.device = device
        self.samples: Dict[str, List[float]] = defaultdict(list)

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name: str):
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.samples[name].append(time.perf_counter() - start)

    def summary(self) -> Dict[str, Dict[str, float]]:
        total = sum(sum(values) for values in self.samples.values())
        out = {}
        for name, values in sorted(self.samples.items(), key=lambda item: sum(item[1]), reverse=True):
            time_sum = sum(values)
            out[name] = {
                "calls": float(len(values)),
                "total_sec": time_sum,
                "avg_ms": (time_sum / max(len(values), 1)) * 1000.0,
                "share_pct": (time_sum / total * 100.0) if total > 0 else 0.0,
            }
        return out


@contextmanager
def patch_method(obj, attr: str, wrapper_factory: Callable[[Callable], Callable]):
    original = getattr(obj, attr)
    patched = wrapper_factory(original)
    setattr(obj, attr, patched)
    try:
        yield
    finally:
        setattr(obj, attr, original)


def make_wrapper(store: TimerStore, name: str):
    def factory(fn: Callable):
        def wrapped(*args, **kwargs):
            with store.measure(name):
                return fn(*args, **kwargs)

        return wrapped

    return factory


def profile_sequence(
    model: OVGGT,
    frames: List[dict],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    store = TimerStore(device)
    original_empty_cache = torch.cuda.empty_cache

    def timed_empty_cache():
        with store.measure("torch.cuda.empty_cache"):
            return original_empty_cache()

    patchers = [
        patch_method(Aggregator, "forward", make_wrapper(store, "Aggregator.forward")),
        patch_method(CameraHead, "forward", make_wrapper(store, "CameraHead.forward")),
        patch_method(FrontendKeyframeManager, "update", make_wrapper(store, "FrontendKeyframeManager.update")),
        patch_method(
            ovggt_model_mod,
            "build_frame_token_metadata_base",
            make_wrapper(store, "build_frame_token_metadata_base"),
        ),
        patch_method(LayerCacheState, "apply_keyframe_event_", make_wrapper(store, "LayerCacheState.apply_keyframe_event_")),
        patch_method(LayerCacheState, "reorder_by_anchor_slots_", make_wrapper(store, "LayerCacheState.reorder_by_anchor_slots_")),
        patch_method(LayerCacheState, "apply_voxel_dedup_", make_wrapper(store, "LayerCacheState.apply_voxel_dedup_")),
        patch_method(LayerCacheState, "commit_pending_update_", make_wrapper(store, "LayerCacheState.commit_pending_update_")),
        patch_method(CameraHead, "apply_keyframe_event", make_wrapper(store, "CameraHead.apply_keyframe_event")),
    ]

    if device.type == "cuda":
        torch.cuda.empty_cache = timed_empty_cache
        torch.cuda.reset_peak_memory_stats(device)

    try:
        with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5], patchers[6], patchers[7], patchers[8]:
            with store.measure("OVGGT.inference_frontend_total"):
                with torch.no_grad():
                    if device.type == "cuda":
                        major = torch.cuda.get_device_capability(device)[0]
                        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            output = model.inference(
                                frames,
                                history_anchor_strategy="fixed_interval",
                                anchor_interval=48,
                            )
                    else:
                        output = model.inference(
                            frames,
                            history_anchor_strategy="fixed_interval",
                            anchor_interval=48,
                        )
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache = original_empty_cache

    summary = store.summary()
    if device.type == "cuda":
        summary["__peak_mem__"] = {
            "calls": 1.0,
            "total_sec": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            "avg_ms": 0.0,
            "share_pct": 0.0,
        }
    summary["__num_frames__"] = {
        "calls": float(len(output.ress) if output.ress is not None else 0),
        "total_sec": 0.0,
        "avg_ms": 0.0,
        "share_pct": 0.0,
    }
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    scene_dir = os.path.join(args.dataset_root, args.scene)
    sequences = parse_sequence_args(args.sequence)

    model = load_model(args.weights, device, frontend_enabled=True, voxel_size=args.voxel_size)
    if not isinstance(model.frontend_cache_config, FrontendCacheConfig):
        raise RuntimeError("Expected frontend-enabled model")

    results = []
    for frame_ids in sequences:
        seq_key = ",".join(str(idx) for idx in frame_ids)
        sequence_data = prepare_sequence_data(scene_dir, frame_ids)
        frames = build_frames(sequence_data["image_paths"], device)
        profile = profile_sequence(model, frames, device)
        item = {
            "sequence": seq_key,
            "profile": profile,
        }
        results.append(item)
        print(json.dumps(item, indent=2))

    output = {
        "dataset_root": args.dataset_root,
        "scene": args.scene,
        "device": args.device,
        "results": results,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
