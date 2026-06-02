"""Sequence manifest utilities for deterministic Phase 4 shard partitioning."""
import hashlib
import json
from pathlib import Path


def build_manifest_from_dataset(config_path, output_path, max_sequences=None):
    """Build a JSONL manifest from the training dataset config. Phase 1 stub."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = int(max_sequences or 0)
    with open(output_path, "w") as f:
        for idx in range(count):
            f.write(json.dumps({
                "sequence_id": f"stub_seq_{idx:06d}",
                "dataset": "stub",
                "frame_count": 0,
                "source_config": str(config_path),
            }) + "\n")


def load_manifest(manifest_path):
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def partition_manifest(manifest, num_shards, shard_id, policy="hash_mod"):
    if policy == "contiguous":
        n = len(manifest)
        chunk = (n + num_shards - 1) // num_shards
        start = shard_id * chunk
        end = min(start + chunk, n)
        return manifest[start:end]
    result = []
    for entry in manifest:
        seq_id = entry.get("sequence_id", "")
        h = int(hashlib.md5(seq_id.encode()).hexdigest(), 16)
        if h % num_shards == shard_id:
            result.append(entry)
    return result
