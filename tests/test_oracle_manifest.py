import json
import pytest
from pathlib import Path
from ovggt.training.oracle_manifest import build_manifest_from_dataset, partition_manifest, load_manifest


class TestSequenceManifest:
    def test_build_manifest_creates_jsonl(self, tmp_path):
        manifest_path = tmp_path / "manifest.jsonl"
        build_manifest_from_dataset(config_path="config/train_frontend_finetune.yaml", output_path=str(manifest_path), max_sequences=10)
        assert manifest_path.exists()
        lines = manifest_path.read_text().strip().splitlines()
        assert len(lines) == 10
        for line in lines:
            entry = json.loads(line)
            assert "sequence_id" in entry
            assert "dataset" in entry
            assert "frame_count" in entry
            assert entry["source_config"] == "config/train_frontend_finetune.yaml"

    def test_partition_manifest_hash_mod(self, tmp_path):
        entries = [{"sequence_id": f"seq_{i:04d}", "dataset": "test"} for i in range(100)]
        manifest_path = tmp_path / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        manifest = load_manifest(str(manifest_path))
        parts = [partition_manifest(manifest, num_shards=4, shard_id=i, policy="hash_mod") for i in range(4)]
        ids = [{e["sequence_id"] for e in p} for p in parts]
        assert sum(len(s) for s in ids) == len(set().union(*ids)) == 100

    def test_partition_manifest_contiguous(self, tmp_path):
        entries = [{"sequence_id": f"seq_{i:04d}", "dataset": "test"} for i in range(100)]
        manifest_path = tmp_path / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        manifest = load_manifest(str(manifest_path))
        part0 = partition_manifest(manifest, num_shards=4, shard_id=0, policy="contiguous")
        assert len(part0) == 25
        assert part0[0]["sequence_id"] == "seq_0000"
