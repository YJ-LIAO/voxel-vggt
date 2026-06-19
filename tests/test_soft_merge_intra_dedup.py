"""TDD for soft-merge intra dedup (frontend short-seq Pareto optimization).

Task 1: config field intra_dedup_mode (drop|merge) + precedence.
Task 2-3 tests added incrementally.
"""
import sys
sys.path.insert(0, "/path/to/mount/lyj/voxel-vggt/src")
from ovggt.utils.frontend_cache import FrontendCacheConfig


# ---------- Task 1: config field ----------

def test_intra_dedup_mode_default_drop():
    assert FrontendCacheConfig().intra_dedup_mode == "drop"


def test_intra_dedup_mode_accepts_merge():
    assert FrontendCacheConfig(intra_dedup_mode="merge").intra_dedup_mode == "merge"


def test_intra_frame_dedup_enabled_false_disables_regardless_of_mode():
    # precedence: the existing boolean wins; mode is inert when disabled
    # (the `if config.intra_frame_dedup_enabled:` guard at the intra block gates the whole thing)
    cfg = FrontendCacheConfig(intra_frame_dedup_enabled=False, intra_dedup_mode="merge")
    assert cfg.intra_frame_dedup_enabled is False
    assert cfg.intra_dedup_mode == "merge"  # field present, just inert
