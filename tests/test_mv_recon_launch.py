import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.mv_recon.launch import (
    build_7scenes_kwargs,
    build_ovggt_kwargs_for_eval,
    resolve_7scenes_root,
    validate_model_mode,
)


def test_build_ovggt_kwargs_legacy_mode():
    args = SimpleNamespace(
        ovggt_mode="legacy",
        frontend_dedup_enabled=True,
        frontend_anchor_interval=8,
    )
    kwargs = build_ovggt_kwargs_for_eval(args)
    assert kwargs["mode"] == "legacy"
    assert "frontend_cache_config" not in kwargs
    assert "keyframe_switch_config" not in kwargs


def test_build_ovggt_kwargs_frontend_mode():
    args = SimpleNamespace(
        ovggt_mode="frontend_eval",
        frontend_dedup_enabled=False,
        frontend_anchor_interval=12,
    )
    kwargs = build_ovggt_kwargs_for_eval(args)
    assert kwargs["mode"] == "frontend_eval"
    assert kwargs["frontend_cache_config"].enabled is True
    assert kwargs["frontend_cache_config"].dedup_enabled is False
    assert kwargs["keyframe_switch_config"].strategy == "fixed_interval"
    assert kwargs["keyframe_switch_config"].interval == 12


def test_resolve_7scenes_root_prefers_explicit_path():
    assert resolve_7scenes_root("/tmp/seven") == "/tmp/seven"


def test_resolve_7scenes_root_falls_back_to_repo_relative_default():
    assert resolve_7scenes_root("") == "./data/7scenes"


def test_validate_model_mode_rejects_frontend_mode_for_vggt():
    with pytest.raises(ValueError, match="frontend mode"):
        validate_model_mode("VGGT", "frontend_eval")


def test_build_7scenes_kwargs_passes_max_frames():
    kwargs = build_7scenes_kwargs(
        data_root="/tmp/seven",
        resolution=(518, 392),
        max_frames=17,
    )
    assert kwargs["ROOT"] == "/tmp/seven"
    assert kwargs["resolution"] == (518, 392)
    assert kwargs["max_frames"] == 17
