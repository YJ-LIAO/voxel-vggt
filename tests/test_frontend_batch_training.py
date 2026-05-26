import torch
import numpy as np
from dust3r.datasets.collate import frontend_collate_fn

def test_collate_mixed_types():
    """B=2 sequences, each with 2 frames, mixed tensor/str fields."""
    sample0 = [
        {"img": torch.randn(3, 518, 392), "dataset": "blendedmvs", "label": "scene_a"},
        {"img": torch.randn(3, 518, 392), "dataset": "blendedmvs", "label": "scene_a"},
    ]
    sample1 = [
        {"img": torch.randn(3, 518, 392), "dataset": "co3d", "label": "scene_b"},
        {"img": torch.randn(3, 518, 392), "dataset": "co3d", "label": "scene_b"},
    ]
    batch = [sample0, sample1]
    result = frontend_collate_fn(batch)
    assert len(result) == 2
    assert result[0]["img"].shape == (2, 3, 518, 392)
    assert result[0]["dataset"] == ["blendedmvs", "co3d"]
