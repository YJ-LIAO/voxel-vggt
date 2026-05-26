import torch
import numpy as np
from dust3r.datasets.collate import frontend_collate_fn

def test_collate_mixed_types():
    """B=2 sequences, each with 2 frames, mixed tensor/str/numpy/int fields."""
    sample0 = [
        {"img": torch.randn(3, 518, 392), "dataset": "blendedmvs", "label": "scene_a",
         "valid_mask": np.ones((518, 392), dtype=bool), "is_metric": True},
        {"img": torch.randn(3, 518, 392), "dataset": "blendedmvs", "label": "scene_a",
         "valid_mask": np.ones((518, 392), dtype=bool), "is_metric": True},
    ]
    sample1 = [
        {"img": torch.randn(3, 518, 392), "dataset": "co3d", "label": "scene_b",
         "valid_mask": np.zeros((518, 392), dtype=bool), "is_metric": False},
        {"img": torch.randn(3, 518, 392), "dataset": "co3d", "label": "scene_b",
         "valid_mask": np.zeros((518, 392), dtype=bool), "is_metric": False},
    ]
    batch = [sample0, sample1]
    result = frontend_collate_fn(batch)
    assert len(result) == 2
    assert result[0]["img"].shape == (2, 3, 518, 392)
    assert result[0]["dataset"] == ["blendedmvs", "co3d"]
    # numpy → stacked torch.Tensor
    assert isinstance(result[0]["valid_mask"], torch.Tensor)
    assert result[0]["valid_mask"].shape == (2, 518, 392)
    # other (bool) → passthrough list
    assert result[0]["is_metric"] == [True, False]
