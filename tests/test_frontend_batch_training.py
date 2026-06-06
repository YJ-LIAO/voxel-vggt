import torch
import numpy as np
import pytest
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


@torch.no_grad()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batch2_forward_no_error():
    """Smoke test: B=2 training forward pass completes without error."""
    from ovggt.models.ovggt import OVGGT
    from ovggt.utils.frontend_cache import FrontendCacheConfig

    B = 2
    num_frames = 4
    H, W = 518, 392
    model = OVGGT(
        mode='frontend_train',
        per_layer_budget=209,
        camera_budget=64,
        use_token_scorer=False,
        frontend_cache_config=FrontendCacheConfig(
            enabled=True,
            dedup_enabled=True,
            intra_frame_dedup_enabled=False,
        ),
    ).cuda().eval()

    frames = []
    for _ in range(num_frames):
        frames.append({"img": torch.randn(B, 3, H, W).cuda()})

    output = model.inference(
        frames,
        history_anchor_strategy='fixed_interval',
        anchor_interval=2,
        max_anchors=2,
        cache_results=True,
    )
    assert output.ress is not None
    assert len(output.ress) == num_frames
    assert output.ress[0]["depth"].shape[0] == B  # batch dim preserved


@torch.no_grad()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batch1_vs_batch2_equivalence():
    """Same 2 sequences: B=1x2steps vs B=2x1step produce similar depth predictions."""
    from ovggt.models.ovggt import OVGGT
    from ovggt.utils.frontend_cache import FrontendCacheConfig

    B = 2
    num_frames = 4
    H, W = 518, 392

    def make_model():
        return OVGGT(
            mode='frontend_train',
            per_layer_budget=209,
            camera_budget=64,
            use_token_scorer=False,
            frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
        ).cuda().eval()

    g0 = torch.Generator().manual_seed(42)
    g1 = torch.Generator().manual_seed(99)
    frames_a = [{"img": torch.randn(1, 3, H, W, generator=g0).cuda()} for _ in range(num_frames)]
    frames_b = [{"img": torch.randn(1, 3, H, W, generator=g1).cuda()} for _ in range(num_frames)]

    # B=1 and B=2 must use identical weights; otherwise this compares two
    # unrelated random initializations instead of batch semantics.
    model1 = make_model()
    model2 = make_model()
    model2.load_state_dict(model1.state_dict())

    # B=1: run both sequences separately
    out1a = model1.inference(frames_a, history_anchor_strategy='fixed_interval',
                              anchor_interval=2, max_anchors=2)
    out1b = model1.inference(frames_b, history_anchor_strategy='fixed_interval',
                              anchor_interval=2, max_anchors=2)

    # B=2: stack into batched frames
    frames_2 = []
    for fa, fb in zip(frames_a, frames_b):
        frames_2.append({"img": torch.cat([fa["img"], fb["img"]], dim=0)})
    out2 = model2.inference(frames_2, history_anchor_strategy='fixed_interval',
                             anchor_interval=2, max_anchors=2)

    # Compare per-frame depth predictions
    for t in range(num_frames):
        d1a = out1a.ress[t]["depth"][0]
        d2a = out2.ress[t]["depth"][0]
        assert torch.allclose(d1a, d2a, rtol=1e-3, atol=1e-5), f"Frame {t}: batch 0 mismatch"
        d1b = out1b.ress[t]["depth"][0]
        d2b = out2.ress[t]["depth"][1]
        assert torch.allclose(d1b, d2b, rtol=1e-3, atol=1e-5), f"Frame {t}: batch 1 mismatch"


@torch.no_grad()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cache_isolation():
    """B=2: batch 0 and batch 1 have independent cache states."""
    from ovggt.models.ovggt import OVGGT
    from ovggt.utils.frontend_cache import FrontendCacheConfig

    B = 2
    model = OVGGT(
        mode='frontend_train', per_layer_budget=209, camera_budget=64,
        use_token_scorer=False,
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    ).cuda().eval()

    g0 = torch.Generator().manual_seed(42)
    g1 = torch.Generator().manual_seed(99)
    frames = [{"img": torch.cat([
        torch.randn(1, 3, 518, 392, generator=g0),
        torch.randn(1, 3, 518, 392, generator=g1),
    ], dim=0).cuda()} for _ in range(3)]
    out = model.inference(frames, history_anchor_strategy='fixed_interval',
                           anchor_interval=2, max_anchors=2)
    d0 = out.ress[-1]["depth"][0]
    d1 = out.ress[-1]["depth"][1]
    assert not torch.allclose(d0, d1), "Cache states not isolated — outputs are identical"
