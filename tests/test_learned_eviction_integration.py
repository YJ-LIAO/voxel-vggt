import os
import sys

import torch
from omegaconf import OmegaConf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.frontend_keyframe import KeyframeSwitchConfig
from train_frontend import build_frontend_cache_config
from train_token_scorer_oracle import build_ovggt_token_scorer_state_dict


@torch.no_grad()
def test_tiny_ovggt_learned_eviction_runs_at_commit_time():
    model = OVGGT(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        per_layer_budget=2,
        camera_budget=32,
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(
            enabled=True,
            dedup_enabled=False,
            learned_eviction_enabled=True,
            score_state_dim=8,
            budget_allocation="uniform",
        ),
        keyframe_switch_config=KeyframeSwitchConfig(
            strategy="fixed_interval",
            interval=2,
            max_history_anchors=2,
        ),
        aggregator_kwargs={
            "patch_embed": "conv",
            "depth": 4,
            "num_heads": 4,
            "num_register_tokens": 2,
        },
        camera_head_kwargs={"trunk_depth": 1, "num_heads": 4},
        depth_head_kwargs={
            "intermediate_layer_idx": [0, 1, 2, 3],
            "features": 16,
            "out_channels": [16, 16, 16, 16],
        },
        point_head_kwargs={
            "intermediate_layer_idx": [0, 1, 2, 3],
            "features": 16,
            "out_channels": [16, 16, 16, 16],
        },
        enable_track_head=False,
        use_token_scorer=True,
        scorer_bottleneck_dim=16,
    ).eval()
    frames = [{"img": torch.rand(1, 3, 28, 28)} for _ in range(3)]

    output = model.inference(frames)

    assert len(output.ress) == 3
    assert model.aggregator.token_scorers is not None
    assert model.aggregator.score_state_projs is not None
    assert output.ress[-1]["depth"].shape == (1, 28, 28, 1)


def test_strict_load_with_scorer_still_reports_missing_backbone_keys():
    model = OVGGT(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        per_layer_budget=2,
        camera_budget=32,
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(
            enabled=True,
            learned_eviction_enabled=True,
            score_state_dim=8,
        ),
        aggregator_kwargs={
            "patch_embed": "conv",
            "depth": 4,
            "num_heads": 4,
            "num_register_tokens": 2,
        },
        camera_head_kwargs={"trunk_depth": 1, "num_heads": 4},
        depth_head_kwargs={
            "intermediate_layer_idx": [0, 1, 2, 3],
            "features": 16,
            "out_channels": [16, 16, 16, 16],
        },
        point_head_kwargs={
            "intermediate_layer_idx": [0, 1, 2, 3],
            "features": 16,
            "out_channels": [16, 16, 16, 16],
        },
        enable_track_head=False,
        use_token_scorer=True,
        scorer_bottleneck_dim=16,
    )
    state_dict = model.state_dict()
    missing_key = next(key for key in state_dict if not key.startswith("aggregator.token_scorers."))
    incomplete = dict(state_dict)
    del incomplete[missing_key]

    try:
        model.load_state_dict(incomplete, strict=True)
    except RuntimeError as exc:
        assert missing_key in str(exc)
    else:
        raise AssertionError("strict=True should report missing non-scorer keys")


def test_oracle_deploy_checkpoint_loads_only_scorer_and_projection(tmp_path):
    model = OVGGT(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        per_layer_budget=2,
        camera_budget=32,
        mode="frontend_eval",
        frontend_cache_config=FrontendCacheConfig(
            enabled=True,
            learned_eviction_enabled=True,
            score_state_dim=8,
        ),
        aggregator_kwargs={
            "patch_embed": "conv",
            "depth": 4,
            "num_heads": 4,
            "num_register_tokens": 2,
        },
        camera_head_kwargs={"trunk_depth": 1, "num_heads": 4},
        depth_head_kwargs={
            "intermediate_layer_idx": [0, 1, 2, 3],
            "features": 16,
            "out_channels": [16, 16, 16, 16],
        },
        point_head_kwargs={
            "intermediate_layer_idx": [0, 1, 2, 3],
            "features": 16,
            "out_channels": [16, 16, 16, 16],
        },
        enable_track_head=False,
        use_token_scorer=True,
        scorer_bottleneck_dim=16,
    )
    scorer_state = model.aggregator.token_scorers[0].state_dict()
    scorer_state = {key: torch.full_like(value, 0.25) for key, value in scorer_state.items()}
    projection_state = {
        "aggregator.score_state_projs.0.weight": torch.full_like(
            model.aggregator.score_state_projs[0].weight,
            0.5,
        ),
        "aggregator.score_state_projs.0.bias": torch.full_like(
            model.aggregator.score_state_projs[0].bias,
            0.5,
        ),
    }
    checkpoint = {
        "model": build_ovggt_token_scorer_state_dict(
            scorer_state=scorer_state,
            num_layers=4,
            score_state_projection_state=projection_state,
        ),
        "backbone.weight": torch.randn(1),
    }
    checkpoint_path = tmp_path / "oracle_scorer.pt"
    torch.save(checkpoint, checkpoint_path)

    load_result = model.load_token_scorer_checkpoint(checkpoint_path)

    assert "backbone.weight" not in load_result.unexpected_keys
    assert torch.allclose(model.aggregator.token_scorers[3].scorer[-1].bias, torch.full_like(
        model.aggregator.token_scorers[3].scorer[-1].bias,
        0.25,
    ))
    assert torch.allclose(model.aggregator.score_state_projs[0].weight, torch.full_like(
        model.aggregator.score_state_projs[0].weight,
        0.5,
    ))


def test_frontend_cache_yaml_config_is_applied_to_model_construction():
    args = OmegaConf.create(
        {
            "frontend_cache": {
                "enabled": True,
                "learned_eviction_enabled": True,
                "score_state_dim": 8,
                "budget_allocation": "uniform",
                "dedup_enabled": False,
            }
        }
    )

    config = build_frontend_cache_config(args)
    model = OVGGT(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        camera_budget=32,
        mode="frontend_eval",
        frontend_cache_config=config,
        aggregator_kwargs={
            "patch_embed": "conv",
            "depth": 4,
            "num_heads": 4,
            "num_register_tokens": 2,
        },
        camera_head_kwargs={"trunk_depth": 1, "num_heads": 4},
        depth_head_kwargs={
            "intermediate_layer_idx": [0, 1, 2, 3],
            "features": 16,
            "out_channels": [16, 16, 16, 16],
        },
        point_head_kwargs={
            "intermediate_layer_idx": [0, 1, 2, 3],
            "features": 16,
            "out_channels": [16, 16, 16, 16],
        },
        enable_track_head=False,
        use_token_scorer=True,
        scorer_bottleneck_dim=16,
    )

    assert model.frontend_cache_config.learned_eviction_enabled is True
    assert model.frontend_cache_config.budget_allocation == "uniform"
    assert model.frontend_cache_config.dedup_enabled is False
    assert model.aggregator.score_state_projs[0].out_features == 8
