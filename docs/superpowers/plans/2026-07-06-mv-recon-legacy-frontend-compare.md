# MV Recon Legacy vs Frontend Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible `mv_recon` evaluation entrypoint that can compare OVGGT legacy inference against migrated frontend-cache inference on the same checkpoint and 7-Scenes dataset.

**Architecture:** Keep a single Python evaluator in `src/eval/mv_recon/launch.py` and parameterize only the OVGGT-specific mode and dataset root. Add two thin shell wrappers, `run_legacy.sh` and `run_frontend.sh`, that differ only in mode and output directory while sharing checkpoint and dataset defaults that work inside the linked worktree.

**Tech Stack:** Python 3.11, argparse, PyTorch, accelerate, bash, pytest.

---

## File Structure

- Modify: `src/eval/mv_recon/launch.py`
  Purpose: add explicit OVGGT evaluation mode selection, configurable 7-Scenes root, fast-fail validation, and honor the existing `--max_frames` argument.
- Create: `src/eval/mv_recon/run_legacy.sh`
  Purpose: launch legacy OVGGT `mv_recon` evaluation with worktree-safe absolute defaults.
- Create: `src/eval/mv_recon/run_frontend.sh`
  Purpose: launch frontend-cache OVGGT `mv_recon` evaluation with the same checkpoint and dataset defaults as legacy.
- Create: `tests/test_mv_recon_launch.py`
  Purpose: cover mode-specific OVGGT constructor kwargs, dataset-root resolution, non-OVGGT mode validation, and max-frame propagation.

## Task 1: Add Launch Helper Tests

**Files:**
- Create: `tests/test_mv_recon_launch.py`
- Test: `tests/test_mv_recon_launch.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mv_recon_launch.py` with:

```python
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.mv_recon.launch import (
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
```

- [ ] **Step 2: Run the test file and verify it fails**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-cache-migration
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest -q tests/test_mv_recon_launch.py
```

Expected: FAIL with `ImportError` because `build_ovggt_kwargs_for_eval`, `resolve_7scenes_root`, and `validate_model_mode` do not exist yet.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_mv_recon_launch.py
git -c user.name=Codex -c user.email=codex@localhost commit -m "test: add mv recon mode selection tests"
```

## Task 2: Parameterize `launch.py`

**Files:**
- Modify: `src/eval/mv_recon/launch.py`
- Test: `tests/test_mv_recon_launch.py`

- [ ] **Step 1: Implement minimal helpers and args**

In `src/eval/mv_recon/launch.py`, add:

```python
def resolve_7scenes_root(data_root: str) -> str:
    return data_root or "./data/7scenes"


def validate_model_mode(model_name: str, ovggt_mode: str) -> None:
    if model_name != "OVGGT" and ovggt_mode != "legacy":
        raise ValueError(
            f"frontend mode is only supported for OVGGT, got model_name={model_name!r}, "
            f"ovggt_mode={ovggt_mode!r}"
        )


def build_ovggt_kwargs_for_eval(args):
    if args.ovggt_mode == "legacy":
        return {"mode": "legacy"}

    from ovggt.utils.frontend_cache import FrontendCacheConfig
    from ovggt.utils.frontend_keyframe import KeyframeSwitchConfig

    return {
        "mode": "frontend_eval",
        "frontend_cache_config": FrontendCacheConfig(
            enabled=True,
            dedup_enabled=args.frontend_dedup_enabled,
        ),
        "keyframe_switch_config": KeyframeSwitchConfig(
            strategy="fixed_interval",
            interval=args.frontend_anchor_interval,
        ),
    }
```

Extend the parser with:

```python
parser.add_argument("--data_root", type=str, default="")
parser.add_argument(
    "--ovggt_mode",
    type=str,
    default="legacy",
    choices=("legacy", "frontend_eval"),
)
parser.add_argument(
    "--frontend_dedup_enabled",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--frontend_anchor_interval", type=int, default=8)
```

- [ ] **Step 2: Wire helpers into dataset construction and OVGGT init**

Change dataset construction to:

```python
seven_scenes_root = resolve_7scenes_root(args.data_root)
datasets_all = {
    "7scenes": SevenScenes(
        split="test",
        ROOT=seven_scenes_root,
        resolution=resolution,
        num_seq=1,
        full_video=True,
        kf_every=2,
        max_frames=args.max_frames,
    ),
}
```

And OVGGT construction to:

```python
validate_model_mode(model_name, args.ovggt_mode)
if model_name == "OVGGT":
    model = OVGGT(**build_ovggt_kwargs_for_eval(args))
```

- [ ] **Step 3: Add fast-fail path checks**

Before dataset/model construction, add:

```python
if args.weights and not os.path.exists(args.weights):
    raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
if seven_scenes_root and not os.path.exists(seven_scenes_root):
    raise FileNotFoundError(f"7-Scenes root not found: {seven_scenes_root}")
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-cache-migration
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest -q tests/test_mv_recon_launch.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/mv_recon/launch.py tests/test_mv_recon_launch.py
git -c user.name=Codex -c user.email=codex@localhost commit -m "feat: parameterize mv recon ovggt evaluation mode"
```

## Task 3: Add Worktree-Safe Compare Wrappers

**Files:**
- Create: `src/eval/mv_recon/run_legacy.sh`
- Create: `src/eval/mv_recon/run_frontend.sh`

- [ ] **Step 1: Add legacy wrapper**

Create `src/eval/mv_recon/run_legacy.sh`:

```bash
#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/Train/LYJ/miniconda3/envs/OVGGT/bin/python}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-/Train/LYJ/workspace/OVGGT/ckpt/checkpoints.pth}"
DATA_ROOT="${DATA_ROOT:-/path/to/mount/lyj/OpenDataLab___7-Scenes/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/mv_recon/OVGGT_checkpoints_legacy}"
MAX_FRAMES="${MAX_FRAMES:-300}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29602}"

echo "${OUTPUT_DIR}"
"${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    "${SCRIPT_DIR}/launch.py" \
    --weights "${MODEL_WEIGHTS}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "OVGGT" \
    --ovggt_mode "legacy" \
    --data_root "${DATA_ROOT}" \
    --max_frames "${MAX_FRAMES}"
```

- [ ] **Step 2: Add frontend wrapper**

Create `src/eval/mv_recon/run_frontend.sh`:

```bash
#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/Train/LYJ/miniconda3/envs/OVGGT/bin/python}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-/Train/LYJ/workspace/OVGGT/ckpt/checkpoints.pth}"
DATA_ROOT="${DATA_ROOT:-/path/to/mount/lyj/OpenDataLab___7-Scenes/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/mv_recon/OVGGT_checkpoints_frontend}"
MAX_FRAMES="${MAX_FRAMES:-300}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29603}"
FRONTEND_ANCHOR_INTERVAL="${FRONTEND_ANCHOR_INTERVAL:-8}"

echo "${OUTPUT_DIR}"
"${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    "${SCRIPT_DIR}/launch.py" \
    --weights "${MODEL_WEIGHTS}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "OVGGT" \
    --ovggt_mode "frontend_eval" \
    --data_root "${DATA_ROOT}" \
    --frontend_anchor_interval "${FRONTEND_ANCHOR_INTERVAL}" \
    --max_frames "${MAX_FRAMES}"
```

- [ ] **Step 3: Make scripts executable and verify shell syntax**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-cache-migration
chmod +x src/eval/mv_recon/run_legacy.sh src/eval/mv_recon/run_frontend.sh
bash -n src/eval/mv_recon/run_legacy.sh
bash -n src/eval/mv_recon/run_frontend.sh
```

Expected: no output, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add src/eval/mv_recon/run_legacy.sh src/eval/mv_recon/run_frontend.sh
git -c user.name=Codex -c user.email=codex@localhost commit -m "feat: add mv recon legacy frontend compare scripts"
```

## Task 4: Verify Real Compare Entry Points

**Files:**
- Verify: `src/eval/mv_recon/launch.py`
- Verify: `src/eval/mv_recon/run_legacy.sh`
- Verify: `src/eval/mv_recon/run_frontend.sh`

- [ ] **Step 1: Run the full Python test suite**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-cache-migration
PYTHONPATH=src /Train/LYJ/miniconda3/envs/OVGGT/bin/python -m pytest -q tests
```

Expected: PASS.

- [ ] **Step 2: Run a real frontend mini-inference**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-cache-migration/src
PYTHONPATH=. /Train/LYJ/miniconda3/envs/OVGGT/bin/python - <<'PY'
import torch
from eval.mv_recon.data import SevenScenes
from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.frontend_keyframe import KeyframeSwitchConfig

root = "/path/to/mount/lyj/OpenDataLab___7-Scenes/raw"
ckpt_path = "/Train/LYJ/workspace/OVGGT/ckpt/checkpoints.pth"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset = SevenScenes(
    split="test",
    ROOT=root,
    resolution=(518, 392),
    num_seq=1,
    full_video=True,
    kf_every=2,
    max_frames=4,
)
frames = dataset[0][:4]
for frame in frames:
    frame["img"] = ((frame["img"] + 1.0) / 2.0).to(device)

model = OVGGT(
    mode="frontend_eval",
    frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False),
    keyframe_switch_config=KeyframeSwitchConfig(strategy="fixed_interval", interval=2),
).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt, strict=True)
model.eval()
with torch.no_grad():
    out = model.inference(frames, cache_results=True, return_views=False)
assert len(out.ress) == 4
assert "camera_pose_rel" in out.ress[0]
print("frontend mini inference OK")
PY
```

Expected: prints `frontend mini inference OK`.

- [ ] **Step 3: Smoke-run legacy wrapper**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-cache-migration
MAX_FRAMES=20 NUM_PROCESSES=1 MAIN_PROCESS_PORT=29612 timeout 180 \
  src/eval/mv_recon/run_legacy.sh
```

Expected: command either completes or times out after starting evaluation, but it must pass checkpoint/data-root validation and begin processing without immediate import/path/model-construction failure.

- [ ] **Step 4: Smoke-run frontend wrapper**

Run:

```bash
cd /Train/LYJ/workspace/OVGGT/.worktrees/frontend-cache-migration
MAX_FRAMES=20 NUM_PROCESSES=1 MAIN_PROCESS_PORT=29613 timeout 180 \
  src/eval/mv_recon/run_frontend.sh
```

Expected: command either completes or times out after starting evaluation, but it must pass checkpoint/data-root validation and begin processing without immediate import/path/model-construction failure.

- [ ] **Step 5: Commit final verification-related adjustments if needed**

If verification requires small fixes:

```bash
git add src/eval/mv_recon/launch.py src/eval/mv_recon/run_legacy.sh src/eval/mv_recon/run_frontend.sh tests/test_mv_recon_launch.py
git -c user.name=Codex -c user.email=codex@localhost commit -m "fix: stabilize mv recon compare entrypoints"
```

If no files changed, skip this commit.

## Self-Review

- Spec coverage: the plan covers launch parameterization, explicit legacy/frontend mode selection, worktree-safe wrappers, configurable dataset root, and real smoke verification on both paths.
- Placeholder scan: no `TODO` or undefined references remain.
- Type consistency: the plan consistently uses `ovggt_mode`, `frontend_dedup_enabled`, `frontend_anchor_interval`, `data_root`, `run_legacy.sh`, and `run_frontend.sh`.
