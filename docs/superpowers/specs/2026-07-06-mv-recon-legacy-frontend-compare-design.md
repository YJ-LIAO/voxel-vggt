# MV Recon Legacy vs Frontend Compare Design

## Goal

Add a reproducible evaluation entrypoint that can compare OVGGT legacy inference against the migrated frontend-cache inference on the same checkpoint and dataset, without duplicating evaluation logic.

## Scope

The change covers:

- Parameterizing `src/eval/mv_recon/launch.py` so OVGGT can be instantiated in either `legacy` or `frontend_eval` mode.
- Making the 7-Scenes dataset root configurable instead of hard-coded to `./data/7scenes`.
- Making the checkpoint path and output directory robust when invoked from a linked worktree.
- Adding two thin shell wrappers:
  - `src/eval/mv_recon/run_legacy.sh`
  - `src/eval/mv_recon/run_frontend.sh`

The change does not cover:

- Changing metric computation.
- Changing the dataset loader format.
- Modifying VGGT evaluation behavior.
- Adding new model features beyond selecting an existing OVGGT inference path.

## Current State

`src/eval/mv_recon/launch.py` currently constructs OVGGT with `OVGGT()` and therefore evaluates the default `legacy` path only. The script also hard-codes 7-Scenes to `ROOT="./data/7scenes"`, and `src/eval/mv_recon/run.sh` assumes `../ckpt/checkpoints.pth`, which is incorrect when run from `.worktrees/frontend-cache-migration/src`.

As a result, current `mv_recon` evaluation cannot directly measure the migrated frontend-cache path and is brittle in a worktree environment.

## Design

### Unified Python Entry

Keep a single Python evaluator, `src/eval/mv_recon/launch.py`, as the source of truth.

Add OVGGT-specific CLI parameters:

- `--ovggt_mode`, allowed values: `legacy`, `frontend_eval`
- `--data_root`, default empty; if omitted, `launch.py` may keep its relative fallback, while the wrapper scripts will always pass an explicit absolute dataset root
- `--frontend_dedup_enabled`, boolean flag, default enabled
- `--frontend_anchor_interval`, integer, default `8` for frontend mode

For `model_name == "OVGGT"`:

- `legacy` mode constructs `OVGGT(mode="legacy")`
- `frontend_eval` constructs `OVGGT(mode="frontend_eval", frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=args.frontend_dedup_enabled), keyframe_switch_config=KeyframeSwitchConfig(strategy="fixed_interval", interval=args.frontend_anchor_interval))`

VGGT behavior remains unchanged.

### Thin Shell Wrappers

Add two wrapper scripts that differ only in OVGGT mode and output directory naming.

`run_legacy.sh`:

- Uses the same checkpoint and dataset as frontend.
- Passes `--ovggt_mode legacy`.
- Writes to `eval_results/mv_recon/OVGGT_checkpoints_legacy`.

`run_frontend.sh`:

- Uses the same checkpoint and dataset as legacy.
- Passes `--ovggt_mode frontend_eval`.
- Writes to `eval_results/mv_recon/OVGGT_checkpoints_frontend`.

Both scripts should:

- Use absolute paths rooted at the repository top level so they work from a linked worktree.
- Accept the local 7-Scenes root at `/path/to/mount/lyj/OpenDataLab___7-Scenes/raw` by default for this workspace.
- Prefer `"$PYTHON" -m accelerate.commands.launch` or the environment `accelerate` binary in a way that works inside the `OVGGT` conda env.

### Path Handling

The wrappers should compute the repo root from the script location rather than relying on `workdir='..'`.

Expected pattern:

- script dir: `src/eval/mv_recon`
- src root: `src`
- repo root: parent of `src`

Checkpoint default:

- `${REPO_ROOT}/ckpt/checkpoints.pth` (override with `MODEL_WEIGHTS=/abs/path` for local checkpoints)

Dataset default:

- `/path/to/mount/lyj/OpenDataLab___7-Scenes/raw`

This avoids creating symlink requirements for routine evaluation.

## Output Layout

Outputs should remain under `eval_results/mv_recon/`, with mode-specific folders:

- `eval_results/mv_recon/OVGGT_checkpoints_legacy`
- `eval_results/mv_recon/OVGGT_checkpoints_frontend`

That keeps downstream comparison simple and avoids mixing artifacts from the two paths.

## Error Handling

Fail fast when:

- checkpoint path does not exist
- dataset root does not exist
- `--ovggt_mode frontend_eval` is requested for a non-OVGGT model
- unsupported mode value is passed

Errors should be raised before long evaluation starts.

## Testing

Add focused tests before implementation:

- a CLI-level or helper-level test that OVGGT construction uses `mode="legacy"` vs `mode="frontend_eval"` correctly
- a test that dataset root is taken from the new argument instead of the hard-coded default
- a test or smoke verification that the wrapper-resolved checkpoint path is absolute and valid in the worktree

After implementation, verify:

- `pytest -q tests`
- a real small `frontend_eval` inference on 7-Scenes frames with the provided checkpoint
- a smoke launch of `mv_recon` legacy mode
- a smoke launch of `mv_recon` frontend mode

## Risks

The main risk is introducing too many frontend-specific knobs into `launch.py`. Keep the new interface minimal and OVGGT-specific.

The second risk is diverging wrapper scripts. They should stay thin and differ only in mode, output path, and optional frontend-specific defaults.

The third risk is assuming relative paths from the current working directory. The wrappers should resolve paths from the script location to avoid worktree-specific breakage.
