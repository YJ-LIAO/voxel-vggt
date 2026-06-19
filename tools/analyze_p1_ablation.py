#!/usr/bin/env python3
"""
Analyze P1 ablation results (JSON per run in tools/p1_ablation_results_lyj/).

Produces the spec §3.2 comparison:
  - Mean +/- std ATE per (config, num_frames) across scenes x seeds
  - Paired diff vs P1-current (same scene+seed) per num_frames, with sign
  - Which config wins at 1000f-equivalent (here 500f) per scene

Paired design is valid because determinism pre-check showed bit-identical ATE across runs.
"""
import os
import sys
import json
import glob
from collections import defaultdict

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1_ablation_results_lyj")
BASELINE = "P1-current"
# Config display order (logical progression).
ORDER = ["P1-none", "P1-current", "P1-v1cap",
         "P1-mechC-r0.1", "P1-mechC-r0.2", "P1-mechC-r0.3", "P1-mechC-r0.5"]


def load_runs():
    """Return list of run dicts (skip detcheck/pose-heavy files, keep single-run results)."""
    runs = []
    for fp in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        base = os.path.basename(fp)
        if base.startswith("detcheck"):
            continue
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            continue
        if "config" in d and "ate_rmse" in d and "scene" in d:
            runs.append(d)
    return runs


def main():
    runs = load_runs()
    if not runs:
        print(f"No results found in {RESULTS_DIR}")
        return

    # group[(config, nf)] = list of ATE across (scene, seed)
    grouped = defaultdict(list)
    # per_scene[(config, nf, scene)] = list of ATE across seeds
    per_scene = defaultdict(list)
    # paired[(nf, scene, seed)] = {config: ATE}
    paired = defaultdict(dict)

    for r in runs:
        cfg = r["config"]; nf = r["num_frames"]; scene = r["scene"]
        ate = r["ate_rmse"]; seed = r["seed"]
        grouped[(cfg, nf)].append(ate)
        per_scene[(cfg, nf, scene)].append(ate)
        paired[(nf, scene, seed)][cfg] = ate

    nfs = sorted({nf for (_, nf) in grouped})
    configs = [c for c in ORDER if any((c, nf) in grouped for nf in nfs)]
    if BASELINE not in configs:
        configs = [BASELINE] + configs

    print("=" * 92)
    print("P1 ABLATION SUMMARY — ATE RMSE (m), mean +/- std over (scene x seed)")
    print("=" * 92)
    header = f"{'config':<16}" + "".join(f"{f'@{nf}f':>16}" for nf in nfs)
    print(header)
    print("-" * 92)
    for cfg in configs:
        cells = []
        for nf in nfs:
            vals = grouped.get((cfg, nf))
            if vals:
                cells.append(f"{np.mean(vals):.4f}+/-{np.std(vals):.4f}")
            else:
                cells.append("—")
        print(f"{cfg:<16}" + "".join(f"{c:>16}" for c in cells))
    n_scenes = len({r["scene"] for r in runs})
    n_seeds = len({r["seed"] for r in runs})
    print(f"\n(scenes={n_scenes}, seeds={n_seeds}, total runs={len(runs)})")

    # Paired diff vs baseline, per nf.
    print("\n" + "=" * 92)
    print(f"PAIRED DIFF vs {BASELINE} (negative = config better than baseline)")
    print("=" * 92)
    for nf in nfs:
        diffs_by_cfg = defaultdict(list)
        for (n2, scene, seed), cmap in paired.items():
            if n2 != nf or BASELINE not in cmap:
                continue
            base = cmap[BASELINE]
            for cfg, ate in cmap.items():
                if cfg == BASELINE:
                    continue
                diffs_by_cfg[cfg].append(ate - base)
        if not diffs_by_cfg:
            continue
        print(f"\n--- {nf} frames ---")
        print(f"{'config':<16}{'mean diff':>14}{'std':>12}{'n_pairs':>10}{'better?':>10}")
        for cfg in [c for c in ORDER if c in diffs_by_cfg]:
            ds = np.array(diffs_by_cfg[cfg])
            better = "<" if ds.mean() < -1e-6 else (">" if ds.mean() > 1e-6 else "=")
            print(f"{cfg:<16}{ds.mean():>14.4f}{ds.std():>12.4f}{len(ds):>10}{better:>10}")

    print("\n" + "=" * 92)
    print("WINNER per scene @ each frame count (lowest mean ATE)")
    print("=" * 92)
    for nf in nfs:
        print(f"\n--- {nf} frames ---")
        for scene in sorted({r["scene"] for r in runs if r["num_frames"] == nf}):
            row = {cfg: (np.mean(per_scene[(cfg, nf, scene)]), len(per_scene[(cfg, nf, scene)]))
                   for cfg in configs if (cfg, nf, scene) in per_scene}
            if not row:
                continue
            winner = min(row, key=lambda c: row[c][0])
            wmean, wn = row[winner]
            second = sorted(row.items(), key=lambda kv: kv[1][0])
            runner = second[1][0] if len(second) > 1 else "—"
            rmean = second[1][1][0] if len(second) > 1 else float("nan")
            print(f"  {scene:<20} winner={winner:<16} ({wmean:.4f}, n={wn})   "
                  f"runner-up={runner:<16} ({rmean:.4f})")

    print("\n" + "=" * 92)


if __name__ == "__main__":
    main()
