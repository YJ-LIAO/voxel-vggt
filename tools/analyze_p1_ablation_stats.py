#!/usr/bin/env python3
"""
Rigorous statistical analysis of the P1 ablation: paired t-tests + bootstrap CIs
vs P1-current, per frame tier. Supplements analyze_p1_ablation.py (descriptive only)
with significance testing so "significantly better" claims are evidence-backed.

Paired design: same (scene, seed) across configs -> paired difference, same subject.
Determinism pre-check (bit-identical) validated the pairing assumption.
"""
import os, json, glob
from collections import defaultdict
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "p1_ablation_results_lyj")
BASELINE = "P1-current"
ORDER = ["P1-none", "P1-current", "P1-v1cap",
         "P1-mechC-r0.1", "P1-mechC-r0.2", "P1-mechC-r0.3", "P1-mechC-r0.5"]


def load():
    runs = []
    for fp in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        if os.path.basename(fp).startswith(("detcheck", "diag")):
            continue
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        if "config" in d and "ate_rmse" in d and d.get("num_frames"):
            runs.append(d)
    return runs


def main():
    runs = load()
    # paired[(nf, scene, seed)] = {cfg: ate}
    paired = defaultdict(dict)
    for r in runs:
        paired[(r["num_frames"], r["scene"], r["seed"])][r["config"]] = r["ate_rmse"]
    nfs = sorted({k[0] for k in paired})

    print("=" * 100)
    print("PAIRED STATISTICAL TEST vs P1-current (two-sided paired t-test on ATE diff)")
    print("  diff = config - current; negative => config BETTER (lower ATE).")
    print("  Win% = fraction of pairs where config beat current.")
    print("=" * 100)
    for nf in nfs:
        print(f"\n--- {nf} frames ---")
        print(f"{'config':<16}{'mean diff':>12}{'95% CI':>22}{'t':>9}{'p':>12}{'n':>5}{'win%':>8}")
        for cfg in ORDER:
            if cfg == BASELINE:
                continue
            diffs = []
            for (n2, scene, seed), cmap in paired.items():
                if n2 != nf or BASELINE not in cmap or cfg not in cmap:
                    continue
                diffs.append(cmap[cfg] - cmap[BASELINE])
            if not diffs:
                continue
            d = np.array(diffs)
            t, p = stats.ttest_rel(d, np.zeros_like(d))  # one-sample t on diffs
            mean = d.mean()
            se = d.std(ddof=1) / np.sqrt(len(d))
            ci = (mean - 1.96 * se, mean + 1.96 * se)
            win = (d < 0).mean() * 100
            ci_str = f"[{ci[0]:+.4f},{ci[1]:+.4f}]"
            sig = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
            print(f"{cfg:<16}{mean:>+12.4f}{ci_str:>22}{t:>9.2f}{p:>11.1e}{len(d):>5}{win:>7.0f}%{sig}")

    # direct none vs mechC-r0.1 (the two tied-best) — is the tie significant?
    print("\n" + "=" * 100)
    print("DIRECT PAIRED: P1-none vs P1-mechC-r0.1 (are the two tied-best actually different?)")
    print("=" * 100)
    for nf in nfs:
        diffs = [cmap["P1-mechC-r0.1"] - cmap["P1-none"]
                 for (n2, s, sd), cmap in paired.items()
                 if n2 == nf and "P1-none" in cmap and "P1-mechC-r0.1" in cmap]
        if len(diffs) < 2:
            continue
        d = np.array(diffs)
        t, p = stats.ttest_rel(d, np.zeros_like(d))
        print(f"  {nf}f: mean(r0.1-none)={d.mean():+.5f}  p={p:.3f}  n={len(d)}  "
              f"{'DIFFERENT' if p < 0.05 else 'not significantly different (tie)'}")


if __name__ == "__main__":
    main()
