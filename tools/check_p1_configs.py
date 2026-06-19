#!/usr/bin/env python3
"""Quick CPU-only check: all 7 P1 configs construct without error (v1/v2 mutex, ring ratios).
Does NOT run GPU inference. Use this to verify the config layer when the GPU smoke is blocked."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from ovggt.utils.frontend_cache import FrontendCacheConfig

SPEC = {
    "P1-none":       dict(fifo_keep_topk=0,  fifo_protected_ring_ratio=0.0, max_protected_ratio=1.0),
    "P1-current":    dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.0, max_protected_ratio=1.0),
    "P1-v1cap":      dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.0, max_protected_ratio=0.5),
    "P1-mechC-r0.1": dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.1, max_protected_ratio=1.0),
    "P1-mechC-r0.2": dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.2, max_protected_ratio=1.0),
    "P1-mechC-r0.3": dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.3, max_protected_ratio=1.0),
    "P1-mechC-r0.5": dict(fifo_keep_topk=80, fifo_protected_ring_ratio=0.5, max_protected_ratio=1.0),
}
PER_LAYER_BUDGET = 8000
ok = True
print(f"{'config':<16}{'fifo_topk':>10}{'ring_cap':>10}{'maxprot':>9}{'status':>9}")
for name, p in SPEC.items():
    try:
        cfg = FrontendCacheConfig(enabled=True, dedup_enabled=True,
                                  intra_frame_dedup_enabled=False,
                                  budget_allocation="dynamic",
                                  learned_fifo_keep_count=False, **p)
        ring_cap = int(cfg.fifo_protected_ring_ratio * PER_LAYER_BUDGET) if cfg.fifo_protected_ring_ratio > 0 else 0
        print(f"{name:<16}{cfg.fifo_keep_topk:>10}{ring_cap:>10}{cfg.max_protected_ratio:>9.1f}{'OK':>9}")
    except Exception as e:
        ok = False
        print(f"{name:<16}{'-':>10}{'-':>10}{'-':>9}{'FAIL':>9}  ({type(e).__name__}: {e})")

# Mutex: must reject ring>0 AND max_protected<1.0
try:
    FrontendCacheConfig(enabled=True, fifo_protected_ring_ratio=0.3, max_protected_ratio=0.5)
    ok = False; print("\nMUTEX: FAIL (should have raised)")
except ValueError:
    print("\nMUTEX: OK (correctly rejects ring>0 + max_protected<1.0)")

print("\nALL CONFIGS OK" if ok else "\nSOME CONFIGS FAILED")
sys.exit(0 if ok else 1)
