#!/usr/bin/env python3
"""
Read-only cache diagnostic probe for confirming the P1 "protection crowds the
budget" mechanism.

Attaches to a FrontendEval model via `model._oracle_eviction_probe = probe` and
records, at every eviction event (per frame/layer/batch), the EXACT values that
test the crowding hypothesis:
  - protected_count : cache_state.protected_count (cached field, no GPU sync)
  - num_tokens      : total live tokens in the layer
  - budget          : the per-layer cache_budget passed to eviction
  - anchor_overflow : protected_count >= budget  (protected set leaves <= 0 room
                       for current-frame candidates -> eviction forced to drop
                       current tokens)
  - slot0_count     : tokens with anchor_slot == 0 (the protected pool incl. the
                       global anchor; P1-mechC's ring caps the non-global part)

ZERO PERTURBATION: on_eviction_candidate MUST return None. A non-None return is
treated as a keep-indices override that would ALTER eviction and perturb ATE
(frontend_cache.py:1205). protected_count is read from the precomputed cached
field; the only GPU syncs are a couple of small .sum().item() calls per eviction
event, negligible vs attention compute.

rescued_pool_count is NOT split out exactly here because global_anchor_keyframe_id
is not forwarded to the eviction-probe callback (frontend_cache.py:1197-1204); we
report slot0_count (exact, includes the global anchor) instead. protected_count +
anchor_overflow are the decisive signals and need no anchor id.
"""
from collections import defaultdict


class CacheDiagProbe:
    def __init__(self):
        # records: list of dicts, one per eviction event (frame, layer, batch)
        self.records = []
        # overflow_frames: set of frame_ids where ANY layer overflowed
        self._overflow_frames = set()
        # first_overflow_frame
        self.first_overflow_frame = None

    def on_eviction_candidate(self, cache_state, layer_id, frame_id,
                              budget, batch_index=0):
        # MUST be read-only: return None so eviction behavior is unchanged.
        try:
            protected = int(cache_state.protected_count)
        except Exception:
            protected = -1
        budget = int(budget)
        num_tokens = int(cache_state.num_tokens())
        try:
            md = cache_state.metadata
            slot0_count = int((md.anchor_slot[0] == 0).sum().item())
        except Exception:
            slot0_count = -1
        overflow = 1 if (protected >= 0 and budget > 0 and protected >= budget) else 0
        self.records.append({
            "frame": int(frame_id),
            "layer": int(layer_id),
            "batch": int(batch_index),
            "num_tokens": num_tokens,
            "budget": budget,
            "protected_count": protected,
            "slot0_count": slot0_count,
            "anchor_overflow": overflow,
        })
        if overflow:
            self._overflow_frames.add(int(frame_id))
            if self.first_overflow_frame is None:
                self.first_overflow_frame = int(frame_id)
        return None

    def on_fifo_topk_candidate(self, *args, **kwargs):
        # also attach as fifo probe if desired; stay read-only
        return None

    # ---- post-run rollups ----

    def per_frame_summary(self):
        """max protected_count over layers, and any-layer overflow, per frame."""
        per_frame = {}
        for r in self.records:
            f = r["frame"]
            d = per_frame.setdefault(f, {"max_protected": 0, "max_slot0": 0,
                                         "max_num_tokens": 0, "budget": r["budget"],
                                         "any_overflow": 0, "n_layers": 0})
            d["max_protected"] = max(d["max_protected"], r["protected_count"])
            d["max_slot0"] = max(d["max_slot0"], r["slot0_count"])
            d["max_num_tokens"] = max(d["max_num_tokens"], r["num_tokens"])
            d["any_overflow"] = max(d["any_overflow"], r["anchor_overflow"])
            d["budget"] = r["budget"]
            d["n_layers"] += 1
        return per_frame

    def headline(self):
        pf = self.per_frame_summary()
        if not pf:
            return {"note": "no eviction events recorded (cache never exceeded budget)"}
        frames = sorted(pf)
        max_prot = max(d["max_protected"] for d in pf.values())
        budget = next(iter(pf.values()))["budget"]
        overflow_frames = sorted(self._overflow_frames)

        # Per-RECORD crowding: the worst protected/budget ratio on any single
        # layer/event, and the dynamic per-layer budget distribution. The
        # per_frame_summary mixes layers (max protected vs some layer's budget),
        # so the clean ratio must come from raw records where each protected_count
        # is paired with its OWN layer budget.
        ratios = []
        budgets = []
        for r in self.records:
            if r["budget"] > 0 and r["protected_count"] >= 0:
                ratios.append(r["protected_count"] / r["budget"])
                budgets.append(r["budget"])
        ratios.sort()
        budgets.sort()
        def pct(xs, p):
            if not xs:
                return None
            return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
        # frame at which protected_count first reaches >= budget (overflow onset)
        # and frame at which it crosses 50% / 90% of budget
        first_at_50 = next((f for f in frames if pf[f]["max_protected"] >= 0.5 * budget), None)
        first_at_90 = next((f for f in frames if pf[f]["max_protected"] >= 0.9 * budget), None)
        first_at_100 = next((f for f in frames if pf[f]["max_protected"] >= budget), None)
        return {
            "budget_per_layer (headline ref)": budget,
            "dynamic_budget_p10/p50/p90": [pct(budgets, p) for p in (0.1, 0.5, 0.9)],
            "frames_with_eviction": len(pf),
            "max_protected_count_any_frame": max_prot,
            "max_protected_over_ref_budget (8000)": round(max_prot / 8000, 3),
            "max_per_layer_ratio (worst protected/budget on any single layer)": round(max(ratios), 3) if ratios else None,
            "first_frame_protected_ge_50pct_budget": first_at_50,
            "first_frame_protected_ge_90pct_budget": first_at_90,
            # onset uses the per-RECORD flag (correct: each protected_count vs its
            # OWN layer budget). Do NOT use first_at_100, which compares the
            # cross-layer max protected vs a single layer's budget (misleading).
            "first_overflow_frame (per-record, protected>=that layer budget)": self.first_overflow_frame,
            "n_overflow_frames": len(overflow_frames),
            "overflow_rate (overflow_frames/eviction_frames)": round(len(overflow_frames) / len(pf), 3),
            "first_5_overflow_frames": overflow_frames[:5],
        }
