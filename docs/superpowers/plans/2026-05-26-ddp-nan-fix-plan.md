# DDP Deadlock & NaN Loss Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix DDP 4-GPU training deadlock by implementing a three-layer NaN defense (teacher output sanitize → loss numerical stability → DDP safety net).

**Architecture:** Three independent defense layers, each modifying a specific code location. Execution order: Layer 3 (safety net, highest impact per line changed) → Layer 2 (loss stability) → Layer 1 (teacher sanitize). No new files, no new dependencies.

**Tech Stack:** PyTorch, HuggingFace Accelerate, bf16 mixed precision

**Spec:** `docs/superpowers/specs/2026-05-26-ddp-nan-fix-design.md`

---

### Task 1: Layer 3 — DDP Safety Net

**Files:**
- Modify: `src/train_frontend.py:597-599`

Fix the broken `loss * 0 + 0` (does nothing for NaN in IEEE 754) to use `torch.nan_to_num`.

- [ ] **Step 1: Replace the NaN handling code**

Read `src/train_frontend.py` lines 595-610, then replace:

```python
            if not math.isfinite(loss_value):
                printer.warning("Replacing non-finite loss with zero: loss=%s, details=%s", loss_value, loss_details)
                loss = loss * 0 + 0  # keep graph, zero value, so backward still runs for DDP sync
```

with:

```python
            if not math.isfinite(loss_value):
                printer.warning(
                    "Replacing non-finite loss with zero: loss=%s, rank=%s, step=%s, details=%s",
                    loss_value, accelerator.process_index, step, loss_details,
                )
                loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
```

- [ ] **Step 2: Verify the edit**

```bash
grep -n "nan_to_num" src/train_frontend.py
```
Expected: one match at the line you just edited.

- [ ] **Step 3: Commit**

```bash
git add src/train_frontend.py
git commit -m "fix: replace broken NaN->0 workaround with torch.nan_to_num (Layer 3 DDP safety net)

loss * 0 + 0 does not fix NaN in IEEE 754 (NaN * 0 = NaN).
torch.nan_to_num correctly replaces NaN/Inf with zero while preserving the
autograd graph, ensuring DDP all-reduce synchronizes across all ranks."
```

---

### Task 2: Layer 2b — `closed_form_scale_and_shift` NaN Guard

**Files:**
- Modify: `src/dust3r/losses.py:1078-1117`

Add NaN fallback to identity (scale=1, shift=0) when inputs are all-zero or degenerate.

- [ ] **Step 1: Read the current function**

Read `src/dust3r/losses.py` lines 1078-1117 to confirm exact content.

- [ ] **Step 2: Add NaN guards after scale/shift computation**

In `closed_form_scale_and_shift`, after computing `scale` and `shift` for both C==1 and C==3 branches, add the same two-line guard. The guard must be added in **both** branches (C==1 at ~line 1104, C==3 at ~line 1114).

For the **C==1 branch** (after `shift = gt_mean - scale * pred_mean`):

```python
        shift = gt_mean - scale * pred_mean
        # Guard: fall back to identity if computation produced NaN (e.g. all-zero input)
        scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
        shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
        return scale, shift
```

For the **C==3 branch** (after `shift = gt_mean - scale * pred_mean`):

```python
        shift = gt_mean - scale * pred_mean
        # Guard: fall back to identity if computation produced NaN (e.g. all-zero input)
        scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
        shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
        return scale, shift
```

- [ ] **Step 3: Verify both branches have the guard**

```bash
grep -c "torch.isfinite(scale)" src/dust3r/losses.py
```
Expected: `2` (one per branch)

- [ ] **Step 4: Commit**

```bash
git add src/dust3r/losses.py
git commit -m "fix: add NaN guard in closed_form_scale_and_shift for degenerate inputs

When pred/gt are all-zero (corrupted depth), the scale computation produces
0/0 = NaN. Guard falls back to identity (scale=1, shift=0) to prevent NaN
from propagating into DepthOrPmapLoss."
```

---

### Task 3: Layer 2a — `DepthOrPmapLoss.forward` Input/Output Sanitize

**Files:**
- Modify: `src/dust3r/losses.py:1265-1291`

Add input clamp at entry and output NaN guard at exit.

- [ ] **Step 1: Read the current forward method**

Read `src/dust3r/losses.py` lines 1265-1291.

- [ ] **Step 2: Add input sanitize at the top of forward()**

Replace the beginning of `forward` (first 2 lines after `def forward`):

```python
    def forward(self, pred, gt, sigma_p, sigma_g, valid_mask):
        if self.training:
```

with:

```python
    def forward(self, pred, gt, sigma_p, sigma_g, valid_mask):
        # Sanitize: clamp extreme values to prevent inf/nan propagation
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        gt = torch.nan_to_num(gt, nan=0.0, posinf=1e4, neginf=-1e4)
        pred = pred.clamp(-1e4, 1e4)
        gt = gt.clamp(-1e4, 1e4)

        if self.training:
```

- [ ] **Step 3: Add output NaN guard at the end of forward()**

Replace the return statement:

```python
        return self.gamma * main_loss + grad_loss + reg_loss
```

with:

```python
        loss = self.gamma * main_loss + grad_loss + reg_loss
        # Safety: replace NaN/Inf with zero while preserving autograd graph
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
        return loss
```

- [ ] **Step 4: Verify the edits**

```bash
grep -n "clamp(-1e4, 1e4)" src/dust3r/losses.py
grep -n "torch.zeros_like(loss)" src/dust3r/losses.py
```
Expected: 2 matches for clamp (pred and gt), 1 match for zeros_like.

- [ ] **Step 5: Commit**

```bash
git add src/dust3r/losses.py
git commit -m "fix: add input sanitize and output NaN guard in DepthOrPmapLoss

Clamp pred/gt extreme values (>1e4) at entry to prevent inf from entering
normalize_pointcloud and closed_form_scale_and_shift. Replace NaN/Inf loss
with zero tensor (preserving autograd) at exit as a final safety net."
```

---

### Task 4: Layer 1 — Teacher Output Sanitize

**Files:**
- Modify: `src/train_frontend.py:330-370`

Add `_sanitize_teacher_outputs()` helper and call it after `teacher.inference()` in `frontend_loss_of_one_batch`.

- [ ] **Step 1: Read the context around the insertion points**

Read `src/train_frontend.py` lines 325-375 to see the `frontend_loss_of_one_batch` function signature and the teacher inference section.

- [ ] **Step 2: Add the helper function**

Add `_sanitize_teacher_outputs()` as a module-level function right before `frontend_loss_of_one_batch` (around line 329). Insert:

```python
def _sanitize_teacher_outputs(teacher_outputs):
    """Clamp inf/nan in teacher depth/pmap to prevent NaN in downstream loss."""
    for i, pred in enumerate(teacher_outputs.ress):
        for key in ("depth", "pts3d_in_other_view"):
            if key in pred:
                t = pred[key]
                if not torch.isfinite(t).all():
                    import logging
                    _logger = logging.getLogger(__name__)
                    _logger.warning(
                        "Teacher %s has inf/nan at frame %d, clamping to finite range.", key, i
                    )
                    pred[key] = torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)
                    pred[key] = pred[key].clamp(-1e4, 1e4)
```

- [ ] **Step 3: Call the helper after teacher inference**

In `frontend_loss_of_one_batch`, after the `teacher_outputs = teacher.inference(...)` block (after the `if teacher_output_to_cpu:` block at ~line 372), add the sanitize call. Insert **before** the `if teacher_weight_offload and get_module_device(teacher).type == "cuda":` line:

```python
        _sanitize_teacher_outputs(teacher_outputs)
```

- [ ] **Step 4: Verify the edit**

```bash
grep -n "_sanitize_teacher_outputs" src/train_frontend.py
```
Expected: 2 matches (function definition + call site).

- [ ] **Step 5: Verify the file still parses**

```bash
python -c "import ast; ast.parse(open('src/train_frontend.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/train_frontend.py
git commit -m "fix: add teacher output sanitize to prevent NaN from entering loss (Layer 1)

Check teacher depth and pts3d_in_other_view for inf/nan after inference.
Clamp to [-1e4, 1e4] to prevent corrupted teacher predictions from
propagating NaN through DepthOrPmapLoss into DDP all-reduce."
```

---

### Task 5: Integration Verification

**Files:** None (read-only)

Quick smoke-test to verify all changes are consistent.

- [ ] **Step 1: Verify git log shows all 4 commits**

```bash
git log --oneline -4
```
Expected: 4 commits in order (Task 1 through Task 4).

- [ ] **Step 2: Verify no unintended files were modified**

```bash
git diff --stat main...HEAD
```
Expected: only `src/train_frontend.py` and `src/dust3r/losses.py` modified, no other files.

- [ ] **Step 3: Visual review of all changes**

```bash
git diff main...HEAD
```
Expected: changes match the 4 tasks above.
- `loss * 0 + 0` → `torch.nan_to_num(loss, ...)`
- `closed_form_scale_and_shift` has `torch.where(torch.isfinite(scale), ...)` in both branches
- `DepthOrPmapLoss.forward` has clamp at entry + `torch.zeros_like` guard at exit
- `_sanitize_teacher_outputs` function defined and called after `teacher.inference()`

- [ ] **Step 4: Quick Python import check**

```bash
python -c "
import ast
for f in ['src/train_frontend.py', 'src/dust3r/losses.py']:
    ast.parse(open(f).read())
    print(f'{f}: OK')
"
```
Expected: both files parse successfully.
