"""TDD RED test for P5 fix: FIFO_SWAP must retain the demoted keyframe's transform.

Drives a real FrontendKeyframeManager to FIFO_SWAP using forced_keyframe_frames,
then verifies (1) slot_pose_updates includes the demoted keyframe, and (2) a
protected token projects correctly after apply_keyframe_event_.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from ovggt.utils.frontend_keyframe import FrontendKeyframeManager, KeyframeSwitchConfig, KeyframeEventType
from ovggt.utils.frontend_cache import TokenMetadata, LayerCacheState


def make_l2w(tx, ty=0.0, tz=0.0):
    """Translation-only local_to_world (4x4)."""
    T = torch.eye(4)
    T[0, 3] = tx; T[1, 3] = ty; T[2, 3] = tz
    return T


def drive_manager_to_fifo_swap():
    """Drive a manager to produce a FIFO_SWAP event.

    max_history_anchors=3 → 4th keyframe (frames 0,1,2,3 all forced) triggers FIFO_SWAP.
    Each keyframe gets a distinct translation so transforms are distinguishable.
    Returns (manager, fifo_event, per_kf_local_to_world).
    """
    cfg = KeyframeSwitchConfig(max_history_anchors=3, strategy="fixed_interval",
                               forced_keyframe_frames=(0, 1, 2, 3, 4))
    mgr = FrontendKeyframeManager(cfg)

    # Distinct translations per keyframe: kf at x = 10*frame
    kf_l2w = {}
    events = []
    image_size_hw = (100, 100)
    depth = torch.ones(1, 1, 100, 100)
    # Build minimal pose_abs_enc via the manager's own pose_encoding round-trip is complex;
    # instead patch current_local_to_world by monkeypatching pose_encoding_to_c2w.
    import ovggt.utils.frontend_keyframe as fkmod

    orig = fkmod.pose_encoding_to_c2w
    frame_to_tx = {0: 0.0, 1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0, 5: 50.0}

    def fake_c2w(pose_enc, image_size_hw):
        # pose_enc carries the frame idx at [0,0,0] (we set it below)
        fi = int(pose_enc[0, 0, 0].item())
        return make_l2w(frame_to_tx[fi])

    fkmod.pose_encoding_to_c2w = fake_c2w
    try:
        for fi in range(5):
            pose_abs_enc = torch.zeros(3, 100, 100)
            pose_abs_enc[0, 0, 0] = float(fi)  # carry frame idx
            ev = mgr.update(fi, depth, pose_abs_enc, image_size_hw)
            events.append(ev)
            kf_l2w[ev.keyframe_id] = make_l2w(frame_to_tx[fi])
    finally:
        fkmod.pose_encoding_to_c2w = orig

    fifo_ev = events[-1]
    assert str(fifo_ev.event_type).endswith("FIFO_SWAP"), f"expected FIFO_SWAP, got {fifo_ev.event_type}"
    return mgr, fifo_ev, kf_l2w


def test_fifo_swap_retains_demoted_transform():
    """After FIFO_SWAP, slot_pose_updates MUST include the demoted keyframe's id."""
    mgr, fifo_ev, kf_l2w = drive_manager_to_fifo_swap()
    # The demoted keyframe is the oldest promoted one (keyframe_id from first PROMOTE after init).
    # init creates kf0 (global). PROMOTEs create kf1,kf2,kf3. FIFO_SWAP demotes the oldest history slot.
    promote_events = [e for e in [fifo_ev]]  # fifo_ev is the 4th
    # Demoted keyframe id = the first history slot's keyframe_id = 1 (kf0=global/init, kf1=first promote)
    demoted_kf_id = 1
    print(f"FIFO_SWAP event slot_pose_updates keys: {list(fifo_ev.slot_pose_updates.keys())}")
    print(f"Demoted keyframe id: {demoted_kf_id}")
    assert demoted_kf_id in fifo_ev.slot_pose_updates, (
        f"P5 BUG: demoted keyframe {demoted_kf_id} missing from slot_pose_updates "
        f"(keys={list(fifo_ev.slot_pose_updates.keys())})"
    )


def test_protected_token_projects_correctly_after_fifo():
    """End-to-end: a token protected from the demoted keyframe projects correctly
    after apply_keyframe_event_."""
    mgr, fifo_ev, kf_l2w = drive_manager_to_fifo_swap()
    demoted_kf_id = 1

    # Build a cache with 1 token from the demoted keyframe (kf1).
    # slot_local_xyz = (1,0,0) in kf1's local frame. kf1 is at x=10 in world.
    # Correct active projection depends on new active (kf4 at x=40):
    #   world = kf1_l2w @ local = (11,0,0)
    #   active = inv(kf4_l2w) @ world = (11-40,0,0) = (-29,0,0)
    meta = TokenMetadata(
        token_kind=torch.tensor([[2]]),
        frame_id=torch.tensor([[1]]),
        anchor_slot=torch.tensor([[1]]),       # in demoted slot
        keyframe_id=torch.tensor([[demoted_kf_id]]),
        slot_id=torch.tensor([[demoted_kf_id]]),
        slot_local_xyz=torch.tensor([[[1.0, 0.0, 0.0]]]),
        importance=torch.tensor([[0.9]]),
        depth_conf=torch.tensor([[1.0]]),
    )
    cs = LayerCacheState(max_history_anchors=3)
    cs.k = torch.randn(1, 4, 1, 8); cs.v = torch.randn(1, 4, 1, 8)
    cs.metadata = meta
    cs._cached_protected_count = cs._compute_protected_count_raw()
    cs.protected_count = cs._cached_protected_count

    # protect_topk: move token to slot 0 (anchor_slot=0), slot_id stays = demoted_kf_id
    cs.protect_topk_on_demotion_(demoted_slot=1, keep_count=1)
    # apply the FIFO_SWAP event (replaces slot_to_active)
    cs.apply_keyframe_event_(fifo_ev)

    proj = cs._project_slot_local_xyz_to_active(cs.metadata.slot_local_xyz, cs.metadata.slot_id)
    got = proj[0, 0].tolist()
    expected = [-29.0, 0.0, 0.0]
    print(f"Protected token projection after FIFO_SWAP: {got}  (expected {expected})")
    for g, e in zip(got, expected):
        assert abs(g - e) < 1e-3, f"P5 BUG: projection {got} != expected {expected}"


def test_demoted_keyframe_transform_is_recomputed_on_later_promotion():
    """A demoted keyframe transform must stay valid when a later keyframe becomes active."""
    mgr, fifo_ev, _ = drive_manager_to_fifo_swap()
    demoted_kf_id = 1

    meta = TokenMetadata(
        token_kind=torch.tensor([[2]]),
        frame_id=torch.tensor([[1]]),
        anchor_slot=torch.tensor([[-1]]),
        keyframe_id=torch.tensor([[demoted_kf_id]]),
        slot_id=torch.tensor([[demoted_kf_id]]),
        slot_local_xyz=torch.tensor([[[1.0, 0.0, 0.0]]]),
        importance=torch.tensor([[0.9]]),
        depth_conf=torch.tensor([[1.0]]),
    )
    cs = LayerCacheState(max_history_anchors=3)
    cs.k = torch.randn(1, 4, 1, 8)
    cs.v = torch.randn(1, 4, 1, 8)
    cs.metadata = meta
    cs._cached_protected_count = cs._compute_protected_count_raw()
    cs.protected_count = cs._cached_protected_count
    cs.apply_keyframe_event_(fifo_ev)

    import ovggt.utils.frontend_keyframe as fkmod

    orig = fkmod.pose_encoding_to_c2w

    def fake_c2w(pose_enc, image_size_hw):
        fi = int(pose_enc[0, 0, 0].item())
        return make_l2w({5: 50.0}[fi])

    fkmod.pose_encoding_to_c2w = fake_c2w
    try:
        mgr.config.forced_keyframe_frames = tuple(mgr.config.forced_keyframe_frames) + (5,)
        depth = torch.ones(1, 1, 100, 100)
        pose_abs_enc = torch.zeros(3, 100, 100)
        pose_abs_enc[0, 0, 0] = 5.0
        promote_ev = mgr.update(5, depth, pose_abs_enc, (100, 100))
    finally:
        fkmod.pose_encoding_to_c2w = orig

    cs.apply_keyframe_event_(promote_ev)
    proj = cs._project_slot_local_xyz_to_active(cs.metadata.slot_local_xyz, cs.metadata.slot_id)
    got = proj[0, 0].tolist()
    expected = [-39.0, 0.0, 0.0]
    for g, e in zip(got, expected):
        assert abs(g - e) < 1e-3, f"expected later promotion projection {expected}, got {got}"


if __name__ == "__main__":
    print("="*60); print("Test 1: FIFO_SWAP retains demoted transform"); print("="*60)
    try:
        test_fifo_swap_retains_demoted_transform()
        print("PASS\n")
    except AssertionError as e:
        print(f"FAIL (expected RED): {e}\n")

    print("="*60); print("Test 2: protected token projects correctly"); print("="*60)
    try:
        test_protected_token_projects_correctly_after_fifo()
        print("PASS\n")
    except AssertionError as e:
        print(f"FAIL (expected RED): {e}\n")
