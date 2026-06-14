"""TDD RED test for P4 fix: voxel hash must not collide for large/negative coords.

P4 (confirmed conditional): hash x + 1000y + 1000000z collides when |voxel|>=1000
(i.e., scene scale >= 100m at voxel_size=0.1). Two distinct voxels map to the same
hash → wrong dedup. Fix: use a collision-free hash.
"""
import sys
sys.path.insert(0, "/path/to/mount/lyj/voxel-vggt/src")
import torch


def old_hash(voxels):
    mult = torch.tensor([1, 1000, 1000000], dtype=torch.long)
    return (voxels.to(torch.long) * mult).sum(-1)


def test_no_collision_large_coords():
    """The hash used by _dedup_single_batch must be collision-free for the voxel
    ranges that occur in real scenes (>= 100m)."""
    # Import the helper that the fix should expose
    from ovggt.utils.frontend_cache import voxel_hash_collision_free

    # Voxel coords spanning a large outdoor scene: ±2500 (250m at voxel_size=0.1)
    torch.manual_seed(0)
    voxels = torch.randint(-2500, 2501, (50000, 3))
    h = voxel_hash_collision_free(voxels)
    # Two distinct voxels must NEVER share a hash
    vox_unique = len(set(map(tuple, voxels.tolist())))
    hash_unique = len(set(h.tolist()))
    print(f"voxels: {vox_unique} unique, hashes: {hash_unique} unique")
    assert hash_unique == vox_unique, (
        f"P4 NOT FIXED: {vox_unique} voxels → only {hash_unique} hashes "
        f"({vox_unique - hash_unique} collisions)"
    )
    print(">>> P4 fix verified: 无碰撞 (大场景坐标)")


def test_no_collision_negative():
    """Explicit collision case: (1000,0,0) vs (0,1,0) must NOT collide."""
    from ovggt.utils.frontend_cache import voxel_hash_collision_free
    v = torch.tensor([[1000, 0, 0], [0, 1, 0], [-5, -5, -5]], dtype=torch.long)
    h = voxel_hash_collision_free(v)
    print(f"voxels {v.tolist()} → hashes {h.tolist()}")
    assert len(set(h.tolist())) == 3, f"P4 NOT FIXED: collision among {h.tolist()}"
    # also confirm the OLD hash DID collide (sanity)
    assert old_hash(v)[0].item() == old_hash(v)[1].item(), "old hash should collide here"
    print(">>> P4 fix verified: 负坐标/大坐标无碰撞 (旧 hash 会碰撞)")


if __name__ == "__main__":
    print("="*60); print("Test 1: no collision for large coords"); print("="*60)
    try:
        test_no_collision_large_coords(); print("PASS\n")
    except (AssertionError, ImportError) as e:
        print(f"FAIL (expected RED): {e}\n")
    print("="*60); print("Test 2: explicit negative/large collision case"); print("="*60)
    try:
        test_no_collision_negative(); print("PASS\n")
    except (AssertionError, ImportError) as e:
        print(f"FAIL (expected RED): {e}\n")
