import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def old_hash(voxels):
    mult = torch.tensor([1, 1000, 1000000], dtype=torch.long)
    return (voxels.to(torch.long) * mult).sum(-1)


def test_no_collision_large_coords():
    from ovggt.utils.frontend_cache import voxel_hash_collision_free

    torch.manual_seed(0)
    voxels = torch.randint(-2500, 2501, (50000, 3))
    h = voxel_hash_collision_free(voxels)
    vox_unique = len(set(map(tuple, voxels.tolist())))
    hash_unique = len(set(h.tolist()))
    assert hash_unique == vox_unique


def test_no_collision_negative():
    from ovggt.utils.frontend_cache import voxel_hash_collision_free

    v = torch.tensor([[1000, 0, 0], [0, 1, 0], [-5, -5, -5]], dtype=torch.long)
    h = voxel_hash_collision_free(v)
    assert len(set(h.tolist())) == 3
    assert old_hash(v)[0].item() == old_hash(v)[1].item()
