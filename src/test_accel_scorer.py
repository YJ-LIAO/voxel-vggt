"""Quick test: verify accelerate backward works with VisionSelector."""
import sys; sys.path.insert(0, '.')
import torch
from accelerate import Accelerator
from ovggt.models.ovggt import OVGGT
from ovggt.losses.frontend_supervised import FrontendSupervisedLoss

accel = Accelerator(mixed_precision="bf16")
device = accel.device

model = OVGGT(
    mode="frontend_train",
    frontend_pose_encoding_type="relT_quaR_FoV",
    per_layer_budget=434,
    camera_budget=128,
    use_token_scorer=True,
).to(device)

for n, p in model.named_parameters():
    p.requires_grad = "token_scorer" in n

scorer_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(scorer_params, lr=1e-4)

criterion = FrontendSupervisedLoss().to(device)

# Load one real batch
from dust3r.datasets.blendedmvs import BlendedMVS_Multi
from torch.utils.data import DataLoader

ds = BlendedMVS_Multi(
    allow_repeat=True, split="train",
    ROOT="/path/to/mount/lyj/blendedmvs_processed",
    aug_crop=16, resolution=[(518, 392)],
    transform=None, num_views=2, n_corres=0,
)
dl = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=lambda x: x)
batch = next(iter(dl))

model, optimizer, dl = accel.prepare(model, optimizer, dl)
model.train()

print(f"Batch: {len(batch)} views, keys: {list(batch[0].keys())[:5]}")

# Forward
with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
    output = model(batch)
    print(f"Output: {len(output.ress)} frames")
    with torch.amp.autocast(device_type="cuda", enabled=False):
        loss, details = criterion(batch, output, output.keyframe_schedule)
    print(f"Loss: {loss.item():.4f}, grad_fn: {loss.grad_fn is not None}, requires_grad: {loss.requires_grad}")

# Backward
accel.backward(loss)
print("Backward OK!")

# Check gradients
grad_count = sum(1 for p in scorer_params if p.grad is not None and p.grad.abs().sum() > 0)
print(f"Scorer params with gradients: {grad_count}/{len(scorer_params)}")
