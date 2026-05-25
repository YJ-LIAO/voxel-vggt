import sys,os; sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))
import gc,torch,numpy as np
from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.load_fn import load_and_preprocess_images
from ovggt.utils.pose_enc import pose_encoding_to_extri_intri, ABS_POSE_ENCODING

scene='/path/to/mount/lyj/OpenDataLab___7-Scenes/raw/chess/seq-03'
sd=torch.load('/mnt/lyj/workspace/StreamVGGT/ckpt/checkpoints.pth',map_location='cpu',weights_only=False)
if isinstance(sd,dict) and 'model' in sd: sd=sd['model']
cfs=sorted([f for f in os.listdir(scene) if f.endswith('.color.png')])[:10]
imgs=load_and_preprocess_images([os.path.join(scene,f) for f in cfs]).cuda()
inputs=[{'img':i.unsqueeze(0)} for i in imgs]

m=OVGGT(mode='frontend_eval',total_budget=200000,
         frontend_pose_encoding_type=ABS_POSE_ENCODING,
         frontend_cache_config=FrontendCacheConfig(enabled=True,dedup_enabled=True))
m.load_state_dict(sd,strict=False); m=m.cuda().eval()
with torch.no_grad():
    o=m.inference(inputs,history_anchor_strategy='fixed_interval',anchor_interval=8,max_anchors=3)
pe=torch.cat([r['camera_pose'] for r in o.ress],0)
ext,_=pose_encoding_to_extri_intri(pe.unsqueeze(0),image_size_hw=(imgs.shape[2],imgs.shape[3]))
print('OK, 10 frames, no crash. Poses:', ext.shape)
