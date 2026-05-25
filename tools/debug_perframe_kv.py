import sys,os; sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))
import gc,torch,numpy as np
from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.load_fn import load_and_preprocess_images
from ovggt.utils.pose_enc import pose_encoding_to_extri_intri, ABS_POSE_ENCODING

scene='/path/to/mount/lyj/OpenDataLab___7-Scenes/raw/chess/seq-03'
ckpt='/mnt/lyj/workspace/StreamVGGT/ckpt/checkpoints.pth'
sd=torch.load(ckpt,map_location='cpu',weights_only=False)
if isinstance(sd,dict) and 'model' in sd: sd=sd['model']

NF=100
cfs=sorted([f for f in os.listdir(scene) if f.endswith('.color.png')])[:NF]
imgs=load_and_preprocess_images([os.path.join(scene,f) for f in cfs]).cuda()
inputs=[{'img':i.unsqueeze(0)} for i in imgs]
gt=np.array([np.loadtxt(os.path.join(scene,f.replace('.color.png','.pose.txt'))).astype(np.float32) for f in cfs])
h,w=imgs.shape[2],imgs.shape[3]

def get_poses(out):
    pe=torch.cat([r['camera_pose'] for r in out.ress],0)
    ext,_=pose_encoding_to_extri_intri(pe.unsqueeze(0),image_size_hw=(h,w))
    ext=ext.squeeze(0).cpu().numpy()
    N=ext.shape[0]; w2c=np.eye(4,dtype=np.float32)[None].repeat(N,0); w2c[:,:3,:]=ext; return np.linalg.inv(w2c)

def align_err(gt,pred):
    n=min(len(gt),len(pred))
    gp,pp=gt[:n,:3,3],pred[:n,:3,3]
    gm,pm=gp.mean(0),pp.mean(0); gc,pc=gp-gm,pp-pm
    H=pc.T@gc; U,S,Vh=np.linalg.svd(H); R=Vh.T@U.T
    if np.linalg.det(R)<0: Vh[2]*=-1; R=Vh.T@U.T
    s=S.sum()/(np.trace(pc.T@pc)+1e-8)
    al=(s*(R@pp.T)).T+(gm-s*R@pm)
    return np.linalg.norm(al-gp,axis=1)

# Frontend with keyframe interval=8
print('Frontend (interval=8)...')
m=OVGGT(mode='frontend_eval',total_budget=200000,
        frontend_pose_encoding_type=ABS_POSE_ENCODING,
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=True))
m.load_state_dict(sd,strict=False); m=m.cuda().eval()
with torch.no_grad(): o1=m.inference(inputs,history_anchor_strategy='fixed_interval',anchor_interval=8,max_anchors=3)
c2w_1=get_poses(o1); err_1=align_err(gt,c2w_1)
del m; gc.collect(); torch.cuda.empty_cache()

# Frontend with keyframe interval=100 (essentially no keyframe events in 100 frames)
print('Frontend (interval=100, no keyframe events)...')
m=OVGGT(mode='frontend_eval',total_budget=200000,
        frontend_pose_encoding_type=ABS_POSE_ENCODING,
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=True))
m.load_state_dict(sd,strict=False); m=m.cuda().eval()
with torch.no_grad(): o2=m.inference(inputs,history_anchor_strategy='fixed_interval',anchor_interval=100,max_anchors=3)
c2w_2=get_poses(o2); err_2=align_err(gt,c2w_2)
del m; gc.collect(); torch.cuda.empty_cache()

# Frontend with dedup DISABLED
print('Frontend (interval=8, dedup=OFF)...')
m=OVGGT(mode='frontend_eval',total_budget=200000,
        frontend_pose_encoding_type=ABS_POSE_ENCODING,
        frontend_cache_config=FrontendCacheConfig(enabled=True, dedup_enabled=False))
m.load_state_dict(sd,strict=False); m=m.cuda().eval()
with torch.no_grad(): o3=m.inference(inputs,history_anchor_strategy='fixed_interval',anchor_interval=8,max_anchors=3)
c2w_3=get_poses(o3); err_3=align_err(gt,c2w_3)
del m; gc.collect(); torch.cuda.empty_cache()

print()
print('{:>5} {:>12} {:>12} {:>12}'.format('Frame','int=8','int=100','int=8/noDedup'))
print('-'*45)
for i in range(NF):
    if i<10 or i%10==0 or err_1[i]>0.03:
        print('{:>5} {:>12.4f} {:>12.4f} {:>12.4f}'.format(i, err_1[i], err_2[i], err_3[i]))

print()
ate_1=np.sqrt(np.mean(err_1**2))
ate_2=np.sqrt(np.mean(err_2**2))
ate_3=np.sqrt(np.mean(err_3**2))
print('ATE RMSE:  int=8: {:.4f}m | int=100: {:.4f}m | int=8/noDedup: {:.4f}m'.format(ate_1, ate_2, ate_3))
