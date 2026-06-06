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

print('Legacy...')
m=OVGGT(mode='legacy',per_layer_budget=8000)
m.load_state_dict(sd,strict=False); m=m.cuda().eval()
with torch.no_grad(): o=m.inference(inputs,history_anchor_strategy='coverage',anchor_interval=250)
c2w_l=get_poses(o); del m; gc.collect(); torch.cuda.empty_cache()
err_l=align_err(gt,c2w_l)

print('Frontend...')
m=OVGGT(mode='frontend_eval',per_layer_budget=8000,
        frontend_pose_encoding_type=ABS_POSE_ENCODING,
        frontend_cache_config=FrontendCacheConfig(enabled=True))
m.load_state_dict(sd,strict=False); m=m.cuda().eval()
with torch.no_grad(): o=m.inference(inputs,history_anchor_strategy='fixed_interval',anchor_interval=8,max_anchors=3)
c2w_f=get_poses(o); del m; gc.collect(); torch.cuda.empty_cache()
err_f=align_err(gt,c2w_f)

print()
print('{:>5} {:>10} {:>10} {:>6}'.format('Frame','Legacy(m)','Front(m)','Ratio'))
print('-'*35)
for i in range(NF):
    r=err_f[i]/err_l[i] if err_l[i]>1e-6 else 0
    if i<15 or i%10==0 or err_f[i]>0.05:
        print('{:>5} {:>10.4f} {:>10.4f} {:>6.2f}x'.format(i,err_l[i],err_f[i],r))
