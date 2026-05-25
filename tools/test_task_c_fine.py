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

def getp(o,h,w):
    pe=torch.cat([r['camera_pose'] for r in o.ress],0)
    ext,_=pose_encoding_to_extri_intri(pe.unsqueeze(0),image_size_hw=(h,w))
    ext=ext.squeeze(0).cpu().numpy(); N=ext.shape[0]
    w2c=np.eye(4,dtype=np.float32)[None].repeat(N,0); w2c[:,:3,:]=ext; return np.linalg.inv(w2c)

def ate(gt,p):
    n=min(len(gt),len(p)); gp,pp=gt[:n,:3,3],p[:n,:3,3]
    gm,pm=gp.mean(0),pp.mean(0); gc,pc=gp-gm,pp-pm
    H=pc.T@gc; U,S,Vh=np.linalg.svd(H); R=Vh.T@U.T
    if np.linalg.det(R)<0: Vh[2]*=-1; R=Vh.T@U.T
    s=S.sum()/(np.trace(pc.T@pc)+1e-8); al=(s*(R@pp.T)).T+(gm-s*R@pm)
    return float(np.sqrt(np.mean(np.sum((al-gp)**2,axis=1))))

for NF in [50,200]:
    cfs=sorted([f for f in os.listdir(scene) if f.endswith('.color.png')])[:NF]
    imgs=load_and_preprocess_images([os.path.join(scene,f) for f in cfs]).cuda()
    inputs=[{'img':i.unsqueeze(0)} for i in imgs]
    gt=np.array([np.loadtxt(os.path.join(scene,f.replace('.color.png','.pose.txt'))).astype(np.float32) for f in cfs])
    h,w=imgs.shape[2],imgs.shape[3]

    def run(mk, fn):
        m=mk(); m.load_state_dict(sd,strict=False); m=m.cuda().eval()
        with torch.no_grad(): o=fn(m)
        a=ate(gt,getp(o,h,w)); del m; gc.collect(); torch.cuda.empty_cache()
        return a

    print('--- {} frames ---'.format(NF))
    for topk in [80, 90, 100, 110, 125]:
        a = run(lambda topk=topk: OVGGT(mode='frontend_eval',total_budget=200000,
                    frontend_pose_encoding_type=ABS_POSE_ENCODING,
                    frontend_cache_config=FrontendCacheConfig(enabled=True,dedup_enabled=True,
                        intra_frame_dedup_enabled=False,fifo_keep_topk=topk)),
                 lambda m: m.inference(inputs,history_anchor_strategy='fixed_interval',anchor_interval=8,max_anchors=3))
        print('  noIntra_fifo={:3d}: ATE={:.4f}'.format(topk, a))
