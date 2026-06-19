import sys,os; sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))
import gc,torch,numpy as np
from ovggt.models.ovggt import OVGGT
from ovggt.utils.frontend_cache import FrontendCacheConfig
from ovggt.utils.load_fn import load_and_preprocess_images
from ovggt.utils.pose_enc import pose_encoding_to_extri_intri, ABS_POSE_ENCODING

scenes = [
    'chess/seq-03',
    'fire/seq-03',
    'office/seq-03',
    'redkitchen/seq-03',
]
base='/path/to/mount/lyj/OpenDataLab___7-Scenes/raw'
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

NF=200
print('=== {} frames, {} scenes ==='.format(NF, len(scenes)))
print('{:>20s} | {:>10s} | {:>10s} | {:>10s}'.format('Scene', 'Legacy', 'FE8_orig', 'FE8_opt'))
print('-'*58)

for scene_name in scenes:
    scene=os.path.join(base, scene_name)
    if not os.path.isdir(scene):
        print('{:>20s} | SKIPPED (not found)'.format(scene_name))
        continue
    cfs=sorted([f for f in os.listdir(scene) if f.endswith('.color.png')])[:NF]
    if len(cfs) < NF:
        print('{:>20s} | SKIPPED (only {} frames)'.format(scene_name, len(cfs)))
        continue
    imgs=load_and_preprocess_images([os.path.join(scene,f) for f in cfs]).cuda()
    inputs=[{'img':i.unsqueeze(0)} for i in imgs]
    gt=np.array([np.loadtxt(os.path.join(scene,f.replace('.color.png','.pose.txt'))).astype(np.float32) for f in cfs])
    h,w=imgs.shape[2],imgs.shape[3]

    def run(mk, fn):
        m=mk(); m.load_state_dict(sd,strict=False); m=m.cuda().eval()
        with torch.no_grad(): o=fn(m)
        a=ate(gt,getp(o,h,w)); del m; gc.collect(); torch.cuda.empty_cache()
        return a

    a_leg = run(lambda: OVGGT(mode='legacy',per_layer_budget=8334),
                lambda m: m.inference(inputs,history_anchor_strategy='coverage',anchor_interval=250))

    a_orig = run(lambda: OVGGT(mode='frontend_eval',per_layer_budget=8334,
                frontend_pose_encoding_type=ABS_POSE_ENCODING,
                frontend_cache_config=FrontendCacheConfig(enabled=True,dedup_enabled=True)),
             lambda m: m.inference(inputs,history_anchor_strategy='fixed_interval',anchor_interval=8,max_anchors=3))

    a_opt = run(lambda: OVGGT(mode='frontend_eval',per_layer_budget=8334,
                frontend_pose_encoding_type=ABS_POSE_ENCODING,
                frontend_cache_config=FrontendCacheConfig(enabled=True,dedup_enabled=True,
                    intra_frame_dedup_enabled=True,fifo_keep_topk=80,
                    fifo_protected_ring_ratio=0.2)),
             lambda m: m.inference(inputs,history_anchor_strategy='fixed_interval',anchor_interval=8,max_anchors=3))
    # Production config (verified 2026-06-17): per_layer_budget=8334 (×depth24
    # =200016≈original total 200000; the old 8000 assumed depth=25 → 192000),
    # intra_frame_dedup ON (with the bounded ring below, cache is no longer
    # crowded by stale protected tokens, so intra dedup's redundancy-merging
    # helps again: chess 200f 0.0263 vs 0.0295 OFF, 500f 0.0528 vs 0.0546 OFF;
    # the old "noIntra is better" finding was under fifo80-without-ring + the
    # 192000 budget bug and no longer applies), fifo_keep_topk=80 with a BOUNDED
    # rescued pool (fifo_protected_ring_ratio=0.2 → cap 1666/layer).
    # FrontendCacheConfig defaults budget_allocation='uniform' + ring_ratio=0.2,
    # which is REQUIRED: dynamic allocation dips below protected_count on
    # budget-poor layers and triggers anchor overflow (verified 4.7–10.2%;
    # uniform→0.0%). See docs/p1_ablation_results.md.

    print('{:>20s} | {:10.4f} | {:10.4f} | {:10.4f}'.format(scene_name, a_leg, a_orig, a_opt))
