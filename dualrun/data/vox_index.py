# vox_index.py
import os, re, numpy as np
SPEAKER_RE = re.compile(r"/mp4/(id\d{5})/")
def ok(p, tmin=8):
    try:
        X=np.load(p, mmap_mode="r"); 
        return X.ndim==2 and X.shape[0]>=tmin and np.isfinite(X).all() and X.std(0).any()
    except: return False

def speaker_of(path:str):
    m=SPEAKER_RE.search(path.replace("\\","/"))
    return m.group(1) if m else "unknown"

def build_index(root:str, tmin=8, val_speakers_ratio=0.05, seed=42):
    paths=[]
    for dp,_,fs in os.walk(root):
        if "lmk_features.npy" in fs:
            p=os.path.realpath(os.path.join(dp,"lmk_features.npy"))
            if ok(p, tmin): paths.append(p)
    # group by speaker
    by_sp={}
    for p in paths: by_sp.setdefault(speaker_of(p), []).append(p)
    # split by speaker
    import random; random.seed(seed)
    speakers=list(by_sp.keys()); speakers.sort(); random.shuffle(speakers)
    n_val=max(1, int(len(speakers)*val_speakers_ratio))
    val_sp=set(speakers[:n_val])
    train=[p for sp,ps in by_sp.items() if sp not in val_sp for p in ps]
    val=[p for sp,ps in by_sp.items() if sp in val_sp for p in ps]
    return train, val
