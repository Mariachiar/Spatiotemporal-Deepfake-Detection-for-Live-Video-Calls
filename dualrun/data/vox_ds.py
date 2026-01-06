# vox_ds.py
import numpy as np, torch
from torch.utils.data import Dataset

class VoxLmkDataset(Dataset):
    def __init__(self, paths, time_warp=(1.0,1.0)):
        self.paths=paths; self.warp=time_warp
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        X=np.load(self.paths[i], mmap_mode="r").astype(np.float32)
        a,b=self.warp
        if a<1.0 or b>1.0:  # leggero time-warp
            r=np.random.uniform(a,b)
            t=np.arange(0, X.shape[0], r, dtype=np.float32)
            idx=np.clip(np.round(t).astype(int), 0, X.shape[0]-1)
            X=X[idx]
        return X

def collate_pad(batch):
    xs=[torch.from_numpy(b) for b in batch]
    T=max(x.size(0) for x in xs); D=xs[0].size(1)
    out=torch.zeros(len(xs),T,D)
    pad=torch.ones(len(xs),T,dtype=torch.bool)  # True=PAD
    for i,x in enumerate(xs):
        L=x.size(0); out[i,:L]=x; pad[i,:L]=False
    return out, pad
