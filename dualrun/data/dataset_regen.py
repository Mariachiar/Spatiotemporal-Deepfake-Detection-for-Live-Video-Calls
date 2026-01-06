# data/dataset_video_regen.py
from __future__ import annotations
import os, cv2, random, re, hashlib
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset_dual import _infer_tech_from_path
from data.make_lmk_features import extract_lmk_seq   # frames -> (T,132) float32
from data.make_au_features  import extract_au_seq    # frames -> (T,36)  float32

cv2.setNumThreads(1)

# ---------- util ----------
def _pad_to_len_np(X: np.ndarray, T: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    t = X.shape[0]
    if t >= T: return np.ascontiguousarray(X[:T])
    pad = np.zeros((T - t, X.shape[1]), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate([X, pad], axis=0))

def _zscore_clip_np(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0: return X
    mu = X.mean(axis=0, keepdims=True)
    sd = np.maximum(X.std(axis=0, keepdims=True), 1e-6)
    return ((X - mu) / sd).astype(np.float32, copy=False)

def _zscore_global_np(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return ((X - mean) / np.maximum(std, 1e-6)).astype(np.float32, copy=False)

def _im_jpeg_recompress(im: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok: return im
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return dec if dec is not None else im

def _down_up(im: np.ndarray, s: float) -> np.ndarray:
    h, w = im.shape[:2]
    w2, h2 = max(1, int(w*s)), max(1, int(h*s))
    im2 = cv2.resize(im, (w2, h2), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(im2, (w, h), interpolation=cv2.INTER_LINEAR)

def _offcenter_crop(im: np.ndarray, frac: float) -> np.ndarray:
    if frac <= 0: return im
    h, w = im.shape[:2]
    dx = int(random.uniform(-frac, frac) * w)
    dy = int(random.uniform(-frac, frac) * h)
    x0 = max(0, dx); y0 = max(0, dy)
    x1 = min(w, w + dx); y1 = min(h, h + dy)
    crop = im[y0:y1, x0:x1]
    if crop.size == 0: return im
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

def _letterbox(im: np.ndarray, p: float=0.0) -> np.ndarray:
    if random.random() >= p: return im
    h, w = im.shape[:2]
    t = random.randint(4, max(4, h//32))
    b = random.randint(0, t)
    l = random.randint(0, max(2, w//64))
    r = random.randint(0, max(2, w//64))
    out = im.copy()
    out[:t,:,:] = 0; out[h-b:,:,:] = 0; out[:, :l, :] = 0; out[:, w-r:, :] = 0
    return out

def _gamma_contrast(im: np.ndarray, p: float=0.0) -> np.ndarray:
    if random.random() >= p: return im
    g = 2.0 ** random.uniform(-0.35, 0.35)
    a = 1.0 + random.uniform(-0.15, 0.15)
    b = random.uniform(-8, 8)
    imf = im.astype(np.float32)
    imf = ((imf/255.0) ** g) * 255.0
    imf = a*imf + b
    return np.clip(imf, 0, 255).astype(np.uint8)

def _motion_blur(im: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return im
    kern = np.zeros((k, k), np.float32)
    if random.random() < 0.5: kern[k//2, :] = 1.0
    else:                     kern[:, k//2] = 1.0
    kern /= np.maximum(kern.sum(), 1e-6)
    return cv2.filter2D(im, -1, kern)

def _add_gauss_noise(im: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0: return im
    noise = np.random.normal(0, sigma, im.shape).astype(np.float32)
    out = np.clip(im.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

def _rand_grayscale(im: np.ndarray, p: float) -> np.ndarray:
    if random.random() >= p: return im
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

def _sample_T_indices(nf: int, T: int, temporal_jitter: float=0.0) -> List[int]:
    if nf <= 0: return [0]*T
    stride = max(1, nf // T)
    start = random.randint(0, max(0, nf - stride*T))
    idxs = [min(nf-1, start + i*stride) for i in range(T)]
    if temporal_jitter > 0:
        d = max(1, int(T * temporal_jitter))
        for _ in range(d):
            pos = random.randrange(T)
            idxs[pos] = max(0, min(nf-1, idxs[pos] + random.choice([-1,1])))
    return idxs

def _read_frames_at(path: str, idxs: List[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    last = None
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(i))
        ok, f = cap.read()
        if not ok:
            f = last if last is not None else np.zeros((224,224,3), np.uint8)
        last = f
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

# ---------- dataset ----------
class DualVideoRegenDataset(Dataset):
    """
    Output:
      (A_clean, L_clean, lengths, trk_id, vid_id, y, extras)
    extras: viste noisy per consistency e dominio.
    Simmetrico: degrado label-agnostic su real e fake.
    """
    def __init__(self, video_paths: List[str], T: int,
                 zscore: str="clip", norm_stats_path: str|None=None,
                 jpeg_q: Tuple[int,int]=(5,35),
                 scale: Tuple[float,float]=(0.35,0.95),
                 offcenter: float = 0.12,
                 mblur_k: Tuple[int,int]=(0,15),
                 # nuovi controlli per simulare upload “sporchi”
                 p_letterbox: float = 0.15,
                 p_gamma_contrast: float = 0.25,
                 temporal_jitter: float = 0.15,
                 p_frame_drop: float = 0.10,
                 p_gauss_noise: float = 0.20,
                 gauss_sigma: float = 3.0,
                 p_grayscale: float = 0.05,
                 is_train: bool = True):
        self.video_paths = list(video_paths)
        self.T = int(T)
        self.zscore = str(zscore)
        self.offcenter = float(offcenter)
        self.jpeg_q = tuple(jpeg_q)
        self.scale = tuple(scale)
        self.mblur_k = tuple(mblur_k)
        self.is_train = bool(is_train)
        self.p_letterbox = float(p_letterbox)
        self.p_gamma_contrast = float(p_gamma_contrast)
        self.temporal_jitter = float(temporal_jitter)
        self.p_frame_drop = float(p_frame_drop)
        self.p_gauss_noise = float(p_gauss_noise)
        self.gauss_sigma = float(gauss_sigma)
        self.p_grayscale = float(p_grayscale)

        # compat logging/engine
        self.clip_dirs = list(self.video_paths)
        self.random_crop = False

        # stats globali
        self._global = None
        if self.zscore == "global" and norm_stats_path and os.path.isfile(norm_stats_path):
            self._global = np.load(norm_stats_path)

        # label real/fake (string matching robusto)
        real_tokens = ("real", "__orig__", "original", "pristine", "youtube-real", "celeb-real", "class0")
        self.labels = []
        for p in self.video_paths:
            s = (os.path.basename(p).lower() + " " + os.path.dirname(p).lower())
            self.labels.append(0 if any(t in s for t in real_tokens) else 1)

        # domini DAT
        self.tech_names = [_infer_tech_from_path(p) for p in self.video_paths]
        fake_techs = sorted({t for t,y in zip(self.tech_names, self.labels) if y==1 and t not in ("real","unknown")})
        self.domain_map = {t:i for i,t in enumerate(fake_techs, start=1)}  # 0=real, 1..C-1=fake
        self.n_domains = max(1, len(fake_techs))

        # id video/track
        self.vid_keys, self.trk_keys = [], []
        def _ids_from_path(p: str):
            seg = p.replace("\\","/").split("/")
            track = next((s for s in seg if re.match(r"track_\d+$", s)), None)
            if track:
                i = seg.index(track)
                tech = seg[i-2] if i >= 2 else "unknown"
                vid  = seg[i-1] if i >= 1 else "unknown"
                return f"{tech}/{vid}", f"{tech}/{vid}/{track}"
            parent = "/".join(seg[:-1]) or "unknown"
            h = hashlib.md5(parent.encode("utf-8")).hexdigest()[:10]
            return parent, f"{parent}/track_{h}"

        for p in self.video_paths:
            vk, tk = _ids_from_path(p)
            self.vid_keys.append(vk); self.trk_keys.append(tk)

        uniq_vid = {k:i for i,k in enumerate(sorted(set(self.vid_keys)))}
        uniq_trk = {k:i for i,k in enumerate(sorted(set(self.trk_keys)))}
        self.vid_ids = np.array([uniq_vid[k] for k in self.vid_keys], dtype=np.int64)
        self.trk_ids = np.array([uniq_trk[k] for k in self.trk_keys], dtype=np.int64)

    def __len__(self): return len(self.video_paths)

    # degradazioni label-agnostic
    def _degrade_clip(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not self.is_train:
            return frames
        q  = random.randint(*self.jpeg_q)
        s  = random.uniform(*self.scale)
        mb = random.choice([k for k in range(self.mblur_k[0], self.mblur_k[1]+1, 2)]) if self.mblur_k[1] > 0 else 0
        out = []
        for im in frames:
            im2 = _letterbox(im, self.p_letterbox)
            im2 = _gamma_contrast(im2, self.p_gamma_contrast)
            im2 = _offcenter_crop(im2, self.offcenter)
            im2 = _down_up(im2, s)
            if mb > 0 and random.random() < 0.6:
                im2 = _motion_blur(im2, mb)
            if random.random() < self.p_grayscale:
                im2 = _rand_grayscale(im2, 1.0)
            if random.random() < self.p_gauss_noise:
                im2 = _add_gauss_noise(im2, self.gauss_sigma)
            if random.random() < self.p_frame_drop:
                # simula frame saltati; garantisci almeno un frame
                if len(out) < max(1, len(frames)//6):
                    continue
            im2 = _im_jpeg_recompress(im2, q)
            out.append(im2)
        if not out:  # fallback sicuro
            out = frames
        return out

    def __getitem__(self, i: int):
        path = os.path.normpath(str(self.video_paths[i])).strip().strip('"').lstrip('\ufeff')
        if not os.path.isfile(path):
            raise RuntimeError(f"Video path not found: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {path}")
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); cap.release()
        if nf <= 0: raise RuntimeError(f"Empty/bad video (nf={nf}): {path}")

        idxs = _sample_T_indices(nf, self.T, temporal_jitter=self.temporal_jitter if self.is_train else 0.0)
        frames_clean = _read_frames_at(path, idxs)
        frames_noisy = self._degrade_clip(frames_clean)

        # feature
        L_clean = extract_lmk_seq(frames_clean).astype(np.float32, copy=False)
        A_clean = extract_au_seq(frames_clean).astype(np.float32, copy=False)
        L_noisy = extract_lmk_seq(frames_noisy).astype(np.float32, copy=False)
        A_noisy = extract_au_seq(frames_noisy).astype(np.float32, copy=False)

        # z-score
        if self._global is not None and self.zscore == "global":
            A_clean = _zscore_global_np(A_clean, self._global["au_mean"],  self._global["au_std"])
            L_clean = _zscore_global_np(L_clean, self._global["lmk_mean"], self._global["lmk_std"])
            A_noisy = _zscore_global_np(A_noisy, self._global["au_mean"],  self._global["au_std"])
            L_noisy = _zscore_global_np(L_noisy, self._global["lmk_mean"], self._global["lmk_std"])
        elif self.zscore == "clip":
            A_clean = _zscore_clip_np(A_clean); L_clean = _zscore_clip_np(L_clean)
            A_noisy = _zscore_clip_np(A_noisy); L_noisy = _zscore_clip_np(L_noisy)

        # pad a T
        A_clean = _pad_to_len_np(A_clean, self.T); L_clean = _pad_to_len_np(L_clean, self.T)
        A_noisy = _pad_to_len_np(A_noisy, self.T); L_noisy = _pad_to_len_np(L_noisy, self.T)

        # meta
        y = int(self.labels[i])
        tech = self.tech_names[i]
        dom_id = 0 if y == 0 else self.domain_map.get(tech, 0)

        # tensori principali
        A = torch.from_numpy(A_clean).to(torch.float32)
        L = torch.from_numpy(L_clean).to(torch.float32)
        lengths_t = torch.tensor(self.T, dtype=torch.long)
        trk_id = torch.tensor(int(self.trk_ids[i]), dtype=torch.long)
        vid_id = torch.tensor(int(self.vid_ids[i]), dtype=torch.long)
        y_t = torch.tensor(y, dtype=torch.long)

        # extras per consistency e DAT
        extras = {
            "au_noisy": torch.from_numpy(A_noisy).to(torch.float32),
            "lmk_noisy": torch.from_numpy(L_noisy).to(torch.float32),
            "dom_id": torch.tensor(dom_id, dtype=torch.long),
            "q": torch.tensor(0, dtype=torch.long),   # placeholder qualità
        }
        return (A, L, lengths_t, trk_id, vid_id, y_t, extras)
