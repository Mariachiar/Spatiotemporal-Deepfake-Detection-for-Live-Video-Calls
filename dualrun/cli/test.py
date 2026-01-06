#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMK test: clip → track(mediana) → video(OR).
Label: 1=real, 0=fake.
CSV: per_clip.csv, per_track.csv, per_video.csv, summary.txt
"""

import os, sys, argparse, logging, csv, json
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

LOG = logging.getLogger("test_lmk_eval")

# ---------------- dataset ----------------
_here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(_here)
try:
    from data.vox_ds import VoxLmkDataset, collate_pad
except Exception:
    sys.path.append(os.path.abspath(os.path.join(_here, "..")))
    from data.vox_ds import VoxLmkDataset, collate_pad

# ---------------- modello: prefisso lmk_enc ----------------
try:
    from model.dual_encoder import BranchEncoder
except Exception:
    from model.dual_encoder import BranchEncoder

class LMKDisc(nn.Module):
    def __init__(self, in_dim: int, d_model=256, nhead=4, num_layers=4, dim_ff=512, dropout=0.20):
        super().__init__()
        mlp_ratio = dim_ff / d_model
        self.lmk_enc = BranchEncoder(
            input_dim=in_dim,
            d_model=d_model,
            depth=num_layers,
            heads=nhead,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pool_tau=0.7,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, key_padding_mask=None):
        z = self.lmk_enc(x, key_padding_mask=key_padding_mask)  # (B,D)
        return self.head(z).squeeze(-1)

# ---------------- util ----------------
def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def label_from_dir(d: str) -> int:
    p = d.lower().replace("\\","/")
    toks = set(p.split("/"))
    is_real = any(t in {"real","original","pristine","authentic","youtube-real","celeb-real","webcam"} for t in toks)
    is_fake = any(t in {"fake","deepfakes","faceswap","face2face","faceshifter","neuraltextures","dfdc","celebdf","ffpp"} for t in toks)
    if is_fake and not is_real: return 0
    if is_real and not is_fake: return 1
    return 0

def parse_ids_from_clip_dir(clip_dir: str) -> Tuple[str,str,str]:
    p = clip_dir.replace("\\","/").rstrip("/")
    parts = p.split("/")
    track_idx = None; clip_idx = None
    for i,s in enumerate(parts):
        if s.startswith("track_"): track_idx = i
        if s.startswith("clip_"):  clip_idx = i
    if track_idx is None: raise ValueError(f"track_* non trovato: {clip_dir}")
    video_id = "/".join(parts[:track_idx])
    track_id = "/".join(parts[:track_idx+1])
    clip_id  = "/".join(parts[:clip_idx+1]) if clip_idx is not None else clip_dir
    return video_id, track_id, clip_id

def median_by_key(values: List[float], keys: List[str]) -> Dict[str, float]:
    bins: Dict[str, List[float]] = {}
    for k,v in zip(keys, values):
        bins.setdefault(k, []).append(float(v))
    return {k: float(np.median(vs)) for k,vs in bins.items()}

def noisy_or(p_tracks: List[float]) -> float:
    p = np.clip(np.asarray(p_tracks, np.float64), 1e-6, 1-1e-6)
    return float(1.0 - np.exp(np.log1p(-p).sum()))

def metrics_at_t(y_true: List[int], p: List[float], t: float) -> Dict[str, Any]:
    y = np.asarray(y_true, np.int64); pr = np.asarray(p, np.float64)
    pred = (pr >= t).astype(np.int64)
    tn = int(((pred==0)&(y==0)).sum()); fp = int(((pred==1)&(y==0)).sum())
    fn = int(((pred==0)&(y==1)).sum()); tp = int(((pred==1)&(y==1)).sum())
    acc = (tp+tn)/max(1,len(y)); prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
    f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    tpr = rec; fpr = fp/max(1,fp+tn)
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,acc=acc,precision=prec,recall=rec,f1=f1,TPR=tpr,FPR=fpr,youden=(tpr-fpr))

def auc_roc(y_true: List[int], p: List[float]) -> float:
    y = np.asarray(y_true, np.int64); s = np.asarray(p, np.float64)
    n1 = int((y==1).sum()); n0 = int((y==0).sum())
    if n1==0 or n0==0: return float("nan")
    # rank-based AUC (Mann–Whitney)
    order = np.argsort(s)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(s)) + 1
    sum_ranks_pos = ranks[y==1].sum()
    auc = (sum_ranks_pos - n1*(n1+1)/2) / (n1*n0)
    return float(auc)

def best_youden(y_true: List[int], p: List[float]) -> Dict[str, Any]:
    y = np.asarray(y_true, np.int64); pr = np.asarray(p, np.float64)
    ths = np.unique(pr)
    best = {"t":0.5,"youden":-1,"metrics":{}}
    for t in ths:
        m = metrics_at_t(y, pr, float(t))
        if m["youden"] > best["youden"]:
            best = {"t": float(t), "youden": float(m["youden"]), "metrics": m}
    return best

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser("Test LMK coherence clip→track→video")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--in-dim", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ff-dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tqdm", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--invert", action="store_true", help="Usa 1-score prima dell’aggregazione")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(message)s")
    set_seed(args.seed)
    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    LOG.info(f"Device: {device}")

    # clip
    clip_dirs: List[str] = []
    for root, _, files in os.walk(args.data):
        if "lmk_features.npy" in files:
            try:
                _ = parse_ids_from_clip_dir(root)
                clip_dirs.append(root)
            except Exception:
                continue
    if not clip_dirs: raise SystemExit("Nessuna clip valida trovata.")
    clip_dirs.sort()
    LOG.info(f"Clip: {len(clip_dirs)}")

    # in_dim
    in_dim = args.in_dim
    if in_dim is None:
        in_dim = int(np.load(os.path.join(clip_dirs[0], "lmk_features.npy"), mmap_mode="r").shape[1])
        LOG.info(f"Inferred in_dim={in_dim}")

    # dataset
    paths = [os.path.join(d, "lmk_features.npy") for d in clip_dirs]
    ds = VoxLmkDataset(paths, time_warp=(1.0,1.0))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
                    collate_fn=collate_pad, persistent_workers=False)

    # modello
    model = LMKDisc(
        in_dim=in_dim, d_model=args.d_model, nhead=args.heads,
        num_layers=args.layers, dim_ff=args.ff_dim, dropout=args.dropout
    ).to(device)

    LOG.info(f"Carico checkpoint: {args.model}")
    ck = torch.load(args.model, map_location="cpu")
    sd = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        LOG.warning(f"State dict: missing={missing} unexpected={unexpected}")

    model.eval()

    # inferenza clip
    scores: List[float] = []
    with torch.no_grad():
        it = tqdm(dl, desc="infer", disable=not args.tqdm)
        for X, pad in it:
            X = X.to(device, non_blocking=True)
            pad = pad.to(device, non_blocking=True)
            logits = model(X, key_padding_mask=pad)
            s = torch.sigmoid(logits).clamp(0,1)
            scores.extend(s.detach().cpu().tolist())
    if len(scores) != len(clip_dirs):
        raise SystemExit(f"Mismatch scores={len(scores)} vs clips={len(clip_dirs)}")

    # invert opzionale PRIMA delle aggregazioni
    if args.invert:
        scores = [1.0 - float(s) for s in scores]

    # id e label
    labels = [label_from_dir(d) for d in clip_dirs]
    vids, trks = [], []
    for d in clip_dirs:
        v,t,_ = parse_ids_from_clip_dir(d)
        vids.append(v); trks.append(t)

    # CSV per-clip
    os.makedirs(args.out_dir, exist_ok=True)
    per_clip_csv = os.path.join(args.out_dir, "per_clip.csv")
    with open(per_clip_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["clip_dir","track_id","video_id","label","score"])
        for d,t,v,y,s in zip(clip_dirs, trks, vids, labels, scores):
            w.writerow([d,t,v,int(y),float(s)])
    LOG.info(f"Salvato: {per_clip_csv}")

    # track = mediana clip
    track2p = median_by_key(scores, trks)
    track2y: Dict[str,int] = {}
    track2v: Dict[str,str] = {}
    for d,t,v,y in zip(clip_dirs, trks, vids, labels):
        track2y.setdefault(t, int(y))
        track2v.setdefault(t, v)

    per_track_csv = os.path.join(args.out_dir, "per_track.csv")
    with open(per_track_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["track_id","video_id","label","score_track_median"])
        for t in sorted(track2p.keys()):
            w.writerow([t, track2v[t], track2y[t], float(track2p[t])])
    LOG.info(f"Salvato: {per_track_csv}")

    # video = OR sui track (score continuo = noisy-OR)
    vid2tracks: Dict[str, List[str]] = {}
    for t,v in track2v.items():
        vid2tracks.setdefault(v, []).append(t)

    vid_rows = []
    for v, tlist in vid2tracks.items():
        p_tracks = [track2p[t] for t in tlist]
        p_video = noisy_or(p_tracks)
        y_tracks = [track2y[t] for t in tlist]
        y_video = int(min(y_tracks))  # 0 se esiste un fake
        vid_rows.append((v, y_video, p_video, len(tlist)))
    vid_rows.sort(key=lambda r: r[0])

    per_video_csv = os.path.join(args.out_dir, "per_video.csv")
    with open(per_video_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["video_id","label","score_video_or","n_tracks"])
        for v,y,p,k in vid_rows:
            w.writerow([v,int(y),float(p),int(k)])
    LOG.info(f"Salvato: {per_video_csv}")

    # metriche @ soglia data
    t_fixed = float(args.thresh)
    def _fmt(m: Dict[str,Any]) -> str:
        return (f"acc={m['acc']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} "
                f"f1={m['f1']:.3f} TPR={m['TPR']:.3f} FPR={m['FPR']:.3f} youden={m['youden']:.3f} "
                f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")

    # arrays track/video
    y_trk = [track2y[k] for k in sorted(track2p.keys())]
    p_trk = [track2p[k] for k in sorted(track2p.keys())]
    y_vid = [y for _,y,_,_ in vid_rows]
    p_vid = [p for _,_,p,_ in vid_rows]

    # AUC
    auc_trk = auc_roc(y_trk, p_trk)
    auc_vid = auc_roc(y_vid, p_vid)

    # metriche @ soglia fissa
    m_trk_fixed = metrics_at_t(y_trk, p_trk, t_fixed)
    m_vid_fixed = metrics_at_t(y_vid, p_vid, t_fixed)

    # soglia ottima Youden
    best_trk = best_youden(y_trk, p_trk)
    best_vid = best_youden(y_vid, p_vid)

    # log
    LOG.info(f"[TRACK @t={t_fixed:.2f}] {_fmt(m_trk_fixed)}  (mediana)  AUC={auc_trk:.3f}")
    LOG.info(f"[VIDEO @t={t_fixed:.2f}] {_fmt(m_vid_fixed)}  (OR sui track)  AUC={auc_vid:.3f}")
    LOG.info(f"[TRACK @t*={best_trk['t']:.4f}] {_fmt(best_trk['metrics'])}  (Youden*={best_trk['youden']:.4f})  AUC={auc_trk:.3f}")
    LOG.info(f"[VIDEO @t*={best_vid['t']:.4f}] {_fmt(best_vid['metrics'])}  (Youden*={best_vid['youden']:.4f})  AUC={auc_vid:.3f}")
    LOG.info("Fatto.")

    # summary.txt
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"invert={args.invert}\n")
        f.write(f"TRACK_AUC={auc_trk:.6f}\n")
        f.write(f"VIDEO_AUC={auc_vid:.6f}\n")
        f.write(f"TRACK_t_fixed={t_fixed:.6f}  metrics={json.dumps(m_trk_fixed)}\n")
        f.write(f"VIDEO_t_fixed={t_fixed:.6f}  metrics={json.dumps(m_vid_fixed)}\n")
        f.write(f"TRACK_best_youden_t={best_trk['t']:.6f}  metrics={json.dumps(best_trk['metrics'])}\n")
        f.write(f"VIDEO_best_youden_t={best_vid['t']:.6f}  metrics={json.dumps(best_vid['metrics'])}\n")
    LOG.info(f"Salvato: {summary_path}")

if __name__ == "__main__":
    main()
