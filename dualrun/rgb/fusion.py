#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning SOLO Dual(AU+LMK, T=8) a livello VIDEO.
Aggregazioni coerenti con AltFreezing:
  clip -> track/persona : MEDIAN
  track -> video        : MAX
Loss = BCE_video(y) + α * BCE_video(teacher_csv_prob)

Valutazione: SOLO Dual (nessun ramo RGB né testa).
Inferenza: SOLO Dual con output per-track e per-video.
"""
import os, re, csv, json, glob, argparse, sys, time, math, logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from types import SimpleNamespace
from collections import Counter
# ---------------------------------------------------------------------
# Path + logger
# ---------------------------------------------------------------------
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, ROOT):
    if p not in sys.path: sys.path.insert(0, p)

def setup_logger(level: str = "INFO", to_file: Optional[str] = None):
    lvl = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if to_file: handlers.append(logging.FileHandler(to_file))
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers
    )
    logging.info(f"log_level={level} file={to_file or '-'}")

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def sigmoid_to_logit(p: float, eps: float = 1e-6) -> float:
    p = max(eps, min(1 - eps, float(p)))
    return float(math.log(p / (1 - p)))

def get_person_id_from_path(p: str) -> str:
    toks = re.split(r"[\\/]", p)
    # path: .../<manip>/<video>/<persona>/<clip*>
    for i in range(len(toks)-1, -1, -1):
        if toks[i].startswith("clip"):
            return toks[i-1] if i-1 >= 0 else "0"
    return toks[-2] if len(toks) >= 2 else "0"

def to_fake_logit(out):
    z = out["bin_logits"] if isinstance(out, dict) and "bin_logits" in out else out
    if z.dim() == 1: z = z.view(-1, 1)
    if z.size(-1) == 2:  # [real, fake]
        return z[:, 1:2] - z[:, 0:1]
    return z[:, :1]
# --- sostituisci queste due funzioni ---
def load_norm_stats(env_var: str = "NORM_STATS") -> Optional[dict]:
    sp = os.getenv(env_var, "").strip()
    if not sp or not os.path.isfile(sp):
        logging.info(f"{env_var} non impostata o file assente. Niente normalizzazione globale.")
        return None
    S = np.load(sp)
    return {
        "au_mean":  S["au_mean"].astype(np.float32),
        "au_std":   np.maximum(S["au_std"].astype(np.float32), 1e-6),
        "lmk_mean": S["lmk_mean"].astype(np.float32),
        "lmk_std":  np.maximum(S["lmk_std"].astype(np.float32), 1e-6),
    }

def normalize_features(
    au: torch.Tensor, lmk: torch.Tensor, stats: Optional[dict],
    mode: str = "clip", apply: str = "both", eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """mode: 'clip' | 'global' | 'none'; apply: 'both' | 'au' | 'lmk'."""
    mode = str(mode).lower()
    apply = str(apply).lower()
    do_au  = apply in ("both","au")
    do_lmk = apply in ("both","lmk")

    if mode == "none":
        return au, lmk

    if mode == "global":
        if stats is None:
            return au, lmk
        if do_au and au.numel() > 0:
            mu = torch.as_tensor(stats["au_mean"], device=au.device, dtype=au.dtype).unsqueeze(0)
            sd = torch.as_tensor(stats["au_std"],  device=au.device, dtype=au.dtype).unsqueeze(0)
            au = (au - mu) / sd.clamp_min(eps)
        if do_lmk and lmk.numel() > 0:
            mu = torch.as_tensor(stats["lmk_mean"], device=lmk.device, dtype=lmk.dtype).unsqueeze(0)
            sd = torch.as_tensor(stats["lmk_std"],  device=lmk.device, dtype=lmk.dtype).unsqueeze(0)
            lmk = (lmk - mu) / sd.clamp_min(eps)
        return torch.nan_to_num(au), torch.nan_to_num(lmk)

    # mode == "clip": z-score per clip sui frame validi
    if do_au and au.numel() > 0:
        mu = au.mean(dim=0, keepdim=True)
        sd = au.std(dim=0, keepdim=True).clamp_min(eps)
        au = (au - mu) / sd
    if do_lmk and lmk.numel() > 0:
        mu = lmk.mean(dim=0, keepdim=True)
        sd = lmk.std(dim=0, keepdim=True).clamp_min(eps)
        lmk = (lmk - mu) / sd
    return torch.nan_to_num(au), torch.nan_to_num(lmk)


from collections import defaultdict
def track_reduce(ld: torch.Tensor, tracks: list[str], mode="median") -> torch.Tensor:
    buckets = defaultdict(list)
    for i, tid in enumerate(tracks): buckets[tid].append(ld[i:i+1])  # [1,1]
    outs = []
    for tid in sorted(buckets, key=lambda t: (0,int(t)) if str(t).isdigit() else (1,str(t))):
        X = torch.cat(buckets[tid], 0)
        outs.append(X.median(0, keepdim=True).values if mode=="median" else X.mean(0, keepdim=True))
    return torch.cat(outs, 0)  # [Tt,1]

def agg_video(logit_track: torch.Tensor, how="max") -> torch.Tensor:
    if how=="median": return logit_track.median(0, keepdim=True).values
    if how=="mean":   return logit_track.mean(0, keepdim=True)
    return logit_track.max(0, keepdim=True).values  # default

# ---------------------------------------------------------------------
# Loader Dual
# ---------------------------------------------------------------------
def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
    
def load_model_cfg_from_json(path: Optional[str]) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        cfg = _load_json(path)
        out = {}
        if "d_model" in cfg: out["d_model"] = int(cfg["d_model"])
        if "heads" in cfg: out["nhead"] = int(cfg["heads"])
        if "layers" in cfg: out["num_layers"] = int(cfg["layers"])
        if "ff_dim" in cfg: out["dim_ff"] = int(cfg["ff_dim"])
        if "dropout" in cfg: out["dropout"] = float(cfg["dropout"])
        return out
    except Exception as e:
        print(f"[CFG] Warning: impossibile leggere config da {path}: {e}")
        return {}
    
def resolve_model_cfg(args) -> dict:
    cfg = {
        "d_model": 256,
        "nhead": 4,
        "num_layers": 4,
        "dim_ff": 768,
        "dropout": 0.10,
    }
    auto_cfg_path = args.config or (os.path.join(os.path.dirname(args.ckpt), "args.json") if os.path.isfile(os.path.join(os.path.dirname(args.ckpt), "args.json")) else None)
    cfg.update(load_model_cfg_from_json(auto_cfg_path))
    if args.d_model is not None: cfg["d_model"] = int(args.d_model)
    if args.heads   is not None: cfg["nhead"]   = int(args.heads)
    if args.layers  is not None: cfg["num_layers"] = int(args.layers)
    if args.ff_dim  is not None: cfg["dim_ff"]  = int(args.ff_dim)
    if args.dropout is not None: cfg["dropout"] = float(args.dropout)
    print("[MODEL CFG]", cfg, f" | source={auto_cfg_path or 'defaults/CLI'}")
    return cfg

def load_dual(module_path, class_name, ckpt_path, device, ft_dual=False,
              au_dim=36, lmk_dim=132, mlp_ratio_cli=None):
    # 1) Modello
    mod = __import__(module_path, fromlist=[class_name]); Cls = getattr(mod, class_name)
    ck = torch.load(ckpt_path, map_location="cpu")

    # 2) Config: args.json accanto al ckpt + override da ckpt["args"]
    ns = SimpleNamespace(ckpt=ckpt_path, config=None,
                         d_model=None, heads=None, layers=None, ff_dim=None, dropout=None)
    cfg = resolve_model_cfg(ns)  # -> d_model, nhead, num_layers, dim_ff, dropout

    args_in = ck.get("args")
    if isinstance(args_in, dict):
        # normalizza possibili chiavi alternative
        if "heads" in args_in:      cfg["nhead"]      = int(args_in["heads"])
        if "layers" in args_in:     cfg["num_layers"] = int(args_in["layers"])
        if "ff_dim" in args_in:     cfg["dim_ff"]     = int(args_in["ff_dim"])
        if "nhead" in args_in:      cfg["nhead"]      = int(args_in["nhead"])
        if "num_layers" in args_in: cfg["num_layers"] = int(args_in["num_layers"])
        if "dim_ff" in args_in:     cfg["dim_ff"]     = int(args_in["dim_ff"])
        if "d_model" in args_in:    cfg["d_model"]    = int(args_in["d_model"])
        if "dropout" in args_in:    cfg["dropout"]    = float(args_in["dropout"])

    # 3) Traduci cfg -> kwargs attesi dal costruttore
    kwargs = dict(
        au_dim=au_dim, lmk_dim=lmk_dim,
        d_model=int(cfg["d_model"]),
        heads=int(cfg["nhead"]),
        depth=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
    )
    # mlp_ratio da ff_dim/d_model o override CLI
    if mlp_ratio_cli is not None:
        kwargs["mlp_ratio"] = int(mlp_ratio_cli)
    else:
        mr = max(1, round(int(cfg["dim_ff"]) / max(1, int(cfg["d_model"]))))
        kwargs["mlp_ratio"] = int(mr)

    m = Cls(**kwargs)

    # 4) Carica pesi
    sd = ck.get("model") or ck.get("state_dict") or ck
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    if isinstance(sd, dict) and sd and all(k.startswith("module.") for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}  # strip "module."
    msd = m.state_dict()
    sd  = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:   logging.warning(f"Dual missing: {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:logging.warning(f"Dual unexpected: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")

    # 5) Modalità e grad
    m.to(device).train(bool(ft_dual))
    for p in m.parameters(): p.requires_grad = bool(ft_dual)
    return m
# ---------------------------------------------------------------------
# Dataset per-VIDEO (AU/LMK) + teacher CSV per-video
# ---------------------------------------------------------------------
class VideoDataset(Dataset):
    def __init__(self, index_json: str, split: Optional[str], csv_pervideo: str,
                 T: int = 8, track_regex: str = r"(?:track[_-]?|id[_-]?)(\d+)", zscore_mode: str = "clip", zscore_apply: str = "both"):
        self.T = int(T)
        self.track_regex = track_regex
        # z-score globale opzionale (compat con Dual originale)
        self._stats = load_norm_stats()  # serve solo per 'global'
        self._zmode = zscore_mode
        self._zapply = zscore_apply

        stem2prob, stem2label = {}, {}

        with open(csv_pervideo, newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                vp   = r["video_path"]
                prob = float(r["video_score"])
                stem = os.path.splitext(os.path.basename(vp))[0]      # es. 439  o 568_628
                tech = os.path.basename(os.path.dirname(vp))          # es. original, FaceSwap...
                mvid = f"{tech}/{stem}"                               # es. original/439

                # usa SOLO chiavi per-video
                for k in (stem, mvid):
                    stem2prob[k] = prob

                gl = r.get("gt_label", "")
                if str(gl).strip() != "":
                    lbl = int(gl)
                    for k in (stem, mvid):
                        stem2label[k] = lbl



        j = json.load(open(index_json))
        items = j.get(split, []) if isinstance(j, dict) and split else j

        groups: Dict[str, Dict[str, Any]] = {}
        skipped, added = 0, 0
        for e in items:
            if isinstance(e, str):
                p, y = e, None
            else:
                p = e.get("path") or e.get("clip_dir") or e.get("dir") or e.get("p")
                y = e.get("y", e.get("label"))
            if not p: skipped += 1; continue

            tokens = re.split(r"[\\/]", p)
            cands = []
            if len(tokens) >= 4:
                cands.append(f"{tokens[-4]}/{tokens[-3]}")  # <manip>/<video>
            cands.append(tokens[-3])                         # <video>
            # ultimo fallback: token che matcha \d+_\d+
            cands.extend([t for t in tokens if re.fullmatch(r"\d+_\d+", t)])

            stem_match = next((t for t in cands if t in stem2prob), None)
            if stem_match is None:
                skipped += 1
                continue

            y_csv = stem2label.get(stem_match)
            if y is None:
                if y_csv is not None: y = y_csv
                else:
                    pl = p.lower()
                    toks_fake = ["fake","deepfake","deepfakes","face2face","faceswap","neuraltextures","celeb-synthesis"]
                    y = 1 if any(t in pl for t in toks_fake) else 0
            else:
                if y_csv is not None: y = y_csv



            g = groups.setdefault(stem_match, {
                "clips": [], "y": int(y), "prob_csv": float(stem2prob[stem_match])
            })
            g["clips"].append(p); added += 1

        self.videos = [v for v in groups.values() if len(v["clips"]) > 0]

        n_videos = len(self.videos)
        # dopo self.videos
        y_list = [v["y"] for v in self.videos]
        logging.info(f"Dataset '{split}': videos={len(self.videos)} pos={sum(y_list)} neg={len(y_list)-sum(y_list)}")

        match_videos = sum(1 for v in groups.values() if len(v["clips"]) > 0)


        logging.info("match_rate=%d/%d videos, skipped=%d",
                    match_videos, len(groups), skipped)
        logging.info(f"Dataset '{split}': videos={n_videos} clips_total={added} skipped={skipped} "
                     f"positives={sum(y_list)} negatives={n_videos - sum(y_list)}")
        by_tech = Counter(
            (lambda parts: parts[-4] if len(parts) >= 4 else "UNK")(re.split(r"[\\/]", c))
            for v in self.videos for c in v["clips"]
        )

        n_orig = sum(1 for v in self.videos if any("/original/" in c or "\\original\\" in c for c in v["clips"]))
        logging.info(f"videos_by_tech={dict(by_tech)} | videos_original={n_orig}/{len(self.videos)}")

    def __len__(self) -> int: return len(self.videos)

    @staticmethod
    def _fixT(x: torch.Tensor, T: int) -> torch.Tensor:
        if x.shape[0] == T: return x
        if x.shape[0] > T:
            s = (x.shape[0] - T) // 2; return x[s:s + T]
        pad = T - x.shape[0]; return torch.cat([x, x[-1:].repeat(pad, 1)], dim=0)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        v = self.videos[i]
        y     = torch.tensor([[v["y"]]], dtype=torch.float32)     # [1,1]
        p_csv = torch.tensor([[v["prob_csv"]]], dtype=torch.float32)  # [1,1] prob teacher

        aus, lmks, tmasks, tracks = [], [], [], []
        for d in v["clips"]:
            try:
                au  = torch.from_numpy(np.asarray(np.load(os.path.join(d, "au_features.npy")),  np.float32))
                lmk = torch.from_numpy(np.asarray(np.load(os.path.join(d, "lmk_features.npy")), np.float32))
            except Exception as e:
                logging.warning(f"Clip skip (missing AU/LMK): {d} err={e}")
                continue
            au  = self._fixT(au,  self.T)
            lmk = self._fixT(lmk, self.T)
            au, lmk = normalize_features(au, lmk, self._stats, mode=self._zmode, apply=self._zapply)


            aus.append(au); lmks.append(lmk); tmasks.append(torch.ones(self.T, dtype=torch.bool))
            tracks.append(get_person_id_from_path(d))

        if len(aus) == 0:
            # segnaposto, raro
            au  = torch.zeros(self.T, 36, dtype=torch.float32)
            lmk = torch.zeros(self.T,132, dtype=torch.float32)
            return {"au":au.unsqueeze(0), "lmk":lmk.unsqueeze(0),
                    "tmask":torch.ones(1,self.T,dtype=torch.bool),
                    "p_csv":p_csv, "y":y, "tracks":["0"]}

        au  = torch.stack(aus,  dim=0)  # [N,T,36]
        lmk = torch.stack(lmks, dim=0)  # [N,T,132]
        tmk = torch.stack(tmasks, dim=0)# [N,T]
        return {"au": au, "lmk": lmk, "tmask": tmk, "p_csv": p_csv, "y": y, "tracks": tracks}

# ---------------------------------------------------------------------
# Val SOLO Dual
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_dual(dual: nn.Module, loader: DataLoader, device: torch.device,
                  track_reduce_mode: str = "median", video_agg: str = "max") -> Dict[str, float]:
    dual.eval()
    ys, ps = [], []
    n_vid, n_clip = 0, 0
    for batch in tqdm(loader, desc="ValDual", leave=False):
        for v in batch:
            au, lmk = v["au"].to(device), v["lmk"].to(device)
            y       = v["y"].to(device)     # [1,1]
            
            N = au.size(0)
            out = dual(au.view(N, au.size(1), au.size(2)),
                       lmk.view(N, lmk.size(1), lmk.size(2)),
                       lengths=None)
            ld = to_fake_logit(out)  
            logit_track = track_reduce(ld, v["tracks"], mode=track_reduce_mode)   # [Tt,1]
            logit_video = agg_video(logit_track, video_agg)                       # [1,1]
            ys.append(y.cpu().numpy()); ps.append(torch.sigmoid(logit_video).cpu().numpy())
            n_vid += 1; n_clip += int(N)

    # --- sostituisci il blocco finale in evaluate_dual ---
    y_true = np.concatenate(ys, 0).astype(np.float32).ravel().astype(np.int32)
    y_prob = np.concatenate(ps, 0).astype(np.float32).ravel()
    if len(np.unique(y_true)) < 2:
        auc = float("nan"); ap = float("nan")
    else:
        auc = roc_auc_score(y_true, y_prob)
        ap  = average_precision_score(y_true, y_prob)


    logging.info(f"[EVAL Dual] videos={n_vid} clips={n_clip} AUC={auc:.4f} PR-AUC={ap:.4f}")
    return {"auc": float(auc), "pr_auc": float(ap), "videos": n_vid, "clips": n_clip}


# ---------------------------------------------------------------------
# Train: FT SOLO Dual con teacher per-VIDEO
# ---------------------------------------------------------------------
def train_dual_video(index: str, split_train: str, split_val: str, csv_pervideo: str,
                     dual_module: str, dual_class: str, dual_ckpt: str,
                     out_dir: str, epochs: int = 5, lr: float = 4e-4, wd: float = 1e-4,
                     T: int = 8, batch_videos: int = 1, workers: int = 4,
                     alpha_distill: float = 0.5,
                     track_reduce_mode: str = "median", video_agg: str = "max",
                     device: Optional[str] = None, dual_mlp_ratio: Optional[int]=None,  zscore_mode: str = "clip", zscore_apply: str = "both"):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Device: {device}")
    logging.info(f"Config: T={T} epochs={epochs} batch_videos={batch_videos} lr={lr} wd={wd} "
                 f"α={alpha_distill} agg=({track_reduce_mode},{video_agg})")

    ds_tr = VideoDataset(index, split_train, csv_pervideo, T=T,
                     zscore_mode=zscore_mode, zscore_apply=zscore_apply)
    ds_va = VideoDataset(index, split_val,   csv_pervideo, T=T,
                        zscore_mode=zscore_mode, zscore_apply=zscore_apply)
    if len(ds_tr)==0 or len(ds_va)==0:
        logging.error("Dataset vuoto."); return

    tr = DataLoader(ds_tr, batch_size=batch_videos, shuffle=True,  num_workers=workers,
                    pin_memory=True, collate_fn=lambda x:x)
    va = DataLoader(ds_va, batch_size=1,            shuffle=False, num_workers=workers,
                    pin_memory=True, collate_fn=lambda x:x)

    dual = load_dual(dual_module, dual_class, dual_ckpt, device, ft_dual=True,
                     au_dim=36, lmk_dim=132, mlp_ratio_cli=dual_mlp_ratio)
    # ottimizzatore
    opt = torch.optim.AdamW(dual.parameters(), lr=lr, weight_decay=wd)
    bce_logits = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_dual_video.pt")

    for ep in range(1, epochs+1):
        dual.train()
        t0 = time.time()
        loss_sum=sup_sum=dis_sum=0.0
        n_steps=n_vids=n_clips=0

        pbar = tqdm(tr, desc=f"TrainDual[{ep:03d}/{epochs:03d}]", leave=False)
        for batch in pbar:
            opt.zero_grad()

            losses = []
            batch_sup = 0.0
            batch_dis = 0.0
            vids_in_batch = 0
            clips_in_batch = 0

            for v in batch:
                au, lmk = v["au"].to(device), v["lmk"].to(device)
                y       = v["y"].to(device)       # [1,1]
                p_csv   = v["p_csv"].to(device).clamp_(1e-4, 1-1e-4)   # [1,1] prob teacher per-video

                N = au.size(0)
                out = dual(au.view(N, au.size(1), au.size(2)),
                        lmk.view(N, lmk.size(1), lmk.size(2)),
                        lengths=None)
                ld = to_fake_logit(out) 
                # clip -> track(mediana) -> video(max)
                logit_track = track_reduce(ld, v["tracks"], mode=track_reduce_mode)  # [Tt,1]
                logit_video = agg_video(logit_track, video_agg)                      # [1,1]

                # supervised + distillazione (su probabilità)
                l_sup = bce_logits(logit_video, y)
                l_dis = nn.functional.binary_cross_entropy(torch.sigmoid(logit_video), p_csv)

                losses.append(l_sup + alpha_distill * l_dis)
                batch_sup += float(l_sup.item())
                batch_dis += float(l_dis.item())
                vids_in_batch += 1
                clips_in_batch += int(N)

            # media delle loss tra i video del batch
            tot_loss = torch.stack(losses).mean()
            tot_loss.backward()

            # diagnostica gradiente medio
            with torch.no_grad():
                gsum, nP = 0.0, 0
                for p in dual.parameters():
                    if p.grad is not None:
                        gsum += float(p.grad.norm().item()); nP += 1
                b_grad = gsum / max(1, nP)

            torch.nn.utils.clip_grad_norm_(dual.parameters(), max_norm=5.0)
            opt.step()

            # logging
            loss_sum += float(tot_loss.item())
            sup_sum  += batch_sup / max(1, vids_in_batch)
            dis_sum  += batch_dis / max(1, vids_in_batch)
            n_steps  += 1
            n_vids   += vids_in_batch
            n_clips  += clips_in_batch

            pbar.set_postfix(loss=f"{tot_loss.item():.4f}",
                            sup=f"{(batch_sup/max(1,vids_in_batch)):.4f}",
                            dist=f"{(batch_dis/max(1,vids_in_batch)):.4f}",
                            vids=n_vids, clips=n_clips)

        # valutazione SOLO Dual
        val = evaluate_dual(dual, va, device,
                            track_reduce_mode=track_reduce_mode, video_agg=video_agg)
        dt = time.time()-t0
        logging.info(f"[{ep:03d}] loss={loss_sum/max(n_steps,1):.4f} "
                     f"(sup={sup_sum/max(n_steps,1):.4f}, dist={dis_sum/max(n_steps,1):.4f}) | "
                     f"valAUC={val['auc']:.4f} PR={val['pr_auc']:.4f} | "
                     f"vids={val['videos']} clips={val['clips']} | {dt:.1f}s")

        if val["auc"] > best_auc:
            best_auc = val["auc"]
            args = {"d_model":256, "nhead":4, "num_layers":4, "dim_ff":768, "dropout":0.10}
            torch.save({"model": dual.state_dict(), "val": val, "args": args}, best_path)

            # scrivi anche il sidecar per sicurezza
            with open(os.path.join(out_dir, "args.json"), "w") as f:
                json.dump(args, f)
            logging.info(f"Checkpoint salvato: {best_path} (AUC={best_auc:.4f})")

    logging.info(f"FT finito. Best Dual AUC={best_auc:.4f}. Path: {best_path}")

# ---------------------------------------------------------------------
# Inferenza SOLO Dual
# ---------------------------------------------------------------------
@torch.no_grad()
def infer_dual(video_dir: str, dual_module: str, dual_class: str, dual_ckpt: str,
               T_dual: int = 8, track_regex: str = r"(?:track[_-]?|id[_-]?)(\d+)",
               video_agg: str = "max", device: Optional[str]=None, zscore_mode: str = "clip", zscore_apply: str = "both") -> Dict[str, Any]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dual = load_dual(dual_module, dual_class, dual_ckpt, device, ft_dual=False)

    _stats = load_norm_stats()

    clip_dirs = [d for d in sorted(glob.glob(os.path.join(video_dir, "clip*")))
                 if os.path.isfile(os.path.join(d, "au_features.npy"))]
    if not clip_dirs:
        raise RuntimeError("Servono subdir clip* con au_features.npy e lmk_features.npy")
    def fixT(x: torch.Tensor, T: int) -> torch.Tensor:
        if x.shape[0]==T: return x
        if x.shape[0]>T:
            s=(x.shape[0]-T)//2; return x[s:s+T]
        pad=T-x.shape[0]; return torch.cat([x, x[-1:].repeat(pad,1)],0)

    ld_list, tracks = [], []
    for d in tqdm(clip_dirs, desc="InferDualClips", leave=False):
        au  = torch.from_numpy(np.asarray(np.load(os.path.join(d, "au_features.npy")),  np.float32))
        lmk = torch.from_numpy(np.asarray(np.load(os.path.join(d, "lmk_features.npy")), np.float32))
        au, lmk = fixT(au, T_dual), fixT(lmk, T_dual)
        au, lmk = normalize_features(au, lmk, _stats, mode=zscore_mode, apply=zscore_apply)

        out = dual(au.unsqueeze(0).to(device), lmk.unsqueeze(0).to(device), lengths=None)
        ld = to_fake_logit(out)
        ld_list.append(ld); tracks.append(get_person_id_from_path(d))

    ld_clip  = torch.cat(ld_list, 0)                 # [N,1]
    log_tr   = track_reduce(ld_clip, tracks, mode="median")
    log_vid  = agg_video(log_tr, video_agg)
    p_tr     = torch.sigmoid(log_tr).squeeze(1).cpu().numpy().tolist()
    p_vid    = float(torch.sigmoid(log_vid).item())

    return {"prob_video_dual": p_vid, "tracks": tracks,
            "prob_dual_per_track": p_tr, "n_clips": int(ld_clip.size(0)),
            "agg": {"track":"median","video":video_agg}}

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser("FT SOLO Dual per-video (median track, max video)")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--log-file",  default=None)

    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--index", required=True)
    tr.add_argument("--csv-pervideo", required=True)
    tr.add_argument("--split-train", default="train")
    tr.add_argument("--split-val",   default="val")
    tr.add_argument("--dual-module", required=True)
    tr.add_argument("--dual-class",  required=True)
    tr.add_argument("--dual-ckpt",   required=True)
    tr.add_argument("--out",         required=True)
    tr.add_argument("--epochs", type=int, default=5)
    tr.add_argument("--lr",     type=float, default=4e-4)
    tr.add_argument("--wd",     type=float, default=1e-4)
    tr.add_argument("--T",      type=int,   default=8)
    tr.add_argument("--batch-videos", type=int, default=1)
    tr.add_argument("--workers",      type=int, default=4)
    tr.add_argument("--alpha-distill", type=float, default=0.5)
    tr.add_argument("--track-reduce", choices=["median","mean"], default="median")
    tr.add_argument("--video-agg",    choices=["max","median","mean"], default="max")
    tr.add_argument("--dual-mlp-ratio", type=int, default=None)
    tr.add_argument("--device", default=None)
    tr.add_argument("--zscore-mode", choices=["clip","global","none"], default="clip")
    tr.add_argument("--zscore-apply", choices=["both","au","lmk"],     default="both")


    inf = sub.add_parser("infer")
    inf.add_argument("--zscore-mode", choices=["clip","global","none"], default="clip")
    inf.add_argument("--zscore-apply", choices=["both","au","lmk"],     default="both")
    inf.add_argument("--video-dir", required=True)
    inf.add_argument("--dual-module", required=True)
    inf.add_argument("--dual-class",  required=True)
    inf.add_argument("--dual-ckpt",   required=True)
    inf.add_argument("--T-dual",    type=int, default=8)
    inf.add_argument("--track-regex", default=r"(?:track[_-]?|id[_-]?)(\d+)")
    inf.add_argument("--video-agg",    choices=["max","median","mean"], default="max")
    inf.add_argument("--device", default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    setup_logger(args.log_level, args.log_file)
    if args.cmd=="train":
        train_dual_video(
            index=args.index, split_train=args.split_train, split_val=args.split_val,
            csv_pervideo=args.csv_pervideo,
            dual_module=args.dual_module, dual_class=args.dual_class, dual_ckpt=args.dual_ckpt,
            out_dir=args.out, epochs=args.epochs, lr=args.lr, wd=args.wd, T=args.T,
            batch_videos=args.batch_videos, workers=args.workers,
            alpha_distill=args.alpha_distill,
            track_reduce_mode=args.track_reduce, video_agg=args.video_agg,
            zscore_mode=args.zscore_mode, zscore_apply=args.zscore_apply,
            device=args.device, dual_mlp_ratio=args.dual_mlp_ratio
        )

    else:
        out = infer_dual(
            video_dir=args.video_dir,
            dual_module=args.dual_module, dual_class=args.dual_class, dual_ckpt=args.dual_ckpt, track_regex=args.track_regex, video_agg=args.video_agg,
            device=args.device, T_dual=args.T_dual, zscore_mode=args.zscore_mode, zscore_apply=args.zscore_apply
        )
        print(out)

if __name__=="__main__":
    main()
