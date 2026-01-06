#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mMLP: fusione per-VIDEO di AltFreezing (p_rgb per-video da CSV) + Dual (AU+LMK).
clip->track = MEDIAN, track->video = MAX.
Loss = BCE(z_fuse, y) + gamma*BCE(g, g_tgt)  [Dual è frozen; lambda_dual solo CLI-compat]

Richiede:
- index.json dei clip AU/LMK (come nel tuo codice).
- csv_pervideo con colonne: video_path, video_score[, gt_label].
- checkpoint e modulo del modello Dual (solo AU+LMK).

Normalizzazione:
- --zscore-mode {clip,global,none}  [default: clip]
- --zscore-apply {both,au,lmk}      [default: both]
- Per "global" serve NORM_STATS con au_mean, au_std, lmk_mean, lmk_std.
"""
import os, re, csv, json, argparse, math, logging, sys, time
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm


# ---------------- Logger ----------------
def setup_logger(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# --------------- Utils ------------------
def sigmoid_to_logit(p: float, eps: float = 1e-6) -> float:
    p = max(eps, min(1 - eps, float(p)))
    return float(math.log(p / (1 - p)))


def to_fake_logit(out: Any) -> torch.Tensor:
    z = out["bin_logits"] if isinstance(out, dict) and "bin_logits" in out else out
    if z.dim() == 1:
        z = z.view(-1, 1)
    if z.size(-1) == 2:  # [real, fake]
        return z[:, 1:2] - z[:, 0:1]
    return z[:, :1]


def get_person_id_from_path(p: str) -> str:
    toks = re.split(r"[\\/]", p)
    for i in range(len(toks) - 1, -1, -1):
        if toks[i].startswith("clip"):
            return toks[i - 1] if i - 1 >= 0 else "0"
    return toks[-2] if len(toks) >= 2 else "0"


def zscore_clip_np(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float32, copy=False)
    mu = X.mean(axis=0, keepdims=True)
    sd = np.maximum(X.std(axis=0, keepdims=True), eps)
    return ((X - mu) / sd).astype(np.float32, copy=False)


def load_norm_stats(env_var: str = "NORM_STATS", override: Optional[dict] = None) -> Optional[dict]:
    if override is not None:
        return {
            "au_mean":  np.asarray(override["au_mean"],  np.float32),
            "au_std":   np.maximum(np.asarray(override["au_std"],  np.float32),  1e-6),
            "lmk_mean": np.asarray(override["lmk_mean"], np.float32),
            "lmk_std":  np.maximum(np.asarray(override["lmk_std"], np.float32), 1e-6),
        }
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


# -------------- Aggregazioni ------------
def track_reduce(ld: torch.Tensor, tracks: List[str], mode: str = "median") -> torch.Tensor:
    buckets: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for i, tid in enumerate(tracks):
        buckets[tid].append(ld[i:i + 1])
    outs: List[torch.Tensor] = []
    for tid in sorted(buckets, key=lambda t: (0, int(t)) if str(t).isdigit() else (1, str(t))):
        X = torch.cat(buckets[tid], 0)
        outs.append(X.median(0, keepdim=True).values if mode == "median" else X.mean(0, keepdim=True))
    return torch.cat(outs, 0)  # [Tt,1]


def agg_video(logit_track: torch.Tensor, how: str = "max") -> torch.Tensor:
    if how == "median":
        return logit_track.median(0, keepdim=True).values
    if how == "mean":
        return logit_track.mean(0, keepdim=True)
    return logit_track.max(0, keepdim=True).values


# --------------- Dual Loader ------------
def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_model_cfg_from_json(path: Optional[str]) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        cfg = _load_json(path)
        out = {}
        if "d_model" in cfg: out["d_model"] = int(cfg["d_model"])
        if "heads"   in cfg: out["nhead"]   = int(cfg["heads"])
        if "layers"  in cfg: out["depth"]   = int(cfg["layers"])
        if "ff_dim"  in cfg: out["dim_ff"]  = int(cfg["ff_dim"])
        if "dropout" in cfg: out["dropout"] = float(cfg["dropout"])
        return out
    except Exception as e:
        logging.warning(f"args.json parse fail: {e}")
        return {}


def _resolve_model_cfg(ckpt_path: str) -> dict:
    cfg = {"d_model": 256, "nhead": 4, "depth": 4, "dim_ff": 768, "dropout": 0.10}
    sidecar = os.path.join(os.path.dirname(ckpt_path), "args.json")
    cfg.update(_load_model_cfg_from_json(sidecar))
    return cfg


def load_dual(module_path: str, class_name: str, ckpt_path: str, device: torch.device,
              au_dim: int = 36, lmk_dim: int = 132) -> nn.Module:
    mod = __import__(module_path, fromlist=[class_name])
    Cls = getattr(mod, class_name)
    ck = torch.load(ckpt_path, map_location="cpu")

    cfg = _resolve_model_cfg(ckpt_path)
    args_in = ck.get("args", {})
    # alias comuni
    if "heads" in args_in:  cfg["nhead"] = int(args_in["heads"])
    if "layers" in args_in: cfg["depth"] = int(args_in["layers"])
    if "ff_dim" in args_in: cfg["dim_ff"] = int(args_in["ff_dim"])
    for k in ("nhead", "depth", "dim_ff", "d_model", "dropout"):
        if k in args_in:
            cfg[k] = int(args_in[k]) if k != "dropout" else float(args_in[k])

    kwargs = dict(
        au_dim=au_dim, lmk_dim=lmk_dim,
        d_model=int(cfg["d_model"]),
        heads=int(cfg["nhead"]),
        depth=int(cfg["depth"]),
        dropout=float(cfg["dropout"]),
        mlp_ratio=max(1, round(int(cfg["dim_ff"]) / max(1, int(cfg["d_model"]))))
    )

    m = Cls(**kwargs)

    sd = ck.get("model") or ck.get("state_dict") or ck
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and sd and all(k.startswith("module.") for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}

    msd = m.state_dict()
    sd_ok = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    miss = [k for k in msd.keys() if k not in sd_ok]
    unex = [k for k in sd.keys() if k not in msd]
    frac = len(sd_ok) / max(1, len(msd))
    logging.info(f"Dual load: matched={len(sd_ok)}/{len(msd)} ({frac*100:.1f}%), missing={len(miss)}, unexpected={len(unex)}")
    if frac < 0.95:
        logging.warning("ATTENZIONE: meno del 95% dei pesi combacia. Architettura non allineata?")

    m.load_state_dict(sd_ok, strict=False)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


@torch.no_grad()
def dual_video_logit(dual: nn.Module, au: torch.Tensor, lmk: torch.Tensor, tracks: List[str],
                     device: torch.device) -> torch.Tensor:
    # au,lmk: [N,T,D]
    N = au.size(0)
    out = dual(
        au.view(N, au.size(1), au.size(2)).to(device),
        lmk.view(N, lmk.size(1), lmk.size(2)).to(device),
        lengths=None
    )
    ld_clip = to_fake_logit(out)                    # [N,1]
    log_tr  = track_reduce(ld_clip, tracks, "median")
    log_vid = agg_video(log_tr, "max")              # [1,1]
    return log_vid


@torch.no_grad()
def eval_dual_only(loader: DataLoader, dual: nn.Module, device: torch.device) -> None:
    ys, ps = [], []
    for batch in loader:
        for v in batch:
            z_dual = dual_video_logit(dual, v["au"], v["lmk"], v["tracks"], device)
            ys.append(int(v["y"].item()))
            ps.append(torch.sigmoid(z_dual).item())
    y = np.array(ys, np.int32)
    p = np.array(ps, np.float32)
    if len(np.unique(y)) >= 2:
        auc = roc_auc_score(y, p)
        ap  = average_precision_score(y, p)
        logging.info(f"[DUAL-ONLY] AUC={auc:.4f} PR-AUC={ap:.4f}")
    else:
        logging.info("[DUAL-ONLY] AUC=nan PR-AUC=nan (val set mono-classe)")


# ----------------- Dataset --------------
class VideoDatasetMMLP(Dataset):
    """
    Per video:
      - lista di clip -> (AU, LMK) T-fix con z-score per-clip|globale|none
      - tracks
      - y
      - p_rgb per-video dal CSV
      - chiave video
    """
    def __init__(self, index_json: str, split: Optional[str], csv_pervideo: str,
                 T: int = 8, zscore_mode: str = "clip", zscore_apply: str = "both"):
        self.T = int(T)
        self.zmode = zscore_mode.lower()
        self.zapply = zscore_apply.lower()
        self.do_au  = self.zapply in ("both", "au")
        self.do_lmk = self.zapply in ("both", "lmk")
        self.stats = load_norm_stats() if self.zmode == "global" else None

        stem2prob: Dict[str, float] = {}
        stem2label: Dict[str, int] = {}
        with open(csv_pervideo, newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                vp = r["video_path"]
                prob = float(r["video_score"])
                stem = os.path.splitext(os.path.basename(vp))[0]
                tech = os.path.basename(os.path.dirname(vp))
                mvid = f"{tech}/{stem}"
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
            if not p:
                skipped += 1
                continue

            tokens = re.split(r"[\\/]", p)
            cands = []
            if len(tokens) >= 4:
                cands.append(f"{tokens[-4]}/{tokens[-3]}")
            cands.append(tokens[-3])
            cands.extend([t for t in tokens if re.fullmatch(r"\d+_\d+", t)])

            stem_match = next((t for t in cands if t in stem2prob), None)
            if stem_match is None:
                skipped += 1
                continue

            y_csv = stem2label.get(stem_match)
            if y is None:
                if y_csv is not None:
                    y = y_csv
                else:
                    pl = p.lower()
                    toks_fake = ["fake", "deepfake", "deepfakes", "face2face", "faceswap", "neuraltextures", "celeb-synthesis"]
                    y = 1 if any(t in pl for t in toks_fake) else 0
            else:
                if y_csv is not None:
                    y = y_csv

            g = groups.setdefault(stem_match, {"clips": [], "y": int(y), "p_rgb": float(stem2prob[stem_match]), "key": stem_match})
            g["clips"].append(p)
            added += 1

        self.videos = [v for v in groups.values() if len(v["clips"]) > 0]
        logging.info(f"Dataset '{split}': videos={len(self.videos)} clips_total={added} skipped={skipped} "
                     f"pos={sum(v['y'] for v in self.videos)} neg={len(self.videos)-sum(v['y'] for v in self.videos)}")

    def __len__(self) -> int:
        return len(self.videos)

    @staticmethod
    def _fixT(x: torch.Tensor, T: int) -> torch.Tensor:
        if x.shape[0] == T:
            return x
        if x.shape[0] > T:
            s = (x.shape[0] - T) // 2
            return x[s:s + T]
        pad = T - x.shape[0]
        return torch.cat([x, x[-1:].repeat(pad, 1)], dim=0)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        v = self.videos[i]
        y = torch.tensor([[v["y"]]], dtype=torch.float32)
        p_rgb = torch.tensor([[v["p_rgb"]]], dtype=torch.float32)

        aus, lmks, tracks = [], [], []
        for d in v["clips"]:
            try:
                au_np  = np.asarray(np.load(os.path.join(d, "au_features.npy")),  np.float32)
                lmk_np = np.asarray(np.load(os.path.join(d, "lmk_features.npy")), np.float32)
            except Exception as e:
                logging.warning(f"Clip skip (missing AU/LMK): {d} err={e}")
                continue

            # z-score per-clip | globale | none
            if self.zmode == "clip":
                if self.do_au:  au_np  = zscore_clip_np(au_np)
                if self.do_lmk: lmk_np = zscore_clip_np(lmk_np)
            elif self.zmode == "global" and self.stats is not None:
                if self.do_au:
                    au_np = (au_np - self.stats["au_mean"][None, :]) / self.stats["au_std"][None, :]
                if self.do_lmk:
                    lmk_np = (lmk_np - self.stats["lmk_mean"][None, :]) / self.stats["lmk_std"][None, :]
                au_np  = np.nan_to_num(au_np,  nan=0.0, posinf=0.0, neginf=0.0)
                lmk_np = np.nan_to_num(lmk_np, nan=0.0, posinf=0.0, neginf=0.0)

            au_t  = torch.from_numpy(au_np)
            lmk_t = torch.from_numpy(lmk_np)
            au_t  = self._fixT(au_t,  self.T)
            lmk_t = self._fixT(lmk_t, self.T)

            aus.append(au_t)
            lmks.append(lmk_t)
            tracks.append(get_person_id_from_path(d))

        if len(aus) == 0:
            au_t  = torch.zeros(self.T, 36,  dtype=torch.float32)
            lmk_t = torch.zeros(self.T, 132, dtype=torch.float32)
            aus = [au_t]; lmks = [lmk_t]; tracks = ["0"]

        au  = torch.stack(aus,  dim=0)  # [N,T,36]
        lmk = torch.stack(lmks, dim=0)  # [N,T,132]
        return {"au": au, "lmk": lmk, "tracks": tracks, "y": y, "p_rgb": p_rgb, "key": v["key"]}


# --------------- mMLP Model -------------
class GatedMoE(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.t_rgb  = nn.Parameter(torch.tensor(1.0))
        self.t_dual = nn.Parameter(torch.tensor(1.0))
        self.gate = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z_rgb: torch.Tensor, z_dual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_rgb, z_dual, torch.abs(z_rgb - z_dual)], dim=1)  # [B,3]
        g = torch.sigmoid(self.gate(x))                                   # [B,1]
        z_rgb_t  = z_rgb  / self.t_rgb.clamp_min(1.0)
        z_dual_t = z_dual / self.t_dual.clamp_min(0.1)
        p = g * torch.sigmoid(z_rgb_t) + (1 - g) * torch.sigmoid(z_dual_t)
        eps = 1e-6
        z = torch.log((p + eps) / (1 - p + eps))
        return z, g


# --------------- Valutazione ------------
@torch.no_grad()
def evaluate_epoch(dual: nn.Module, mmlp: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    mmlp.eval()
    ys, pfuse, pdual, prgb = [], [], [], []
    g_means: List[float] = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        for v in batch:
            z_dual_v = dual_video_logit(dual, v["au"], v["lmk"], v["tracks"], device)
            z_rgb_v  = torch.tensor([[sigmoid_to_logit(float(v["p_rgb"].item()))]],
                                    dtype=torch.float32, device=device)
            z_fuse_v, g = mmlp(z_rgb_v, z_dual_v.to(device))

            ys.append(int(v["y"].item()))
            pfuse.append(torch.sigmoid(z_fuse_v).item())
            pdual.append(torch.sigmoid(z_dual_v).item())
            prgb.append(float(v["p_rgb"].item()))
            g_means.append(g.detach().cpu().mean().item())

    y_true = np.array(ys, dtype=np.int32)

    def _metrics(p: np.ndarray) -> Tuple[float, float]:
        if len(np.unique(y_true)) < 2:
            return float("nan"), float("nan")
        return float(roc_auc_score(y_true, p)), float(average_precision_score(y_true, p))

    auc_f, pr_f   = _metrics(np.array(pfuse, dtype=np.float32))
    auc_du, pr_du = _metrics(np.array(pdual, dtype=np.float32))
    auc_rg, pr_rg = _metrics(np.array(prgb,  dtype=np.float32))
    g_mean = float(np.mean(g_means)) if g_means else float("nan")

    logging.info(f"[VAL] fuse AUC={auc_f:.4f} PR={pr_f:.4f} | dual AUC={auc_du:.4f} PR={pr_du:.4f} | rgb AUC={auc_rg:.4f} PR={pr_rg:.4f} | gate_mean={g_mean:.3f}")
    return {"auc": auc_f, "pr_auc": pr_f, "auc_dual": auc_du, "auc_rgb": auc_rg}


# --------------- Train ------------------
def train_mmlp(index: str, csv_pervideo: str, split_train: str, split_val: str,
               dual_module: str, dual_class: str, dual_ckpt: str,
               out_dir: str, epochs: int = 5, lr: float = 1e-3, wd: float = 0.0,
               T: int = 8, batch_videos: int = 1, workers: int = 4,
               lambda_dual: float = 0.2,  # compatibilità CLI, non usato nella loss
               device: Optional[str] = None,
               zscore_mode: str = "clip", zscore_apply: str = "both") -> None:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Device: {device_t}")

    ds_tr = VideoDatasetMMLP(index, split_train, csv_pervideo, T=T,
                             zscore_mode=zscore_mode, zscore_apply=zscore_apply)
    ds_va = VideoDatasetMMLP(index, split_val,   csv_pervideo, T=T,
                             zscore_mode=zscore_mode, zscore_apply=zscore_apply)
    if len(ds_tr) == 0 or len(ds_va) == 0:
        logging.error("Dataset vuoto.")
        return

    tr = DataLoader(ds_tr, batch_size=batch_videos, shuffle=True,  num_workers=workers,
                    pin_memory=True, collate_fn=lambda x: x)
    va = DataLoader(ds_va, batch_size=1,            shuffle=False, num_workers=workers,
                    pin_memory=True, collate_fn=lambda x: x)

    dual = load_dual(dual_module, dual_class, dual_ckpt, device_t)
    eval_dual_only(va, dual, device_t)

    mmlp = GatedMoE().to(device_t)
    opt  = torch.optim.AdamW(mmlp.parameters(), lr=lr, weight_decay=wd)

    # iperparametri interni (stessi valori funzionali del codice precedente)
    beta  = 5.0   # peso disaccordo |p_rgb - p_dual|
    k     = 2.0   # ripidità target del gate
    gamma = 0.1   # peso regolarizzazione del gate

    best_auc = -1.0
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_mmlp.pt")

    for ep in range(1, epochs + 1):
        mmlp.train()
        t0 = time.time()
        loss_sum = sup_sum = gate_sum = dual_bce_sum = 0.0
        n_steps = n_vids = 0

        pbar = tqdm(tr, desc=f"Train mMLP [{ep:03d}/{epochs:03d}]", leave=False)
        for batch in pbar:
            opt.zero_grad()
            per_item_losses: List[torch.Tensor] = []
            gate_batch_mean = 0.0

            for v in batch:
                # logits per-video
                z_dual_v = dual_video_logit(dual, v["au"], v["lmk"], v["tracks"], device_t).to(device_t)  # [1,1]
                z_rgb_v  = torch.tensor([[sigmoid_to_logit(float(v["p_rgb"].item()))]],
                                        dtype=torch.float32, device=device_t)                           # [1,1]
                z_fuse_v, g = mmlp(z_rgb_v, z_dual_v)  # g∈(0,1)
                y = v["y"].to(device_t)                # [1,1]

                # peso sui disaccordi
                p_rgb  = torch.sigmoid(z_rgb_v)
                p_dual = torch.sigmoid(z_dual_v)
                w = 1.0 + beta * torch.abs(p_rgb - p_dual)

                # supervised pesata
                L_sup = (nn.functional.binary_cross_entropy_with_logits(z_fuse_v, y, reduction="none") * w).mean()

                # regolarizzazione del gate verso il modello più "sicuro"
                margin = torch.abs(z_dual_v) - torch.abs(z_rgb_v)  # >0 => preferisci Dual
                g_tgt  = torch.sigmoid(k * margin)
                L_gate = nn.functional.binary_cross_entropy(g, g_tgt)

                # solo logging per Dual
                L_dual_log = nn.functional.binary_cross_entropy_with_logits(z_dual_v, y)

                per_item_losses.append(L_sup + gamma * L_gate)

                sup_sum      += float(L_sup.item())
                gate_sum     += float(L_gate.item())
                dual_bce_sum += float(L_dual_log.item())
                n_vids       += 1
                gate_batch_mean += float(g.detach().mean().item())

            tot_loss = torch.stack(per_item_losses).mean()
            tot_loss.backward()
            nn.utils.clip_grad_norm_(mmlp.parameters(), max_norm=5.0)
            opt.step()

            loss_sum += float(tot_loss.item())
            n_steps += 1
            gate_batch_mean /= max(1, len(per_item_losses))
            pbar.set_postfix(
                loss=f"{tot_loss.item():.4f}",
                gate=f"{gate_batch_mean:.3f}",
                sup=f"{(sup_sum/max(1,n_vids)):.4f}",
                gate_reg=f"{(gate_sum/max(1,n_vids)):.4f}",
                dual_bce=f"{(dual_bce_sum/max(1,n_vids)):.4f}",
                vids=n_vids
            )

        # validazione
        val = evaluate_epoch(dual, mmlp, va, device_t)
        dt = time.time() - t0
        tr_t = (mmlp.t_rgb.detach().item(), mmlp.t_dual.detach().item())
        logging.info(f"temps rgb={tr_t[0]:.3f} dual={tr_t[1]:.3f}")
        logging.info(f"[{ep:03d}] loss={loss_sum/max(1,n_steps):.4f} "
                     f"(sup={sup_sum/max(1,n_vids):.4f}, gate={gate_sum/max(1,n_vids):.4f}, dual_bce={dual_bce_sum/max(1,n_vids):.4f}) | "
                     f"valAUC={val['auc']:.4f} PR={val['pr_auc']:.4f} | {dt:.1f}s")

        if val["auc"] > best_auc:
            best_auc = val["auc"]
            torch.save({"model": mmlp.state_dict(),
                        "val": val,
                        "cfg": {"beta": beta, "k": k, "gamma": gamma, "zmode": zscore_mode, "zapply": zscore_apply}},
                       best_path)
            logging.info(f"Checkpoint: {best_path} (AUC={best_auc:.4f})")

    logging.info(f"Training finito. Best AUC={best_auc:.4f} -> {best_path}")


# --------------- CLI --------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Train mMLP fusion (AltFreezing per-video + Dual AU/LMK)")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--index", required=True)
    ap.add_argument("--csv-pervideo", required=True)
    ap.add_argument("--split-train", default="train")
    ap.add_argument("--split-val",   default="val")
    ap.add_argument("--dual-module", required=True)
    ap.add_argument("--dual-class",  required=True)
    ap.add_argument("--dual-ckpt",   required=True)
    ap.add_argument("--out",         required=True)
    ap.add_argument("--epochs",      type=int,   default=5)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--wd",          type=float, default=0.0)
    ap.add_argument("--T",           type=int,   default=8)
    ap.add_argument("--batch-videos", type=int,  default=1)
    ap.add_argument("--workers",      type=int,  default=4)
    ap.add_argument("--lambda-dual",  type=float, default=0.2)  # compat
    ap.add_argument("--device",       default=None)
    ap.add_argument("--zscore-mode",  choices=["clip", "global", "none"], default="clip")
    ap.add_argument("--zscore-apply", choices=["both", "au", "lmk"],      default="both")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)
    train_mmlp(
        index=args.index, csv_pervideo=args.csv_pervideo,
        split_train=args.split_train, split_val=args.split_val,
        dual_module=args.dual_module, dual_class=args.dual_class, dual_ckpt=args.dual_ckpt,
        out_dir=args.out, epochs=args.epochs, lr=args.lr, wd=args.wd,
        T=args.T, batch_videos=args.batch_videos, workers=args.workers,
        lambda_dual=args.lambda_dual, device=args.device,
        zscore_mode=args.zscore_mode, zscore_apply=args.zscore_apply  # type: ignore[attr-defined]
    )


if __name__ == "__main__":
    main()
