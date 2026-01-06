#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Valutazione DualEncoder AU+LMK con analisi clip-by-clip **e** video-by-video.

Novità principali rispetto alle versioni precedenti
---------------------------------------------------
1) Analisi CLIP-BY-CLIP (come prima) + ANALISI VIDEO-BY-VIDEO (nuova):
   - Ogni video può contenere più persone (tracks). In ogni track ci sono più clip (di 8 frame ciascuna).
   - Due criteri di decisione a livello track/video (selezionabili con --agg-mode):
     * track_mean     : un track è FAKE se la **media** delle probabilità delle sue clip supera la soglia.
     * track_majority : un track è FAKE se **la maggioranza** delle sue clip è FAKE.
     In entrambi i casi: **se almeno un track è FAKE, tutto il video è FAKE**.

2) Metriche richieste:
   - Globali REAL vs FAKE (clip-level e/o video-level).
   - Per tecnica di manipolazione **e REAL**: accuratezza **a livello VIDEO**, specificando **su quanti video** è calcolata.

3) Configurazione modello selezionabile:
   - Defaults interni + caricamento automatico di args.json (dirname(ckpt)/args.json) o via --config
     con precedenza: CLI > config file > defaults.

4) Funzionalità utili mantenute:
   - Bilanciamenti (--max-per-tech, --balance), limit globale, sweep soglia (--sweep/--best-of),
     Target-FPR (--target-fpr), temperature scaling (--apply-temp), zscore globale/clip (--zscore),
     soglia salvata (--use-best-thresh).

Uso tipico
----------
python best_video_eval.py \
  --ckpt runs/phase1/best.pt \
  --index-cache runs/phase1/splits_used.json \
  --subset test \
  --apply-temp --use-best-thresh \
  --agg-mode track_mean \
  --video-metrics

Oppure per discovery da una directory (senza index-cache):
python best_video_eval.py --ckpt CKPT.pt --dir /path/dataset --video-metrics --agg-mode track_majority
"""
from __future__ import annotations
import os, json, argparse, random, sys
from pathlib import Path, PurePath
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Iterable
from types import MethodType

# consenti import dal pacchetto 'dualrun'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, average_precision_score,
)
from tqdm import tqdm

# ====== Import robusti dei moduli progetto (top-level o package) ======

    # layout package (es. data/, model/)
from data.dataset_dual import DualFeaturesClipDataset
from model.dual_encoder import DualEncoderAU_LMK

# subito dopo le import
def _norm(p: str) -> str:
    return str(PurePath(p))


# ====================== Costanti/Token ======================
REAL_TOKENS = {"original", "origina", "pristine", "authentic", "real", "celeb-real", "youtube-real"}

AggMode = ("track_mean", "track_majority", "track_median")

# ====================== Utilità JSON ======================
def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

# ====================== Deduzioni da path ======================
def _contains_real_token(p: PurePath) -> bool:
    parts_lower = [s.lower() for s in p.parts]
    return any(tok in parts_lower for tok in REAL_TOKENS)

def _tech_from_clip_path(p: PurePath) -> str:
    """Heuristica robusta per ottenere la tecnica dal path clip."""
    if _contains_real_token(p):
        return "REAL"
    parts = list(p.parts)
    for i, seg in enumerate(parts):
        if seg.lower().startswith("track_"):
            if i - 2 >= 0:
                tech = parts[i - 2]
                return "REAL" if tech.lower() in REAL_TOKENS else tech
    for i, seg in enumerate(parts):
        if seg.lower().startswith("clip"):
            if i - 3 >= 0:
                tech = parts[i - 3]
                return "REAL" if tech.lower() in REAL_TOKENS else tech
    try:
        tech = p.parents[2].name
    except IndexError:
        tech = p.parent.name or "UNKNOWN"
    return "REAL" if tech.lower() in REAL_TOKENS else tech

def _video_track_from_path(p: PurePath) -> Tuple[str, str]:
    """
    Estrae (video_id, track_id) da un path.
    Regole robuste:
    - track_id = primo segmento che inizia con 'track_'.
    - video_id = segmenti TRA 'tecnica' e 'track_*'.
      Se NON esistono segmenti tra tecnica e track (es. tech/track_000/clip_000/…),
      usa il track_id come video_id (meglio che usare la tecnica!).
    """
    parts = list(p.parts)

    # trova indice del primo 'track_*'
    i_track = next((i for i, s in enumerate(parts) if s.lower().startswith("track_")), None)
    if i_track is None:
        # Fallback ragionevole
        track_id = parts[-2] if len(parts) >= 2 else "track_unknown"
        video_id = parts[-3] if len(parts) >= 3 else track_id
        return video_id, track_id

    track_id = parts[i_track]

    # ipotizziamo layout standard: <base>/<tech>/[<video_path...>]/track_*/clip_*
    # la "tecnica" è il primo segmento dopo la base usata a monte (collect / discovery)
    # quindi, i segmenti tra indice 1 e i_track sono il "video path"
    if i_track > 1:
        between = parts[1:i_track]
        video_id = "/".join(between) if between else track_id
    else:
        # track_* direttamente sotto tecnica → niente cartella video: usa track_id
        video_id = track_id

    return video_id, track_id

# ====================== Scoperta clip ======================
def _discover_from_dir(dir_path: str, pre_limit_videos: int = 0):
    base = Path(dir_path).resolve()
    tech_of, vt_of, clip_dirs = {}, {}, []
    videos_seen = defaultdict(set)  # tech -> set(video_id)

    for root, _, files in os.walk(base):
        if "au_features.npy" in files and "lmk_features.npy" in files:
            p = Path(root).resolve()
            try:
                rel = p.relative_to(base); tech = rel.parts[0].lower() if len(rel.parts)>0 else "unknown"
            except Exception:
                tech = _tech_from_clip_path(p).lower()
            tech = "REAL" if tech in REAL_TOKENS else tech

            vid, tid = _video_track_from_path(p)
            if pre_limit_videos and (vid not in videos_seen[tech]) and len(videos_seen[tech]) >= pre_limit_videos:
                continue  # salta nuovi video di questa tecnica

            clip_dirs.append(str(p)); tech_of[str(p)] = tech
            vt_of[str(p)] = (vid, tid)
            videos_seen[tech].add(vid)
    clip_dirs.sort()
    return clip_dirs, tech_of, vt_of



def _from_training_splits(obj, subset=None) -> Tuple[List[str], Dict[str, str], Dict[str, Tuple[str,str]]]:
    if "splits" in obj and isinstance(obj["splits"], dict):
        obj = obj["splits"]
    tech_of: Dict[str, str] = {}
    vt_of: Dict[str, Tuple[str,str]] = {}
    clip_dirs: List[str] = []
    keys = ["train", "val", "test"]
    use_keys = [subset] if subset in keys else keys if subset is None else []
    for k in use_keys:
        if k in obj and isinstance(obj[k], list):
            for p_str in obj[k]:
                clip_dirs.append(p_str)
                tech = _tech_from_clip_path(Path(p_str)).lower()
                if tech in REAL_TOKENS:
                    tech = "REAL"
                tech_of[p_str] = tech
                vt_of[p_str] = _video_track_from_path(Path(p_str))
    clip_dirs = sorted(set(clip_dirs))
    return clip_dirs, tech_of, vt_of


def _from_tech_map(obj, subset=None) -> Tuple[List[str], Dict[str, str], Dict[str, Tuple[str,str]]]:
    tech_of: Dict[str, str] = {}
    vt_of: Dict[str, Tuple[str,str]] = {}
    clip_dirs: List[str] = []
    pairs = [(subset, obj[subset])] if subset else list(obj.items())
    for tech, lst in pairs:
        for p_str in lst:
            clip_dirs.append(p_str)
            t = str(tech).lower()
            if t == "__orig__" or t in REAL_TOKENS:
                tech_of[p_str] = "REAL"
            else:
                tech_of[p_str] = t
            vt_of[p_str] = _video_track_from_path(Path(p_str))
    clip_dirs = sorted(set(clip_dirs))
    return clip_dirs, tech_of, vt_of

def collect_clip_dirs(index_cache=None, subset=None, dir_path=None):
    if dir_path:
        return _discover_from_dir(dir_path)
    if not index_cache or not os.path.isfile(index_cache):
        raise SystemExit(f"Index cache non trovata: {index_cache}")
    obj = _load_json(index_cache)
    if isinstance(obj, dict) and ({"train", "val", "test"} <= set(obj.keys()) or "splits" in obj):
        return _from_training_splits(obj, subset=subset)
    if isinstance(obj, dict):
        return _from_tech_map(obj, subset=subset)
    raise SystemExit("Formato index_cache non riconosciuto.")

# ====================== Filtri/bilanciamenti ======================
def cap_per_tech(clip_dirs: List[str], tech_of: Dict[str, str], max_per_tech=0, seed=1):
    if not max_per_tech or max_per_tech <= 0:
        return clip_dirs, tech_of
    rnd = random.Random(seed)
    by_tech = defaultdict(list)
    for p in clip_dirs:
        by_tech[tech_of.get(p, "UNK")].append(p)
    for t in by_tech:
        rnd.shuffle(by_tech[t])
    limited = []
    for _, lst in by_tech.items():
        limited.extend(lst[:max_per_tech])
    limited = sorted(set(limited))
    new_tech_of = {p: tech_of.get(p, "UNK") for p in limited}
    return limited, new_tech_of

def balance_real_fake(clip_dirs: List[str], tech_of: Dict[str, str], seed=1):
    if not clip_dirs:
        return clip_dirs, tech_of
    rnd = random.Random(seed)
    real = [p for p in clip_dirs if tech_of.get(p, "UNK") == "REAL"]
    fake = [p for p in clip_dirs if tech_of.get(p, "UNK") != "REAL"]
    if not real or not fake:
        return clip_dirs, tech_of
    rnd.shuffle(real); rnd.shuffle(fake)
    k = min(len(real), len(fake))
    balanced = sorted(real[:k] + fake[:k])
    new_tech_of = {p: tech_of.get(p, "UNK") for p in balanced}
    return balanced, new_tech_of

# ====================== Config modello ======================
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

# ====================== Sanity ======================
def sanity_feature_stats(dl):
    try:
        X, y0 = next(iter(dl))
        A0, L0 = (X[0], X[1]) if isinstance(X,(tuple,list)) else (X["A"], X["L"])
        A0 = A0.float(); L0 = L0.float()
        print("AU  mean/std:", round(A0.mean().item(), 6), round(A0.std().item(), 6))
        print("LMK mean/std:", round(L0.mean().item(), 6), round(L0.std().item(), 6))
        print("y   mean:", round(y0.float().mean().item(), 6), "(1=fake)")
    except Exception as e:
        print("[SANITY] Impossibile leggere primo batch per stats:", e)


@torch.no_grad()
def params_l2_norm(model):
    s = 0.0
    for p in model.parameters():
        if p.requires_grad:
            s += p.norm().item()
    return s

def _extract_batch(X):
    trk_id = vid_id = paths = None
    if isinstance(X, (tuple, list)):
        A, L, lengths = X[0], X[1], X[2]
        # possibili code: [..., trk_id, vid_id] oppure [..., trk_id, vid_id, paths]
        if len(X) >= 6:
            trk_id, vid_id, paths = X[-3], X[-2], X[-1]
        elif len(X) == 5:
            trk_id, vid_id = X[-2], X[-1]
    else:
        A, L, lengths = X["A"], X["L"], X["lengths"]
        trk_id = X.get("trk_id"); vid_id = X.get("vid_id"); paths = X.get("paths")
    return A, L, lengths, trk_id, vid_id, paths



# ====================== Eval core (clip-by-clip) ======================
def evaluate_ckpt(
    ckpt_path: str,
    clip_dirs: List[str],
    tech_of: Dict[str, str],
    vt_of: Dict[str, Tuple[str,str]],
    model_cfg: dict,
    T: int = 8,
    batch_size: int = 128,
    num_workers: int = 8,
    use_logits: bool = False,
    prob_thresh: float = 0.5,
    do_sanity: bool = True,
    zscore: str = "none",
    random_crop: bool = False,
    norm_stats_path: str | None = None,
    apply_temp: bool = False,
    zscore_apply: str = "both",  # "both" | "au" | "lmk"
    temp_file: str | None = None,
    stitch_k: int = 1,   # <-- 
):
    if not clip_dirs:
        raise SystemExit("Nessuna clip da valutare.")

    # Global zscore
    if zscore == "global":
        if norm_stats_path:
            os.environ["NORM_STATS"] = norm_stats_path
        elif "NORM_STATS" not in os.environ:
            raise SystemExit("zscore='global' richiesto ma non hai passato --norm-stats e NORM_STATS non è settata.")

    ds = DualFeaturesClipDataset(
        clip_dirs=clip_dirs, T=T, validate=False, mmap=True,
        is_train=False, random_crop=random_crop, zscore=zscore, zscore_apply=zscore_apply,  stitch_k=stitch_k
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0 and device == "cuda"),
    )
    if do_sanity:
        sanity_feature_stats(dl)



    # Modello
        # Modello
    mlp_ratio = float(model_cfg.get("dim_ff", 256)) / float(model_cfg.get("d_model", 128))
    model = DualEncoderAU_LMK(
        au_dim=ds.au_dim, lmk_dim=ds.lmk_dim,
    d_model=model_cfg.get("d_model", 128),
    heads=model_cfg.get("nhead", 4),          # ← mapping
    depth=model_cfg.get("num_layers", 2),     # ← mapping
    mlp_ratio=mlp_ratio,                      # ← dim_ff / d_model
    dropout=model_cfg.get("dropout", 0.15),
    use_dat=False, domain_classes=0,
    ).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    if isinstance(sd, torch.nn.Module):
        sd = sd.state_dict()

    # --- RIMAPPA NOMI VECCHI → NUOVI ---
    rename = {
        "au_enc.pooling.attention_vector": "au_enc.pool.v",
        "lmk_enc.pooling.attention_vector": "lmk_enc.pool.v",
    }
    sd = { (rename.get(k, k)): v for k, v in sd.items() }

    model_sd = model.state_dict()
    keep = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    drop = sorted(set(sd) - set(keep))
    print(f"[LOAD] Keeping {len(keep)} keys, dropping {len(drop)} due to shape/name mismatch.")
    if drop:
        for k in drop[:10]:
            print(f"  - drop: {k}")
    missing, unexpected = model.load_state_dict(keep, strict=False)
    print(f"[LOAD] Missing in ckpt: {missing} | Unexpected in ckpt: {unexpected}")
    model.eval()

    # Temperature scaling
    T_star = 1.0
    if apply_temp:
        ck_dir = os.path.dirname(ckpt_path)
        temp_path = temp_file or os.path.join(ck_dir, "temperature.txt")
        if os.path.isfile(temp_path):
            try:
                with open(temp_path, "r") as f:
                    T_star = float(f.read().strip())
                print(f"[TEMP] Using T*={T_star:.6f} from {temp_path}")
            except Exception as e:
                print(f"[TEMP] Warning: failed to read {temp_path}: {e}. Using T*=1.0")
        else:
            print(f"[TEMP] No temperature file found at {temp_path}. Using T*=1.0")

    all_y: List[int] = []
    all_p: List[float] = []
    all_pred: List[int] = []

    # Per-clip meta (aligned con clip_dirs/loader output)
    meta_clip: List[Tuple[str,str,str]] = []  # (clip_dir, video_id, track_id)
        
    ordered_dirs = list(ds.clip_dirs)

    with torch.no_grad():
        first = True
        idx = 0
        for X, y in tqdm(dl, desc="Eval", unit="batch"):
            A, L, t_valid, trk_id, vid_id, paths = _extract_batch(X)

            A = A.to(device, non_blocking=(device == "cuda"))
            L = L.to(device, non_blocking=(device == "cuda"))
            t_valid = t_valid.to(device, non_blocking=(device == "cuda"))

            out = model(A, L, lengths=t_valid, dat_lambda=0.0)
            logits = out["bin_logits"].view(-1)
            if apply_temp and (T_star != 1.0):
                logits = logits / float(T_star)

            if first:
                lg = logits.detach().cpu().numpy()
                print("Logit stats -> min: %.4f  max: %.4f  mean: %.4f" % (lg.min(), lg.max(), lg.mean()))
                first = False

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (logits.cpu().numpy() > 0).astype(np.int64) if use_logits else (probs > prob_thresh).astype(np.int64)
            y_np = y.numpy().astype(np.int64)

            all_y.extend(y_np.tolist())
            all_p.extend(probs.tolist())
            all_pred.extend(preds.tolist())

            bsz = len(y_np)
            for b in range(bsz):
                if paths is not None:
                    key = str(PurePath(paths[b]))
                else:
                    pos = idx + b
                    if pos >= len(ordered_dirs):
                        continue
                    key = str(PurePath(ordered_dirs[pos]))
                v, t = vt_of.get(key, _video_track_from_path(PurePath(key)))
                meta_clip.append((key, v, t))
            idx += bsz

    # ---- Metriche globali clip-level ----
    metrics_clip = compute_global_metrics(all_y, all_pred, all_p)

    # ---- Metriche per tecnica (clip accuracy) ----
    tech_acc_clip = compute_per_tech_accuracy_clip(ordered_dirs, tech_of, all_y, all_pred)


    return np.array(all_y), np.array(all_p), np.array(all_pred), meta_clip, metrics_clip, tech_acc_clip

# ====================== Metriche helper ======================
def compute_global_metrics(y_true: Iterable[int], y_pred: Iterable[int], y_score: Optional[Iterable[float]] = None) -> dict:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    out = {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        y_score = np.asarray(list(y_score), dtype=float)
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
            out["pr_auc"]  = float(average_precision_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = float("nan")
            out["pr_auc"]  = float("nan")
    return out

def compute_per_tech_accuracy_clip(clip_dirs, tech_of, y_true, y_pred):
    n = min(len(clip_dirs), len(y_true), len(y_pred))
    if n < len(clip_dirs):
        print(f"[WARN] per-tech: uso {n}/{len(clip_dirs)} voci (stitch_k>1?)")
    per_tech_true = defaultdict(list)
    per_tech_pred = defaultdict(list)
    for i in range(n):
      p = clip_dirs[i]
      key = str(PurePath(p))
      tname = tech_of.get(key, tech_of.get(p, _tech_from_clip_path(PurePath(p))))
      tname = "REAL" if str(tname).lower() in REAL_TOKENS else str(tname)
      per_tech_true[tname].append(int(y_true[i]))
      per_tech_pred[tname].append(int(y_pred[i]))
    out = {}
    for tname, ys in per_tech_true.items():
      y_t = np.asarray(ys, dtype=int)
      y_p = np.asarray(per_tech_pred[tname], dtype=int)
      out[tname] = {"accuracy": float(accuracy_score(y_t, y_p)), "clips": int(len(y_t))}
    return out


# ====================== Aggregazione video-by-video ======================
def aggregate_video_predictions(
    clip_dirs: List[str],
    tech_of: Dict[str,str],
    meta_clip: List[Tuple[str,str,str]],  # (clip_path, video_id, track_id)
    y_true_clip: np.ndarray,
    y_score_clip: np.ndarray,
    prob_thresh: float,
    agg_mode: str = "track_mean",
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    assert agg_mode in AggMode, f"--agg-mode deve essere in {AggMode}"

    videos: Dict[str, dict] = {}

    for i, (clip_path, vid, tid) in enumerate(meta_clip):
        prob = float(y_score_clip[i])
        pred = 1 if prob >= prob_thresh else 0
        y    = int(y_true_clip[i])

        # tecnica dalla clip (mapping affidabile)
        tech = tech_of.get(clip_path, _tech_from_clip_path(PurePath(clip_path)))
        tech = "REAL" if str(tech).lower() in REAL_TOKENS else str(tech)

        # *** CHIAVE VIDEO NAMESPACED PER TECNICA ***
        vid_key = f"{tech}::{vid}"

        if vid_key not in videos:
            videos[vid_key] = {"tracks": {}, "y_list": [], "tech": tech, "orig_video_id": vid}

        if tid not in videos[vid_key]["tracks"]:
            videos[vid_key]["tracks"][tid] = {"probs": [], "preds": [], "y": []}

        videos[vid_key]["tracks"][tid]["probs"].append(prob)
        videos[vid_key]["tracks"][tid]["preds"].append(pred)
        videos[vid_key]["tracks"][tid]["y"].append(y)
        videos[vid_key]["y_list"].append(y)

    # Determina predizione per ogni track e per il video
    for vid_key, V in videos.items():
        track_scores = []
        any_track_fake = False

        for tid, T in V["tracks"].items():
            probs = np.asarray(T["probs"], dtype=float)
            preds = np.asarray(T["preds"], dtype=int)

            if agg_mode == "track_mean":
                score = float(probs.mean()) if len(probs) else 0.0
                track_pred = int(score >= prob_thresh)
            elif agg_mode == "track_median":
                score = float(np.median(probs)) if len(probs) else 0.0
                track_pred = int(score >= prob_thresh)
            else:  # track_majority
                frac_fake = float((preds == 1).mean()) if len(preds) else 0.0
                track_pred = int(frac_fake >= 0.5)
                score = float(probs.mean()) if len(probs) else 0.0

            T["track_pred"] = int(track_pred)
            T["track_score"] = float(score)
            track_scores.append(score)
            if track_pred == 1:
                any_track_fake = True

        V["video_pred"]  = 1 if any_track_fake else 0
        V["video_score"] = float(max(track_scores) if track_scores else 0.0)

        y_list = np.asarray(V["y_list"], dtype=int)
        V["y_true"] = int(np.argmax(np.bincount(y_list, minlength=2))) if len(y_list) else 0

    # Per-tecnica: elenco video (conteggio videos=…)
    per_tech_video: Dict[str, dict] = defaultdict(lambda: {"videos": []})
    for vid_key, V in videos.items():
        per_tech_video[V["tech"]]["videos"].append(vid_key)

    return videos, per_tech_video


def compute_video_metrics(videos: Dict[str, dict]) -> Tuple[dict, Dict[str, dict]]:
    """Ritorna metriche globali video-level e accuracy per tecnica con conteggio video."""
    y_true = [V["y_true"] for V in videos.values()]
    y_pred = [V["video_pred"] for V in videos.values()]
    y_score = [V["video_score"] for V in videos.values()]
    metrics_video = compute_global_metrics(y_true, y_pred, y_score)

    # Per tecnica (accuracy + numero di video)
    per_tech = defaultdict(lambda: {"videos": 0, "correct": 0})
    for vid, V in videos.items():
        tech = V["tech"]
        per_tech[tech]["videos"] += 1
        per_tech[tech]["correct"] += int(V["video_pred"] == V["y_true"])

    per_tech_acc = {}
    for tech, d in per_tech.items():
        n = d["videos"]
        acc = d["correct"] / n if n > 0 else 0.0
        per_tech_acc[tech] = {"accuracy": float(acc), "videos": int(n)}

    return metrics_video, per_tech_acc

# ====================== Threshold sweep ======================
def sweep_threshold(all_y: np.ndarray, all_p: np.ndarray, mode="acc", target_fpr=None):
    if target_fpr is not None:
        fpr, tpr, thr = roc_curve(all_y, all_p)
        cand = [(fp, tp, th) for fp, tp, th in zip(fpr, tpr, thr)]
        cand = sorted(cand, key=lambda x: (abs(x[0]-target_fpr), x[0]))
        best = None
        for fp, tp, th in cand:
            if fp <= target_fpr:
                best = (fp, tp, th); break
        if best is None:
            best = cand[0]
        fp, tp, th = best
        preds_best = (all_p >= th).astype(int)
        acc = (preds_best == all_y).mean()
        prec = precision_score(all_y, preds_best, zero_division=0)
        rec  = recall_score(all_y, preds_best, zero_division=0)
        f1   = f1_score(all_y, preds_best, zero_division=0)
        cm   = confusion_matrix(all_y, preds_best)
        print("\n=== THRESHOLD @ TARGET FPR ===")
        print(f"Target FPR: {target_fpr:.3f}  -> chosen thr={th:.4f}  (FPR≈{fp:.4f}, TPR≈{tp:.4f})")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print("Confusion Matrix:\n", cm)
        return th

    best_t, best_score = 0.5, -1.0
    grid = np.linspace(0.05, 0.95, 19)
    for t in grid:
        preds_t = (all_p >= t).astype(int)
        if mode == "acc":
            score = (preds_t == all_y).mean()
        elif mode == "youden":
            tn, fp, fn, tp = confusion_matrix(all_y, preds_t).ravel()
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            score = tpr - fpr
        elif mode == "f1":
            score = f1_score(all_y, preds_t, zero_division=0)
        else:
            raise SystemExit(f"Modo sweep non supportato: {mode}")
        if score > best_score:
            best_score, best_t = score, t

    preds_best = (all_p >= best_t).astype(int)
    prec = precision_score(all_y, preds_best, zero_division=0)
    rec  = recall_score(all_y, preds_best, zero_division=0)
    f1   = f1_score(all_y, preds_best, zero_division=0)
    cm   = confusion_matrix(all_y, preds_best)

    print("\n=== THRESHOLD SWEEP ===")
    print(f"Miglior soglia ({mode}): {best_t:.2f}  -> {mode}={best_score:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    return best_t

# ====================== CLI & main ======================
def build_argparser():
    ap = argparse.ArgumentParser("Valutazione DualEncoder AU+LMK (clip e video)")
    ap.add_argument("--ckpt", required=True, help="Checkpoint .pt")
    ap.add_argument("--index-cache", default=None, help="JSON: split usato nel training o mappa tecnica->clip")
    ap.add_argument("--subset", default=None, help="train|val|test (se split) oppure nome tecnica|__ORIG__ (se tech map)")
    ap.add_argument("--dir", default=None, help="Valuta tutte le clip sotto questa directory")

    # Dataloader / dataset
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8, dest="num_workers")
    ap.add_argument("--zscore", choices=["none", "clip", "global"], default="none")
    ap.add_argument("--norm-stats", default=None, help="Path .npz per zscore='global' (altrimenti usa env NORM_STATS)")
    ap.add_argument("--random-crop", action="store_true", help="Eval non-deterministica (sconsigliata)")

    # Soglia / logit
    ap.add_argument("--use-logits", action="store_true", help="Soglia su logit=0 invece che su probabilità")
    ap.add_argument("--thresh", type=float, default=0.5, help="Soglia su probabilità (ignorata se --use-logits)")

    # Sweep soglia
    ap.add_argument("--sweep", action="store_true", help="Cerca automaticamente la soglia migliore (compatibile con --best-of)")
    ap.add_argument("--best-of", choices=["acc", "youden", "f1"], default="acc", help="Criterio per --sweep")
    ap.add_argument("--target-fpr", type=float, default=None, help="Se impostato, seleziona soglia con FPR<=target")

    # Temperature scaling & soglie salvate
    ap.add_argument("--apply-temp", action="store_true", help="Applica temperature scaling leggendo T* da file")
    ap.add_argument("--temp-file", default=None, help="File con T* (default: dirname(ckpt)/temperature.txt)")
    ap.add_argument("--use-best-thresh", action="store_true", help="Usa la soglia salvata (ignora sweep/target-fpr)")
    ap.add_argument("--best-thresh-file", default=None,
                    help="File soglia (default: calibrated->best_threshold_calibrated.txt, altrimenti best_threshold.txt)")

    # Filtri/ricampionamento
    ap.add_argument("--max-per-tech", type=int, default=0, dest="max_per_tech", help="Al più N clip per tecnica (inclusi REAL)")
    ap.add_argument("--balance", action="store_true", help="Bilancia REAL vs FAKE (sottocampiona)")
    ap.add_argument("--limit", type=int, default=0, help="Limite globale #clip dopo i filtri")
    ap.add_argument("--seed", type=int, default=1, help="Seed per shuffle deterministico")

    # Config modello
    ap.add_argument("--config", default=None, help="Path ad args.json del training (fallback: dirname(ckpt)/args.json)")
    ap.add_argument("--d-model", type=int, default=None, dest="d_model")
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--ff-dim", type=int, default=None, dest="ff_dim")
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--pre-limit-videos", type=int, default=20,
                help="Limita già in discovery il numero di video per tecnica (early stop).")
    ap.add_argument('--zscore_apply', default='both', choices=['both','au','lmk'],
                  help='A quali modalità applicare lo z-score.')


    # Video-level
    ap.add_argument("--video-metrics", action="store_true", help="Calcola metriche a livello VIDEO oltre a quelle clip")
    ap.add_argument("--agg-mode", choices=list(AggMode), default="track_mean",
                    help="Criterio track: track_mean (media clip >= soglia) | track_majority (maggioranza clip fake)")
    
    # --- Video-level threshold (calcolo opzionale + riuso) ---
    ap.add_argument("--calc-video-thresh", action="store_true",
                    help="Calcola soglia a livello video (ROC su validation) e la stampa/salva.")
    ap.add_argument("--video-best-of", choices=["youden","balacc","acc","f1","auc"], default="youden",
                    help="Criterio per la scelta soglia video se --calc-video-thresh.")
    ap.add_argument("--video-target-fpr", type=float, default=None,
                    help="Se impostato, vincola la soglia video a FPR <= target (ROC).")
    ap.add_argument("--save-video-thresh", default=None,
                    help="Path file per salvare la soglia video (default: dirname(ckpt)/best_video_threshold.txt).")
    ap.add_argument("--use-best-video-thresh", action="store_true",
                    help="Usa soglia video salvata da file (senza ricalcolarla).")
    ap.add_argument("--video-best-file", default=None,
                    help="Percorso alternativo del file soglia video (se diverso dal default).")

    ap.add_argument("--limit-videos", type=int, default=None,
                help="Limita il numero di VIDEO selezionati (preserva tutte le clip dei video scelti)")
    
    ap.add_argument("--stitch_k", type=int, default=1)

    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    if (args.index_cache is None) and (args.dir is None):
        raise SystemExit("Specifica --index-cache o --dir.")

    # Raccogli clip (con tecnica + (video, track))
    clip_dirs, tech_of, vt_of = collect_clip_dirs(
        index_cache=args.index_cache, subset=args.subset, dir_path=args.dir
    )

    # --- Filtri ---
    # 1) cap_per_tech
    clip_dirs, tech_of = cap_per_tech(clip_dirs, tech_of, max_per_tech=args.max_per_tech, seed=args.seed)
    # riallinea mapping
    tech_of = {p: tech_of.get(p, "UNK") for p in clip_dirs}
    vt_of   = {p: vt_of.get(p, _video_track_from_path(PurePath(p))) for p in clip_dirs}

    # 2) balance
    if args.balance:
        clip_dirs, tech_of = balance_real_fake(clip_dirs, tech_of, seed=args.seed)
        tech_of = {p: tech_of.get(p, "UNK") for p in clip_dirs}
        vt_of   = {p: vt_of.get(p, _video_track_from_path(PurePath(p))) for p in clip_dirs}

    # 3bis) limit-videos (se definito)
    if getattr(args, "limit_videos", None):
        by_video = {}
        for p in clip_dirs:
            vid, _ = vt_of[p]
            by_video.setdefault(vid, []).append(p)
        keep_vids = sorted(by_video.keys())[:int(args.limit_videos)]
        clip_dirs = sorted([p for v in keep_vids for p in by_video[v]])
        tech_of = {p: tech_of[p] for p in clip_dirs}
        vt_of   = {p: vt_of[p]   for p in clip_dirs}


    # 3) limit
    if args.limit and args.limit > 0:
        clip_dirs = clip_dirs[:args.limit]
        tech_of = {p: tech_of.get(p, "UNK") for p in clip_dirs}
        vt_of   = {p: vt_of.get(p, _video_track_from_path(PurePath(p))) for p in clip_dirs}

    # --- Normalizza path/chiavi una volta per tutte ---
    clip_dirs = [str(PurePath(p)) for p in clip_dirs]
    tech_of   = {
        str(PurePath(p)): tech_of.get(p, tech_of.get(str(PurePath(p)), "UNK"))
        for p in clip_dirs
    }
    vt_of     = {
        str(PurePath(p)): vt_of.get(
            p,
            vt_of.get(str(PurePath(p)), _video_track_from_path(PurePath(p)))
        )
        for p in clip_dirs
    }


    if not clip_dirs:
        raise SystemExit("Nessuna clip trovata dopo i filtri.")

    # Config modello
    model_cfg = resolve_model_cfg(args)

    # Soglia salvata?
    if args.use_best_thresh:
        ck_dir = os.path.dirname(args.ckpt)
        if args.best_thresh_file is None:
            cand1 = os.path.join(ck_dir, "best_threshold_calibrated.txt")
            cand2 = os.path.join(ck_dir, "best_threshold.txt")
            best_thresh_file = cand1 if os.path.isfile(cand1) else cand2
        else:
            best_thresh_file = args.best_thresh_file
        if os.path.isfile(best_thresh_file):
            try:
                with open(best_thresh_file, "r") as f:
                    args.thresh = float(f.read().strip())
                print(f"[THRESH] Using saved threshold t={args.thresh:.6f} from {best_thresh_file}")
            except Exception as e:
                print(f"[THRESH] Warning: failed to read {best_thresh_file}: {e}. Falling back to --thresh")
        else:
            print(f"[THRESH] File not found: {best_thresh_file}. Falling back to --thresh")
    
    if not hasattr(args, "stitch_k"): args.stitch_k = 1

    # ===== Eval clip-by-clip =====
    all_y, all_p, all_pred, meta_clip, metrics_clip, tech_acc_clip = evaluate_ckpt(
        ckpt_path=args.ckpt,
        clip_dirs=clip_dirs,
        tech_of=tech_of,
        vt_of=vt_of,
        model_cfg=model_cfg,
        T=args.T,
        batch_size=args.batch,
        num_workers=args.num_workers,
        use_logits=args.use_logits,
        prob_thresh=args.thresh,
        do_sanity=True,
        zscore=args.zscore,
        zscore_apply= args.zscore_apply,  # "both" | "au" | "lmk"
        random_crop=args.random_crop,
        norm_stats_path=args.norm_stats,
        apply_temp=args.apply_temp,
        temp_file=args.temp_file,
        stitch_k=args.stitch_k
    )

    # === Report clip-level ===
    print("\n=== GLOBAL (CLIP-LEVEL) ===")
    print(f"Samples  : {metrics_clip['samples']}")
    print(f"Accuracy : {metrics_clip['accuracy']:.4f}")
    print(f"Precision: {metrics_clip['precision']:.4f}")
    print(f"Recall   : {metrics_clip['recall']:.4f}")
    print(f"F1-score : {metrics_clip['f1']:.4f}")
    print(f"ROC AUC  : {metrics_clip.get('roc_auc', float('nan')):.4f}")
    print(f"PR  AUC  : {metrics_clip.get('pr_auc', float('nan')):.4f}")
    print("Confusion Matrix:\n", np.array(metrics_clip['confusion_matrix']))

    print("\n=== PER TECNICA (CLIP ACCURACY) ===")
    for tname in sorted(tech_acc_clip.keys()):
        info = tech_acc_clip[tname]
        print(f"{tname:18s}  acc={info['accuracy']:.4f}  (clips={info['clips']})")

    if args.use_logits:
        print("\n[AVVISO] --sweep/--target-fpr ignorati con --use-logits (soglia fissa su logit=0).")
    elif not args.use_best_thresh:
        chosen = None
        if args.target_fpr is not None:
            chosen = sweep_threshold(all_y, all_p, target_fpr=args.target_fpr)
        elif args.sweep:
            chosen = sweep_threshold(all_y, all_p, mode=args.best_of)
        if chosen is not None:
            print(f"\n>> Soglia consigliata: {chosen:.4f}")

    # ===== Video-level (opzionale) =====
    need_video = args.video_metrics or args.calc_video_thresh or args.use_best_video_thresh

    if need_video:
        # 1) Aggregazione per-video
        videos, per_tech_video_map = aggregate_video_predictions(
            clip_dirs=clip_dirs,
            tech_of=tech_of,
            meta_clip=meta_clip,
            y_true_clip=all_y,
            y_score_clip=all_p,
            prob_thresh=args.thresh,
            agg_mode=args.agg_mode,
        )

        # 2) ROC vettori
        y_true_video  = np.array([V["y_true"] for V in videos.values()], dtype=int)
        y_score_video = np.array([V["video_score"] for V in videos.values()], dtype=float)

        # 3) Soglia video
        t_video = None
        if args.calc_video_thresh:
            # ATTENZIONE: verifica il path del modulo thresholds (train.thresholds vs dualrun/train/thresholds)
            from train.thresholds import threshold_from_roc
            t_video, stats_v = threshold_from_roc(
                y_score_video, y_true_video,
                metric=args.video_best_of,
                target_fpr=args.video_target_fpr
            )
            print("\n=== VIDEO THRESHOLD (from ROC on validation) ===")
            print(f"metric={args.video_best_of}  target_fpr={args.video_target_fpr}")
            print(f"t_video={t_video:.6f}  -> acc={stats_v['acc']:.4f}  youden={stats_v['youden']:.4f}  "
                  f"balacc={stats_v['balacc']:.4f}  f1={stats_v['f1']:.4f}  FPR={stats_v['FPR']:.4f}  TPR={stats_v['TPR']:.4f}")

            out_path = args.save_video_thresh or os.path.join(os.path.dirname(args.ckpt), "best_video_threshold.txt")
            try:
                with open(out_path, "w") as f:
                    f.write(f"{float(t_video):.6f}\n")
                print(f"[VIDEO THRESH] Saved video threshold to: {out_path}")
            except Exception as e:
                print(f"[VIDEO THRESH] Warning: could not save to {out_path}: {e}")

        if (t_video is None) and args.use_best_video_thresh:
            best_file = args.video_best_file or os.path.join(os.path.dirname(args.ckpt), "best_video_threshold.txt")
            if os.path.isfile(best_file):
                try:
                    with open(best_file, "r") as f:
                        t_video = float(f.read().strip())
                    print(f"[VIDEO THRESH] Using saved video threshold t={t_video:.6f} from {best_file}")
                except Exception as e:
                    print(f"[VIDEO THRESH] Warning: failed to read {best_file}: {e}")

        if t_video is None:
            t_video = 0.5
            print(f"[VIDEO THRESH] No video threshold selected; defaulting to t_video={t_video:.6f}")

        # 4) Metriche @ soglia video
        y_pred_video = (y_score_video >= float(t_video)).astype(int)
        acc  = accuracy_score(y_true_video, y_pred_video)
        prec = precision_score(y_true_video, y_pred_video, zero_division=0)
        rec  = recall_score(y_true_video, y_pred_video, zero_division=0)
        f1   = f1_score(y_true_video, y_pred_video, zero_division=0)
        roc_auc = float(roc_auc_score(y_true_video, y_score_video))
        pr_auc  = float(average_precision_score(y_true_video, y_score_video))
        fpr, tpr, _ = roc_curve(y_true_video, y_score_video)
        cm   = confusion_matrix(y_true_video, y_pred_video)
        print("\n=== GLOBAL (VIDEO-LEVEL @ t_video) ===")
        print(f"Samples  : {len(y_true_video)}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")
        print(f"PR  AUC  : {pr_auc:.4f}")
        print("Confusion Matrix:\n", cm)

        # 5) Metriche baseline video (se richieste)
        if args.video_metrics:
            metrics_video, per_tech_acc = compute_video_metrics(videos)
            print("\n================ VIDEO-LEVEL ================")
            print(f"Criterio track: {args.agg_mode}")
            print("\n--- GLOBAL (VIDEO-LEVEL) ---")
            print(f"Videos   : {len(videos)}")
            print(f"Accuracy : {metrics_video['accuracy']:.4f}")
            print(f"Precision: {metrics_video['precision']:.4f}")
            print(f"Recall   : {metrics_video['recall']:.4f}")
            print(f"F1-score : {metrics_video['f1']:.4f}")
            print(f"ROC AUC  : {metrics_video.get('roc_auc', float('nan')):.4f}")
            print(f"PR  AUC  : {metrics_video.get('pr_auc', float('nan')):.4f}")
            print("Confusion Matrix:\n", np.array(metrics_video['confusion_matrix']))

            print("\n--- PER TECNICA (VIDEO ACCURACY con conteggio video) ---")
            for tech in sorted(per_tech_acc.keys()):
                acc_t = per_tech_acc[tech]["accuracy"]
                n_t   = per_tech_acc[tech]["videos"]
                print(f"{tech:18s}  acc={acc_t:.4f}  (videos={n_t})")

        # 6) Avviso su eventuali label clip incoerenti nel video (una sola volta)
        inconsistent = []
        for _, V in videos.items():
            ys = set(V["tracks"][tid]["y"][0] for tid in V["tracks"] if V["tracks"][tid]["y"])
            if len(ys) > 1:
                inconsistent.append(True)
        if inconsistent:
            print(f"\n[WARNING] {len(inconsistent)} video hanno label per-clip incoerenti. "
                  f"È stato usato il majority vote sul video.")

if __name__ == "__main__":
    main()
