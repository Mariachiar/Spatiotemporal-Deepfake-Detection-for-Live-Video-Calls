#!/usr/bin/env python3
# make_au_features.py
import os
import argparse
import glob
import numpy as np
import logging
from tqdm import tqdm
try:
    from data.make_lmk_features import extract_lmk_seq, KEY_LANDMARKS_IDXS
except Exception:
    try:
        from .make_lmk_features import extract_lmk_seq, KEY_LANDMARKS_IDXS
    except Exception:
        from make_lmk_features import extract_lmk_seq, KEY_LANDMARKS_IDXS



LOG = logging.getLogger("make_au")

# ----------------- AU UTILS -----------------
def infer_au_order(au_list, prefer_suffix=None):
    """
    au_list: list[dict] (una sequenza, T frame)
    prefer_suffix: se i tuoi key hanno suffissi (es. "_r" o "_c"), usa questo (es. prefer_suffix="_r")
    """
    keys = set()
    for d in au_list:
        keys.update(d.keys())
    keys = sorted(keys)
    if prefer_suffix:
        base = sorted({k.replace(prefer_suffix, "") for k in keys if k.endswith(prefer_suffix)})
        order = [b + prefer_suffix for b in base]
    else:
        order = keys
    return order

def au_dict_to_vec(d, order):
    return np.array([float(d.get(k, 0.0)) for k in order], dtype=np.float32)

def seq_au_to_features(au_seq, order, use_delta=True, use_delta2=True):
    X = np.stack([au_dict_to_vec(d, order) for d in au_seq], axis=0)  # [T, K]
    feats = [X]
    if use_delta:
        d1 = np.diff(X, axis=0, prepend=X[:1])
        feats.append(d1)
    if use_delta2:
        if not use_delta:
            d1 = np.diff(X, axis=0, prepend=X[:1])
        d2 = np.diff(d1, axis=0, prepend=d1[:1])
        feats.append(d2)
    F = np.concatenate(feats, axis=-1).astype(np.float32)             # [T, F_au]
    return F

# ----------------- CORE -----------------
def process_tree(base_dir, prefer_suffix=None, use_delta=True, use_delta2=True):
    base_dir = os.path.abspath(base_dir)
    LOG.info(f"Base dataset: {base_dir}")
    LOG.info(f"Config: prefer_suffix={prefer_suffix} | use_delta={use_delta} | use_delta2={use_delta2}")

    clips = glob.glob(os.path.join(base_dir, "**", "track_*", "clip_*"), recursive=True)
    if not clips:
        raise SystemExit(f"Nessuna clip trovata in {base_dir}")
    LOG.info(f"Clip trovate: {len(clips)}")

    n_ok = 0
    n_skip = 0
    n_err = 0

    pbar = tqdm(clips, desc="AU→features", unit="clip")
    for clip in pbar:
        try:
            au_path = os.path.join(clip, "aus.npy")
            out_path = os.path.join(clip, "au_features.npy")

            # skip se manca il sorgente
            if not os.path.isfile(au_path):
                LOG.debug(f"SKIP (manca aus.npy): {clip}")
                n_skip += 1
                continue

            # carica sequenza
            au_seq = np.load(au_path, allow_pickle=True).tolist()  # list[dict], len=T
            if not au_seq:
                LOG.debug(f"SKIP (sequenza vuota): {clip}")
                n_skip += 1
                continue

            # ordine e features
            au_order = infer_au_order(au_seq, prefer_suffix=prefer_suffix)
            F = seq_au_to_features(au_seq, au_order, use_delta, use_delta2)  # [T, F_au]

            # salva
            np.save(out_path, F)
            n_ok += 1

            # update barra
            pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err)
        except Exception as e:
            LOG.error(f"Errore su clip: {clip} | {e}")
            n_err += 1
            pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err)

    LOG.info(f"✅ Completato. OK={n_ok} | SKIP={n_skip} | ERR={n_err}")

# ----------------- CLI -----------------
def setup_logging(level: str):
    level = level.upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    LOG.info(f"Logging inizializzato: {level}")

# ===== API per training on-the-fly =====
def extract_au_seq(frames):
    """
    frames: List[np.ndarray HxWx3 RGB]
    ritorna: (T, 36) float32
    Implementazione proxy: 12 descrittori geometrici da landmarks → +Δ,+Δ2 = 36.
    Se possiedi un vero modello AU, sostituisci questa funzione con il tuo infer.
    """
    L = extract_lmk_seq(frames)  # (T,132)
    T = L.shape[0]
    if T == 0:
        return np.zeros((0, 36), np.float32)

    pts = L.reshape(T, -1, 2)  # (T,66,2)

    def dist(a, b):  # euclidea
        return np.linalg.norm(a - b, axis=-1)

    # indici utili nel sottoinsieme 66 (approssimazioni robuste):
    # occhi sx/dx (palpebre), bocca (apertura), sopracciglia (altezza), naso ↔ bocca (scala proxy)
    # NB: usiamo posizioni relative già normalizzate in extract_lmk_seq

    # mappature robuste (usa posizioni relative nel vettore 66):
    eyeL_top, eyeL_bot = 3, 8
    eyeR_top, eyeR_bot = 19, 24
    browL, browR = 16, 32
    nose_tip = 55
    mouth_top, mouth_bot = 53, 60
    mouth_left, mouth_right = 51, 57

    # 12 feature base
    f = []
    f.append(dist(pts[:, eyeL_top], pts[:, eyeL_bot]))          # EAR L
    f.append(dist(pts[:, eyeR_top], pts[:, eyeR_bot]))          # EAR R
    f.append(dist(pts[:, mouth_top], pts[:, mouth_bot]))        # apertura bocca
    f.append(dist(pts[:, mouth_left], pts[:, mouth_right]))     # larghezza bocca
    f.append(pts[:, browL, 1])                                  # altezza brow L (y)
    f.append(pts[:, browR, 1])                                  # altezza brow R (y)
    f.append(dist(pts[:, nose_tip], pts[:, mouth_top]))         # naso↔bocca top
    f.append(dist(pts[:, nose_tip], pts[:, mouth_bot]))         # naso↔bocca bot
    f.append(pts[:, mouth_left, 1] - pts[:, mouth_right, 1])    # asimmetria bocca (y)
    f.append(pts[:, eyeL_top, 0] - pts[:, eyeR_top, 0])         # divergenza occhi (x)
    f.append(pts[:, browL, 0] - pts[:, browR, 0])               # divergenza sopracciglia (x)
    f.append(pts[:, browL, 1] - pts[:, browR, 1])               # asimmetria sopracciglia (y)

    base = np.stack(f, axis=1).astype(np.float32)  # (T,12)

    # Δ e Δ2
    d1 = np.diff(base, axis=0, prepend=base[:1])
    d2 = np.diff(d1,   axis=0, prepend=d1[:1])

    F = np.concatenate([base, d1, d2], axis=1).astype(np.float32)  # (T, 36)
    return F


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Root del dataset (es. ./datasets/processed_dataset)")
    ap.add_argument("--prefer_suffix", default=None, help="Es. _r se i dict hanno chiavi tipo AU01_r/AU01_c")
    ap.add_argument("--no_delta", action="store_true")
    ap.add_argument("--no_delta2", action="store_true")
    ap.add_argument("--log-level", default="INFO", help="DEBUG | INFO | WARNING | ERROR")
    args = ap.parse_args()

    setup_logging(args.log_level)
    process_tree(
        args.base,
        prefer_suffix=args.prefer_suffix,
        use_delta=not args.no_delta,
        use_delta2=not args.no_delta2
    )
