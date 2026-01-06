# compute_norm_stats.py
import os
import json
import argparse
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

LOG = logging.getLogger("compute_stats")

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    from pathlib import Path

def iter_clip_dirs(p: str):
    d = Path(p)
    au = d / "au_features.npy"
    lmk = d / "lmk_features.npy"
    if au.is_file() and lmk.is_file():
        yield str(d)
        return
    # fallback: cerca nelle clip_*
    for cp in sorted(d.glob("clip_*")):
        if (cp / "au_features.npy").is_file() and (cp / "lmk_features.npy").is_file():
            yield str(cp)


def _flatten_train_list(index_obj):
    """
    Supporta tre schemi possibili di index:
    A) {"train": [...], "val": [...], "test": [...]}  # liste di clip path
    B) {"splits": {"train": [...], "val": [...], "test": [...]}}
    C) {"train": {"tech1":[...], "tech2":[...], ...}, ...}  # per-tecnica

    Ritorna una lista di directory di clip per il train.
    """
    # Schema B con wrapper "splits"
    if isinstance(index_obj, dict) and "splits" in index_obj:
        return _flatten_train_list(index_obj["splits"])  # ricorsione: cadrà in A o C

    # Schema A (liste semplici)
    if isinstance(index_obj, dict) and "train" in index_obj:
        train = index_obj["train"]
        if isinstance(train, dict):  # Schema C
            out = []
            for v in train.values():
                out.extend(list(v))
            return out
        return list(train)

    # Schema "lista nuda": trattala come train-only
    if isinstance(index_obj, list):
        return list(index_obj)

    raise ValueError("Index JSON non contiene 'train' in uno schema riconosciuto.")

class SumStats:
    """
    Accumulatore con somma, somma dei quadrati e conteggio (per-feature).
    È numericamente stabile e molto veloce (evita loop per frame).
    """
    def __init__(self):
        self.count = 0  # numero totale di frame visti
        self.sum = None
        self.sumsq = None

    def update(self, X: np.ndarray):
        """
        X: array (T, F). Accumulo sui frame validi T.
        """
        if X is None or X.size == 0:
            return
        # Converte a float64 per stabilità numerica in somma/sumsq
        X = np.asarray(X, dtype=np.float64)
        if self.sum is None:
            self.sum = np.zeros(X.shape[1], dtype=np.float64)
            self.sumsq = np.zeros(X.shape[1], dtype=np.float64)
        self.sum += X.sum(axis=0)
        self.sumsq += np.square(X).sum(axis=0)
        self.count += X.shape[0]

    def finalize(self, eps: float = 1e-6):
        """
        Ritorna (mean, std) come float32. std è clampata a eps minimo.
        """
        if self.count <= 1:
            # fallback: evitiamo NaN, ritorna mean=0, std=1
            f = self.sum.shape[0] if self.sum is not None else 1
            mean = np.zeros((f,), dtype=np.float32)
            std = np.ones((f,), dtype=np.float32)
            return mean, std
        mean = self.sum / self.count
        var = self.sumsq / self.count - np.square(mean)
        var = np.maximum(var, eps**2)
        std = np.sqrt(var)
        return mean.astype(np.float32), std.astype(np.float32)

def run(base_path: str, index_json_path: str, out_path: str):
    LOG.info(f"Lettura index: {index_json_path}")
    with open(index_json_path, "r") as f:
        index_obj = json.load(f)

    # Estrai lista clip solo dal TRAIN (schema-agnostico)
    train_clips = _flatten_train_list(index_obj)
    LOG.info(f"Clip nel TRAIN: {len(train_clips)}")

    au_stats = SumStats()
    lmk_stats = SumStats()

    pbar = tqdm(train_clips, desc="Calcolo statistiche (global AU/LMK)")
    n_missing_au = n_missing_lmk = 0
    for clip_or_track in pbar:
        # risolvi base
        cbase = clip_or_track
        if not os.path.isabs(cbase) and not os.path.exists(cbase):
            cbase = os.path.join(base_path, clip_or_track)

        # espansione: usa dir stessa se ha i file, altrimenti scorri le clip_*
        for cdir in iter_clip_dirs(cbase):
            au_path = os.path.join(cdir, "au_features.npy")
            lmk_path = os.path.join(cdir, "lmk_features.npy")
            try:
                if os.path.exists(au_path):
                    A = np.load(au_path, mmap_mode="r"); au_stats.update(A)
                else:
                    n_missing_au += 1
                if os.path.exists(lmk_path):
                    L = np.load(lmk_path, mmap_mode="r"); lmk_stats.update(L)
                else:
                    n_missing_lmk += 1
            except Exception as e:
                LOG.warning(f"Errore su {cdir}: {e}")

    au_mean, au_std = au_stats.finalize()
    lmk_mean, lmk_std = lmk_stats.finalize()

    LOG.info(f"AU: frames={au_stats.count}  dims={au_mean.shape[0]}  missing={n_missing_au}")
    LOG.info(f"LMK: frames={lmk_stats.count} dims={lmk_mean.shape[0]} missing={n_missing_lmk}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, au_mean=au_mean, au_std=au_std, lmk_mean=lmk_mean, lmk_std=lmk_std)
    LOG.info(f"✅ Salvato: {out_path}")

if __name__ == "__main__":
    setup_logging()
    ap = argparse.ArgumentParser(description="Calcola media e std globali (per-feature) sul TRAIN.")
    ap.add_argument("--base", required=False, default="", help="Root del dataset (usata se i path nel JSON sono relativi).")
    ap.add_argument("--index-json", required=True, help="Path all'index JSON con gli split.")
    ap.add_argument("--out", default="./norm_stats.npz", help="Path file .npz di output.")
    args = ap.parse_args()

    run(args.base, args.index_json, args.out)
