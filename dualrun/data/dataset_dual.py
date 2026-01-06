# dataset_dual.py
import os, logging, numpy as np, torch, math
from torch.utils.data import Dataset
from typing import Iterable, List, Tuple, Optional, Dict, Union
import re

LOG = logging.getLogger("dual_dataset")

# ---------------- inferenza tecnica dal path (usata anche per DAT) ----------------
def _infer_tech_from_path(path: str) -> str:
    p = path.lower().replace("\\", "/")
    parts_raw = [seg for seg in p.split("/") if seg]

    REAL_TOKENS = {"original", "origina", "pristine", "authentic", "real", "youtube-real", "celeb-real"}
    if any(seg in REAL_TOKENS for seg in parts_raw):
        return "real"

    def norm(s: str) -> str: return s.replace("-", "").replace("_", "")

    ALIASES = {
        "deepfakedetection": "dfdc", "dfdc": "dfdc",
        "deepfakes": "deepfakes", "face2face": "face2face",
        "faceswap": "faceswap", "neuraltextures": "neuraltextures",
        "faceshifter": "faceshifter", "stylegan": "stylegan",
        "styleswap": "styleswap",
        "celebdf": "celebdf", "celebsynthesis": "celebdf",
        "celebd": "celebd", "uadfv": "uadfv",
        "ffpp": "ffpp", "ff++": "ffpp",
    }
    for seg in parts_raw:
        k = norm(seg)
        if k in ALIASES: return ALIASES[k]
    for k in list(ALIASES.keys()):
        if f"/{k}/" in p: return ALIASES[k]

    SKIP_PREFIXES = ("track_", "fold_", "split_", "part_", "seg_")
    parts = [seg for seg in parts_raw if not any(seg.startswith(pr) for pr in SKIP_PREFIXES)]
    for i, seg in enumerate(parts):
        if seg.startswith("clip_") and i > 0:
            parent = parts[i - 1]; parent_k = norm(parent)
            return ALIASES.get(parent_k, parent)
    return "unknown"


# ----------------------------- Dataset AU+LMK (clip-level o finestre stitched) -----------------------------
class DualFeaturesClipDataset(Dataset):
    """
    Output:
      - return_tech & return_quality:  (A[T,Fa], L[T,Fl], t_valid, tech_id, q), y
      - return_tech:                   (A[T,Fa], L[T,Fl], t_valid, tech_id), y
      - return_quality:                (A[T,Fa], L[T,Fl], t_valid, q), y
      - altrimenti:                    (A[T,Fa], L[T,Fl], t_valid), y
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        clip_dirs: Optional[Iterable[str]] = None,
        stitch_k: int = 1,          # quante clip consecutive unire (1=nessuna)
        T: int = 8,
        return_ids: bool = False,
        real_markers: Tuple[str, ...] = ("webcam","original","pristine","authentic","real","youtube-real","celeb-real"),
        label_override: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        validate: bool = True,
        au_dim: Optional[int] = None,
        lmk_dim: Optional[int] = None,
        mmap: bool = True,
        is_train: bool = False,
        random_crop: bool = False,
        zscore: str = "none",   # "none" | "clip" | "global"
        zscore_apply: str = "both",  # "both" | "au" | "lmk"
        eps: float = 1e-6,
        return_tech: bool = False,
        norm_stats_path: Optional[str] = None,
        aug_noise_au: float = 0.0,
        aug_noise_lmk: float = 0.0,
        aug_tdrop: float = 0.0,
        # I/O robuste
        safe_mmap: bool = True,
        skip_on_error: bool = True,
        eject_broken: bool = False,
        max_warn: int = 50,
        # qualità e degradazioni
        return_quality: bool = False,
        qual_factorized: bool = False,
        dirty_p: float = 0.0,
        clean_fake_p: float = 0.2,
        clean_real_p: float = 0.3,
        protect_real_for_consistency: bool = True,
        # LMK
        lmk_affine_deg: float = 2.0,
        lmk_dropout_p: float = 0.0,
        lmk_temporal_alpha: float = 0.0,
        # AU
        au_dropout_p: float = 0.0,
        au_temporal_alpha: float = 0.0,
        # Dinamica LMK opzionale
        lmk_add_deltas: bool = False,  # se True concatena Δ e Δ² a L
        allow_missing_au: bool=False, **kwargs):
        super().__init__()
        self.allow_missing_au = allow_missing_au
        self.T = int(T)
        self.dtype = dtype
        self.mmap = bool(mmap)
        self.safe_mmap = bool(safe_mmap)
        self.skip_on_error = bool(skip_on_error)
        self.eject_broken = bool(eject_broken)
        self._warn_count = 0
        self._max_warn = int(max_warn)
        self.stitch_k = max(1, int(stitch_k))
        self.lmk_add_deltas = bool(lmk_add_deltas)

        self.is_train = bool(is_train)
        self.random_crop = bool(random_crop)
        self.zscore = str(zscore).lower()
        za = str(zscore_apply).lower()
        if za not in ("both","au","lmk"):
            raise ValueError("zscore_apply must be 'both', 'au', or 'lmk'")
        self._z_au  = (za in ("both","au"))
        self._z_lmk = (za in ("both","lmk"))

        self.eps = float(eps)
        self.return_tech = bool(return_tech)
        self.return_quality = bool(return_quality)
        self.protect_real_for_consistency = bool(protect_real_for_consistency)

        self.aug_noise_au = float(aug_noise_au)
        self.aug_noise_lmk = float(aug_noise_lmk)
        self.aug_tdrop = float(aug_tdrop)

        self.qual_factorized = bool(qual_factorized)
        self.dirty_p = float(dirty_p)
        self.clean_fake_p = float(clean_fake_p)
        self.clean_real_p = float(clean_real_p)

        self.lmk_affine_deg = float(lmk_affine_deg)
        self.lmk_dropout_p = float(lmk_dropout_p)
        self.lmk_temporal_alpha = float(lmk_temporal_alpha)
        self.au_dropout_p = float(au_dropout_p)
        self.au_temporal_alpha = float(au_temporal_alpha)
        self.return_ids = bool(return_ids)

        self.real_markers = tuple(m.lower() for m in real_markers)
        

        # --- raccolta clip base (singole) ---
        base_samples: List[Tuple[str, str, int, str]] = []  # (au_path, lmk_path, label, clip_dir)

        if clip_dirs is not None:
            for d in list(clip_dirs):
                au = os.path.join(d, "au_features.npy")
                lm = os.path.join(d, "lmk_features.npy")
                if validate:
                    has_lmk = os.path.isfile(lm)
                    has_au  = os.path.isfile(au)
                    if not has_lmk: 
                        continue
                    if not self.allow_missing_au and not has_au:
                        continue
                lab = int(label_override) if label_override is not None else self._label_from_dir(d)
                base_samples.append((au, lm, lab, d))
        elif root_dir is not None:
            roots = root_dir if isinstance(root_dir, (list, tuple)) else [root_dir]
            for base in roots:
                for root, _, files in os.walk(base):
                    has_lmk = "lmk_features.npy" in files
                    has_au  = "au_features.npy" in files
                    if not has_lmk:
                        continue
                    if not self.allow_missing_au and not has_au:
                        continue
                    au = os.path.join(root, "au_features.npy")  # può non esistere: _safe_load gestisce
                    lm = os.path.join(root, "lmk_features.npy")
                    lab = int(label_override) if label_override is not None else self._label_from_dir(root)
                    base_samples.append((au, lm, lab, root))
        else:
            raise ValueError("Passa 'root_dir' o 'clip_dirs'.")

        if not base_samples:
            raise RuntimeError("Nessuna clip trovata.")

        # opzionale: filtro clip illeggibili in anticipo
        if self.eject_broken:
            ok = []
            for au, lm, lab, d in base_samples:
                if self._is_loadable_pair(au, lm):
                    ok.append((au, lm, lab, d))
            dropped = len(base_samples) - len(ok)
            if dropped > 0: self._warn_once(f"Eject di {dropped} clip corrotte.")
            base_samples = ok
            if not base_samples: raise RuntimeError("Tutte le clip risultano illeggibili dopo il filtro.")

        def _clip_is_valid(self, clip_dir):
            has_lmk = os.path.isfile(os.path.join(clip_dir, "lmk_features.npy"))
            if self.allow_missing_au:
                return has_lmk
            has_au  = os.path.isfile(os.path.join(clip_dir, "au_features.npy"))
            return has_lmk and has_au

        # --- inferisci dimensioni feature (robusto) ---
        def _peek_dims_multi(max_try=1024):
            for au_p, lm_p, _, _ in base_samples[:max_try]:
                try:
                    A = np.asarray(np.load(au_p, mmap_mode="r" if self.mmap else None))
                    L = np.asarray(np.load(lm_p, mmap_mode="r" if self.mmap else None))
                    if A.ndim == 2 and L.ndim == 2 and A.shape[1] > 0 and L.shape[1] > 0:
                        return int(A.shape[1]), int(L.shape[1])
                except Exception:
                    continue
            self._warn_once("Impossibile dedurre le dimensioni; uso fallback (36/132).")
            return 36, 132

        if au_dim is not None and lmk_dim is not None:
            self.au_dim, self.lmk_dim = int(au_dim), int(lmk_dim)
        else:
            self.au_dim, self.lmk_dim = _peek_dims_multi()

        # --- normalizzazione globale ---
        self.norm_stats = None
        if self.zscore == "global":
            stats_path = norm_stats_path or os.getenv("NORM_STATS", "").strip()
            if stats_path:
                try:
                    S = np.load(stats_path)
                    au_mean = S["au_mean"].astype(np.float32)
                    au_std  = np.maximum(S["au_std"].astype(np.float32), self.eps)
                    lmk_mean= S["lmk_mean"].astype(np.float32)
                    lmk_std = np.maximum(S["lmk_std"].astype(np.float32), self.eps)
                    if au_mean.shape[0] != self.au_dim or au_std.shape[0] != self.au_dim: raise ValueError("AU stats shape mismatch.")
                    if lmk_mean.shape[0] != self.lmk_dim or lmk_std.shape[0] != self.lmk_dim: raise ValueError("LMK stats shape mismatch.")
                    self.norm_stats = {"au_mean":au_mean,"au_std":au_std,"lmk_mean":lmk_mean,"lmk_std":lmk_std}
                    LOG.info(f"Loaded global norm stats from {stats_path}")
                except Exception as e:
                    self._warn_once(f"Could not load global norm stats: {e}.")
            else:
                self._warn_once("zscore='global' richiesto ma nessun percorso stats fornito.")

        # --- costruzione items: stitching per track oppure singole clip ---
        # items: lista di voci; ogni voce è:
        #   - ("single", (au,lm,label,dir))
        #   - ("window", [(au,lm,label,dir), ...] di lunghezza stitch_k)
        self.items: List[Tuple[str, Union[Tuple[str,str,int,str], List[Tuple[str,str,int,str]]]]] = []


        if self.stitch_k <= 1:
            for s in base_samples:
                self.items.append(("single", s))
        else:
            # raggruppa per track_id e ordina per indice clip
            buckets: Dict[str, List[Tuple[str,str,int,str,int]]] = {}
            for au, lm, lab, d in base_samples:
                segs = d.replace("\\","/").split("/")
                track = next((s for s in segs if s.startswith("track_")), None)
                clip_s = next((x for x in segs if x.startswith("clip_")), None)
                if track is None or clip_s is None:  # salta se non compatibile
                    continue
                try:
                    clip_idx = int(clip_s.split("_")[-1])
                except Exception:
                    clip_idx = -1
                key = "/".join(segs[:segs.index(track)+1])  # .../track_X
                buckets.setdefault(key, []).append((au, lm, lab, d, clip_idx))
            for key in list(buckets.keys()):
                buckets[key].sort(key=lambda r: r[-1])

            for _, lst in buckets.items():
                if len(lst) < self.stitch_k: continue
                # finestre scorrevoli
                for i in range(0, len(lst) - self.stitch_k + 1):
                    win_raw = lst[i:i+self.stitch_k]
                    win = [(au,lm,lab,d) for (au,lm,lab,d,_) in win_raw]
                    # etichetta: usa la del primo elemento (i dataset sono coerenti a livello track)
                    self.items.append(("window", win))

        if not self.items:
            raise RuntimeError("Nessun sample disponibile dopo stitching/filtri.")

        # ------- metadati per domini/tecniche allineati a items -------

        self.tech_names, self.labels = [], []
        for kind, payload in self.items:
            if kind == "single":
                _, _, lab, d = payload
            else:
                _, _, lab, d = payload[0]
            self.labels.append(int(lab))
            self.tech_names.append(_infer_tech_from_path(d))

        fake_techs = sorted({t for t, y in zip(self.tech_names, self.labels) if y == 1 and t != "unknown"})
        self.domain_map = {t: i+1 for i, t in enumerate(fake_techs)}
        self.n_domains  = 1 + len(fake_techs)

        # --- PERSONA/VIDEO IDs ---
        import re
        def _ids_from_dir(d: str):
            p = d.replace("\\","/").split("/")
            track = next((s for s in p if re.match(r"track_\d+$", s)), None)
            if not track:
                return ("unknown/unknown", "unknown/unknown/track_0")
            i = p.index(track)
            tech = p[i-2] if i >= 2 else "unknown"
            vid  = p[i-1] if i >= 1 else "unknown"
            return (f"{tech}/{vid}", f"{tech}/{vid}/{track}")

        self.vid_keys, self.track_keys = [], []
        for kind, payload in self.items:
            d = payload[3] if kind=="single" else payload[0][3]
            vk, tk = _ids_from_dir(d)
            self.vid_keys.append(vk); self.track_keys.append(tk)

        uniq_vid = {k:i for i,k in enumerate(sorted(set(self.vid_keys)))}
        uniq_trk = {k:i for i,k in enumerate(sorted(set(self.track_keys)))}
        self.vid_ids = np.array([uniq_vid[k] for k in self.vid_keys], dtype=np.int64)
        self.trk_ids = np.array([uniq_trk[k] for k in self.track_keys], dtype=np.int64)


        LOG.info(f"{self.__class__.__name__}(N={len(self)}, T={self.T}, au_dim={self.au_dim}, lmk_dim={self.lmk_dim}, "
                 f"mmap={self.mmap}, is_train={self.is_train}, random_crop={self.random_crop}, zscore='{self.zscore}', "
                 f"return_tech={self.return_tech}, n_domains={self.n_domains}, aug_noise_au={self.aug_noise_au}, "
                 f"aug_noise_lmk={self.aug_noise_lmk}, aug_tdrop={self.aug_tdrop}, safe_mmap={self.safe_mmap}, "
                 f"skip_on_error={self.skip_on_error}, eject_broken={self.eject_broken}, "
                 f"return_quality={self.return_quality}, qual_factorized={self.qual_factorized}, "
                 f"stitch_k={self.stitch_k}, lmk_add_deltas={self.lmk_add_deltas})")

    # ----------------------------- utils/loader -----------------------------
    def __len__(self) -> int: return len(self.items)
    @staticmethod
    def _path_like(p: str) -> str: return p.lower().replace("\\", "/")

    def _label_from_dir(self, d: str) -> int:
        p = self._path_like(d)
        tokens = [seg for seg in p.split("/") if seg]
        real_tokens = {"youtube-real","celeb-real","original","pristine","authentic","real"}
        return 0 if any(t in real_tokens for t in tokens) else 1


    @staticmethod
    def _as_writable_c_contiguous(X: np.ndarray, dtype=np.float32) -> np.ndarray:
        return np.array(X, dtype=dtype, copy=True, order="C")

    def _select_window(self, X: np.ndarray, T: int) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32); t = int(X.shape[0])
        if t > T:
            if self.is_train and self.random_crop:
                start = np.random.randint(0, t - T + 1); out = X[start:start+T]
            else:
                out = X[:T]
        else:
            out = X
        return self._as_writable_c_contiguous(out, dtype=np.float32)

    @staticmethod
    def _pad_to_len(X: np.ndarray, T: int) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32); t, f = int(X.shape[0]), int(X.shape[1])
        if t >= T: return np.array(X[:T], dtype=np.float32, copy=True, order="C")
        pad = np.zeros((T - t, f), dtype=np.float32); out = np.concatenate([X, pad], axis=0)
        return np.array(out, dtype=np.float32, copy=True, order="C")

    def _zscore_clip(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0: return self._as_writable_c_contiguous(X)
        mu = X.mean(axis=0, keepdims=True); sd = np.maximum(X.std(axis=0, keepdims=True), self.eps)
        return self._as_writable_c_contiguous((X - mu) / sd)

    def _zscore_global(self, X: np.ndarray, kind: str) -> np.ndarray:
        if X.size == 0 or self.norm_stats is None: return self._as_writable_c_contiguous(X)
        mu = self.norm_stats[f"{kind}_mean"][None,:]; sd = self.norm_stats[f"{kind}_std"][None,:]
        return self._as_writable_c_contiguous((X - mu) / sd)

    def _warn_once(self, msg: str):
        if self._warn_count < self._max_warn:
            LOG.warning(msg); self._warn_count += 1
        elif self._warn_count == self._max_warn:
            LOG.warning("Altri warning soppressi..."); self._warn_count += 1

    def _safe_load(self, path: str) -> Optional[np.ndarray]:
        try:
            return np.load(path, mmap_mode="r" if self.mmap else None)
        except Exception as e:
            if self.mmap and self.safe_mmap:
                try: return np.load(path, mmap_mode=None)
                except Exception as e2:
                    self._warn_once(f"Load fallito per {path} (no mmap): {e2}"); return None
            else:
                self._warn_once(f"Load fallito per {path}: {e}"); return None

    def _fix_feat_dim(self, X: np.ndarray, want_dim: int) -> np.ndarray:
        if X.ndim != 2: return np.zeros((0, want_dim), np.float32)
        Tcur, Fcur = int(X.shape[0]), int(X.shape[1])
        if Fcur == want_dim: return X.astype(np.float32, copy=False)
        if Fcur > want_dim:  return X[:, :want_dim].astype(np.float32, copy=False)
        pad = np.zeros((Tcur, want_dim - Fcur), dtype=np.float32)
        return np.concatenate([X.astype(np.float32, copy=False), pad], axis=1)

    def _is_loadable_pair(self, au_path: str, lm_path: str) -> bool:
        A = self._safe_load(au_path); L = self._safe_load(lm_path)
        return (A is not None) and (L is not None)

    # ---------------------- degradazioni feature-space ----------------------
    @staticmethod
    def _ema_time_np(X: np.ndarray, alpha: float) -> np.ndarray:
        if alpha <= 0.0 or X.shape[0] <= 1: return X
        Y = X.copy()
        for t in range(1, X.shape[0]): Y[t] = alpha * Y[t-1] + (1.0 - alpha) * X[t]
        return Y
    
        # --- compat: elenco clip_dirs “primari” per item ---
    @property
    def clip_dirs(self):
        dirs = []
        for kind, payload in self.items:
            if kind == "single":
                _, _, _, d = payload  # (au,lm,label,dir)
                dirs.append(d)
            else:
                # usa la prima clip della finestra come rappresentante
                d = payload[0][3]
                dirs.append(d)
        return dirs


    def _deg_lmk_np(self, L: np.ndarray) -> np.ndarray:
        if L.size == 0: return L
        T_, Fl = L.shape
        if Fl % 2 != 0:
            X = L.copy()
        else:
            P = Fl // 2; X = L.reshape(T_, P, 2).copy()
            if self.lmk_affine_deg > 0.0:
                th = math.radians(np.random.uniform(-self.lmk_affine_deg, self.lmk_affine_deg))
                c, s = math.cos(th), math.sin(th); R = np.array([[c, -s], [s, c]], dtype=np.float32)
                scale = 1.0 + np.random.uniform(-0.02, 0.02)
                t = np.array([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01)], dtype=np.float32)
                X = (X @ (R * scale)) + t
            if self.aug_noise_lmk > 0.0:
                X = X + np.random.randn(*X.shape).astype(np.float32) * float(self.aug_noise_lmk)
            if self.lmk_dropout_p > 0.0:
                mask = (np.random.rand(X.shape[1]) > self.lmk_dropout_p).astype(np.float32)
                X = X * mask[None, :, None]
            X = X.reshape(T_, Fl)
        if self.lmk_temporal_alpha > 0.0:
            X = self._ema_time_np(X, float(self.lmk_temporal_alpha))
        return X.astype(np.float32, copy=False)

    def _deg_au_np(self, A: np.ndarray) -> np.ndarray:
        if A.size == 0: return A
        X = A.copy()
        if self.aug_noise_au > 0.0:
            X = X + np.random.randn(*X.shape).astype(np.float32) * float(self.aug_noise_au)
        if self.au_dropout_p > 0.0:
            mask = (np.random.rand(*X.shape) > self.au_dropout_p).astype(np.float32); X = X * mask
        if self.au_temporal_alpha > 0.0:
            X = self._ema_time_np(X, float(self.au_temporal_alpha))
        return X.astype(np.float32, copy=False)

    # ----------------------------- helpers per caricamento singola clip -----------------------------
    def _load_single_clip(self, au_path: str, lm_path: str):
        A_np = self._safe_load(au_path)
        L_np = self._safe_load(lm_path)

        if L_np is None:
            # senza LMK non possiamo usare la clip
            self._warn_once(f"[SKIP] LMK assenti/corrotti: {lm_path}")
            return np.zeros((0, self.au_dim), np.float32), np.zeros((0, self.lmk_dim), np.float32), 0

        L_np = np.asarray(L_np, np.float32)

        if A_np is None:
            if self.allow_missing_au:
                # AU mancanti: usa zeri ma lunga quanto i LMK
                A_np = np.zeros((int(L_np.shape[0]), self.au_dim), np.float32)
            else:
                self._warn_once(f"[SKIP] AU assenti/corrotti: {au_path}")
                return np.zeros((0, self.au_dim), np.float32), np.zeros((0, self.lmk_dim), np.float32), 0
        else:
            A_np = np.asarray(A_np, np.float32)

        # fix dimensioni se servono
        if A_np.ndim == 2 and A_np.shape[1] != self.au_dim:  A_np = self._fix_feat_dim(A_np, self.au_dim)
        if L_np.ndim == 2 and L_np.shape[1] != self.lmk_dim: L_np = self._fix_feat_dim(L_np, self.lmk_dim)

        # allinea per sicurezza
        t_raw = min(int(A_np.shape[0]), int(L_np.shape[0]))
        if t_raw > 0:
            A_np, L_np = A_np[:t_raw], L_np[:t_raw]
        else:
            A_np = np.zeros((0, self.au_dim), np.float32)
            L_np = np.zeros((0, self.lmk_dim), np.float32)
        return A_np, L_np, t_raw


    # ----------------------------- __getitem__ -----------------------------
    def __getitem__(self, idx: int):
        if isinstance(idx, np.generic): idx = int(idx)
        elif isinstance(idx, np.ndarray): idx = int(idx.ravel()[0]) if idx.ndim>0 else int(idx)
        elif isinstance(idx, torch.Tensor) and idx.numel()==1: idx = int(idx.item())

        kind, payload = self.items[idx]
        if kind == "single":
            au_path, lm_path, label_int, _dir = payload
            A_np, L_np, t_raw = self._load_single_clip(au_path, lm_path)
        else:
            win = payload
            label_int = int(win[0][2]) if len(win) else 0
            A_list, L_list = [], []
            for au_path, lm_path, _, _ in win:
                Ai, Li, ti = self._load_single_clip(au_path, lm_path)
                tseg = int(min(ti, Ai.shape[0], Li.shape[0]))
                if tseg>0:
                    A_list.append(Ai[:tseg]); L_list.append(Li[:tseg])
            A_np = np.concatenate(A_list, axis=0) if A_list else np.zeros((0,self.au_dim),np.float32)
            L_np = np.concatenate(L_list, axis=0) if L_list else np.zeros((0,self.lmk_dim),np.float32)
            t_raw = int(min(A_np.shape[0], L_np.shape[0]))

        # finestra sincronizzata e pad
        t_raw = int(min(t_raw, A_np.shape[0], L_np.shape[0]))
        if t_raw>0:
            start = np.random.randint(0, t_raw - self.T + 1) if (self.is_train and self.random_crop and t_raw>self.T) else 0
            A_win = A_np[start:start+self.T]; L_win = L_np[start:start+self.T]
            t_valid = int(min(self.T, A_win.shape[0], L_win.shape[0]))
            if t_valid < self.T:
                if t_valid>0:
                    pad = self.T - t_valid
                    A_win = np.pad(A_win, ((0,pad),(0,0)), mode="edge")
                    L_win = np.pad(L_win, ((0,pad),(0,0)), mode="edge")
                else:
                    A_win = np.zeros((self.T,self.au_dim),np.float32)
                    L_win = np.zeros((self.T,self.lmk_dim),np.float32)
        else:
            A_win = np.zeros((self.T,self.au_dim),np.float32)
            L_win = np.zeros((self.T,self.lmk_dim),np.float32)
            t_valid = 0

        # copia reale, C-contiguous, sempre scrivibile
        A_win = np.array(A_win, dtype=np.float32, copy=True, order="C")
        L_win = np.array(L_win, dtype=np.float32, copy=True, order="C")



        # z-score
        if t_valid>0:
            if self.zscore=="clip":
                if self._z_au:  A_win[:t_valid] = self._zscore_clip(A_win[:t_valid])
                if self._z_lmk: L_win[:t_valid] = self._zscore_clip(L_win[:t_valid])
            elif self.zscore=="global":
                if self._z_au:  A_win[:t_valid] = self._zscore_global(A_win[:t_valid], "au")
                if self._z_lmk: L_win[:t_valid] = self._zscore_global(L_win[:t_valid], "lmk")
            A_win[:t_valid] = np.nan_to_num(A_win[:t_valid], nan=0.0, posinf=0.0, neginf=0.0)
            L_win[:t_valid] = np.nan_to_num(L_win[:t_valid], nan=0.0, posinf=0.0, neginf=0.0)

        # qualità/aug
        q = 0
        if self.qual_factorized:
            q = 1 if np.random.rand() < getattr(self,"dirty_p",0.0) else 0
        else:
            cf = float(getattr(self,"clean_fake_p",1.0))
            cr = float(getattr(self,"clean_real_p",1.0))
            q = 0 if (int(label_int)==1 and np.random.rand()<cf) or (int(label_int)==0 and np.random.rand()<cr) else 1
        if self.is_train and q==1 and t_valid>0:
            L_win[:t_valid] = self._deg_lmk_np(L_win[:t_valid])
            A_win[:t_valid] = self._deg_au_np(A_win[:t_valid])

        if self.is_train and getattr(self,"aug_tdrop",0.0)>0.0 and t_valid>2:
            protect_real = bool(getattr(self,"protect_real_for_consistency",False))
            if not (protect_real and int(label_int)==0):
                k = max(0, min(int(round(self.aug_tdrop*t_valid)), t_valid-2))
                if k>0:
                    idx_pool = np.arange(1,t_valid)
                    drop_idx = np.random.choice(idx_pool, size=k, replace=False)
                    m = np.ones((t_valid,1), dtype=A_win.dtype); m[drop_idx]=0.0
                    A_win[:t_valid] *= m; L_win[:t_valid] *= m

        if self.lmk_add_deltas and L_win.shape[0]>0:
            d1 = np.diff(L_win, axis=0, prepend=L_win[:1])
            d2 = np.diff(d1,   axis=0, prepend=d1[:1])
            L_win = np.concatenate([L_win,d1,d2], axis=1).astype(np.float32, copy=False)

        A_fix = self._pad_to_len(A_win, self.T)
        L_fix = self._pad_to_len(L_win, self.T)
        Fwant = int(getattr(self,"lmk_dim_out", self.lmk_dim))
        assert L_fix.shape[1] == Fwant, f"{L_fix.shape[1]} != {Fwant}"

        A = torch.from_numpy(A_fix).to(dtype=self.dtype)
        L = torch.from_numpy(L_fix).to(dtype=self.dtype)
        y = torch.tensor(int(label_int), dtype=torch.long)
        t_valid_tensor = torch.tensor(int(min(t_valid,self.T)), dtype=torch.long)
        q_tensor = torch.tensor(int(q), dtype=torch.long)

        tech = self.tech_names[idx]
        dom_id_int = 0 if int(label_int)==0 else 1 + int(self.domain_map.get(tech,0))
        dom_id = torch.tensor(dom_id_int, dtype=torch.long)

        trk_id = torch.tensor(int(self.trk_ids[idx]), dtype=torch.long)
        vid_id = torch.tensor(int(self.vid_ids[idx]), dtype=torch.long)

        out = [A, L, t_valid_tensor]
        if self.return_tech: out.append(dom_id)
        if self.return_quality: out.append(q_tensor)
        out.extend([trk_id, vid_id])
        return tuple(out), y
