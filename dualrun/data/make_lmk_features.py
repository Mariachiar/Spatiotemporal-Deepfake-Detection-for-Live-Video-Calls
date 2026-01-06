#!/usr/bin/env python3
import os, argparse, glob,cv2, numpy as np, logging
from tqdm import tqdm
from math import atan2, cos, sin
try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError("extract_lmk_seq richiede mediapipe") from e

LOG = logging.getLogger("make_lmk")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
os.environ.setdefault("GLOG_minloglevel","2")
os.environ.setdefault("ABSL_LOG_SEVERITY","error")
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass

_FACE_MESH = None
_FACE_MESH_PID = None

def _get_facemesh():
    global _FACE_MESH, _FACE_MESH_PID
    pid = os.getpid()
    if (_FACE_MESH is None) or (_FACE_MESH_PID != pid):
        import mediapipe as mp
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True
        )  # niente "with": persiste nel processo
        _FACE_MESH_PID = pid
    return _FACE_MESH



# ---- 66 keypoints + 3 di riferimento (MediaPipe FaceMesh) ----
KEY_LANDMARKS_IDXS = [
    # occhio sx + sopracciglio
    33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,
    70,63,105,66,107,55,65,52,53,46,
    # occhio dx + sopracciglio
    263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466,
    300,293,334,296,336,285,295,282,283,276,
    # labbra esterne
    61,146,91,181,84,17,314,405,321,375,291,
    # riferimenti
    1, 78, 308,
]
NOSE_TIP_IDX, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX = 1, 78, 308
REQ_MIN_LANDMARKS = max(KEY_LANDMARKS_IDXS + [NOSE_TIP_IDX, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX]) + 1  # 309

# ---------- parser frame → (N>=309, 2) ----------
def _ndarray1d_to_xy(frame1d):
    """Converte un ndarray 1D (len>=309) dove ciascun elem è dict/.x,.y/seq in (N,2)."""
    if frame1d.shape[0] < REQ_MIN_LANDMARKS:
        return None
    f0 = frame1d[0]
    # array di dict {'x','y'}
    if isinstance(f0, dict) and "x" in f0 and "y" in f0:
        try:
            xy = np.array([[float(lm["x"]), float(lm["y"])] for lm in frame1d], dtype=np.float32)
            return xy
        except Exception:
            return None
    # array di oggetti con .x/.y
    if hasattr(f0, "x") and hasattr(f0, "y"):
        try:
            xy = np.array([[float(lm.x), float(lm.y)] for lm in frame1d], dtype=np.float32)
            return xy
        except Exception:
            return None
    # array di tuple/list/ndarray
    if isinstance(f0, (list, tuple, np.ndarray)) and len(f0) >= 2:
        try:
            arr = np.asarray(frame1d, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
        except Exception:
            return None
    return None

def _frame_to_xy_array(frame):
    """
    Converte un frame in array float32 (N>=309,2).
    Supporta:
      - list[dict{'x','y'(,'z')}], list[(x,y[,z])], list[np.ndarray]
      - np.ndarray 2D (N,2|3)
      - np.ndarray 1D (N,) di oggetti (dict/.x,.y/seq)
      - dict{'x': array(N,), 'y': array(N,)}
      - oggetti con attributi .x/.y
    """
    if frame is None:
        return None

    # --- ndarray ---
    if isinstance(frame, np.ndarray):
        # 2D numerico
        if frame.ndim == 2 and frame.shape[1] >= 2 and frame.shape[0] >= REQ_MIN_LANDMARKS:
            return frame[:, :2].astype(np.float32, copy=False)
        # 1D di oggetti (il tuo caso: shape (478,))
        if frame.ndim == 1:
            return _ndarray1d_to_xy(frame)
        return None

    # --- dict di vettori 'x','y' ---
    if isinstance(frame, dict) and "x" in frame and "y" in frame:
        x, y = np.asarray(frame["x"]), np.asarray(frame["y"])
        if x.ndim == y.ndim == 1 and x.shape[0] == y.shape[0] >= REQ_MIN_LANDMARKS:
            return np.stack([x, y], axis=1).astype(np.float32, copy=False)
        return None

    # --- lista ---
    if isinstance(frame, list):
        if len(frame) < REQ_MIN_LANDMARKS:
            return None
        f0 = frame[0]
        # list[dict]
        if isinstance(f0, dict) and "x" in f0 and "y" in f0:
            try:
                xy = np.array([[float(lm["x"]), float(lm["y"])] for lm in frame], dtype=np.float32)
                return xy
            except Exception:
                return None
        # list[oggetti con .x/.y]
        if hasattr(f0, "x") and hasattr(f0, "y"):
            try:
                xy = np.array([[float(lm.x), float(lm.y)] for lm in frame], dtype=np.float32)
                return xy
            except Exception:
                return None
        # list[tuple/list/ndarray]
        if isinstance(f0, (list, tuple, np.ndarray)) and len(f0) >= 2:
            try:
                arr = np.asarray(frame, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return arr[:, :2]
            except Exception:
                return None

    return None

# ---------- normalizzazione ----------
def _rotate_to_mouth(points, mouth_left, mouth_right):
    dx, dy = (mouth_right[0]-mouth_left[0]), (mouth_right[1]-mouth_left[1])
    theta = atan2(dy, dx)
    c, s = cos(-theta), sin(-theta)
    R = np.array([[c, -s],[s, c]], dtype=np.float32)
    return (points @ R.T).astype(np.float32, copy=False)

def _frame_to_features(frame, rot_invariant: bool, dbg: dict) -> np.ndarray | None:
    xy = _frame_to_xy_array(frame)
    if xy is None:
        dbg["bad_format"] += 1
        return None
    if xy.shape[0] < REQ_MIN_LANDMARKS:
        dbg["too_few_points"] += 1
        return None

    points = xy[KEY_LANDMARKS_IDXS, :]             # (K,2)
    nose   = xy[NOSE_TIP_IDX, :]
    ml     = xy[MOUTH_LEFT_IDX, :]
    mr     = xy[MOUTH_RIGHT_IDX, :]

    centered = points - nose
    scale = np.linalg.norm(ml - mr)
    if not np.isfinite(scale) or scale < 1e-8:
        dbg["bad_scale"] += 1
        return None
    normed = centered / (scale + 1e-6)

    if rot_invariant:
        ml_c = (ml - nose) / (scale + 1e-6)
        mr_c = (mr - nose) / (scale + 1e-6)
        normed = _rotate_to_mouth(normed, ml_c, mr_c)

    return normed.reshape(-1).astype(np.float32, copy=False)

def _seq_to_features(seq_landmarks, rot_invariant: bool):
    dbg = {"bad_format":0, "too_few_points":0, "bad_scale":0}
    feats = []
    for frame in seq_landmarks:
        v = _frame_to_features(frame, rot_invariant, dbg)
        if v is not None:
            feats.append(v)
    return (np.stack(feats, axis=0) if feats else np.zeros((0, len(KEY_LANDMARKS_IDXS)*2), np.float32)), dbg

# ---------- core ----------
def process_tree(base_dir, overwrite=False, min_frames=1, rot_invariant=False):
    base_dir = os.path.abspath(base_dir)
    dims = len(KEY_LANDMARKS_IDXS) * 2
    LOG.info(f"Base dataset: {base_dir}")
    LOG.info(f"Feature set: coordinate normalizzate ({dims} dims){' + rot-invariant' if rot_invariant else ''}")

    clips = glob.glob(os.path.join(base_dir, "**", "track_*", "clip_*"), recursive=True)
    if not clips:
        raise SystemExit(f"Nessuna clip trovata in {base_dir}")
    LOG.info(f"Clip trovate: {len(clips)}")

    n_ok = n_skip = n_err = n_exist = 0
    pbar = tqdm(clips, desc="LMK→features", unit="clip")
    for clip in pbar:
        try:
            lmk_path = os.path.join(clip, "landmarks.npy")
            out_path = os.path.join(clip, "lmk_features.npy")

            if not os.path.isfile(lmk_path):
                n_skip += 1
                pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err, exist=n_exist)
                continue

            if os.path.isfile(out_path) and not overwrite:
                n_exist += 1
                pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err, exist=n_exist)
                continue

            # Carica senza .tolist()
            seq = np.load(lmk_path, allow_pickle=True)

            # Filtra None o liste vuote []
            if isinstance(seq, np.ndarray):
                frames = [f for f in seq if f is not None and not (isinstance(f, list) and len(f) == 0)]
            else:
                frames = [f for f in list(seq) if f is not None and not (isinstance(f, list) and len(f) == 0)]

            F, dbg = _seq_to_features(frames, rot_invariant=rot_invariant)
            if F.shape[0] < min_frames:
                LOG.warning(f"SKIP (frame validi {F.shape[0]} < min_frames={min_frames}) "
                            f"| bad_format={dbg['bad_format']} too_few_points={dbg['too_few_points']} bad_scale={dbg['bad_scale']}: {clip}")
                n_skip += 1
                pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err, exist=n_exist)
                continue

            np.save(out_path, F.astype(np.float32, copy=False))
            n_ok += 1
            pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err, exist=n_exist)

        except Exception as e:
            LOG.error(f"Errore su clip: {clip} | {e}", exc_info=True)
            n_err += 1
            pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err, exist=n_exist)

    LOG.info(f"✅ Completato. OK={n_ok} | EXIST={n_exist} | SKIP={n_skip} | ERR={n_err}")

# ---------- CLI ----------
def setup_logging(level: str):
    level = level.upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    LOG.info(f"Logging inizializzato: {level}")


# ===== API per training on-the-fly =====
def extract_lmk_seq(frames):
    """
    frames: List[np.ndarray HxWx3 RGB]
    ritorna: (T, 132) float32  [= 66 punti * 2]
    Richiede mediapipe; installa se assente: pip install mediapipe
    """
    fm = _get_facemesh()
    out = []
    for im in frames:
        res = fm.process(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        if not res.multi_face_landmarks:
                # frame vuoto → vettore zero
            out.append(np.zeros((len(KEY_LANDMARKS_IDXS)*2,), np.float32))
            continue
        lmks = res.multi_face_landmarks[0].landmark
            # costruisci array (N>=309, 2)
        arr = np.array([[lm.x, lm.y] for lm in lmks], dtype=np.float32)
        v, _dbg = _seq_to_features([arr], rot_invariant=False)  # una sola frame → (1,132) o (0,132)
        if v.shape[0] == 0:
            out.append(np.zeros((len(KEY_LANDMARKS_IDXS)*2,), np.float32))
        else:
            out.append(v[0])
    return np.stack(out, axis=0).astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser(description="Estrae feature LMK robuste dai landmarks.npy delle clip.")
    ap.add_argument("--base", required=True, help="Root del dataset (es. ./datasets/processed_dataset)")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--overwrite", action="store_true", help="Rigenera lmk_features.npy anche se esiste.")
    ap.add_argument("--min-frames", type=int, default=1, help="Minimo di frame validi richiesti per salvare le feature.")
    ap.add_argument("--rot-invariant", action="store_true", help="Allinea orizzontalmente alla linea della bocca.")
    args = ap.parse_args()

    setup_logging(args.log_level)
    process_tree(args.base, overwrite=args.overwrite, min_frames=args.min_frames, rot_invariant=args.rot_invariant)

if __name__ == "__main__":
    main()