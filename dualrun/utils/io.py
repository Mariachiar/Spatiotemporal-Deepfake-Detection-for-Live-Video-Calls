import os, json, logging, random
from typing import Any, Optional
from data.dataset_dual import DualFeaturesClipDataset

LOG = logging.getLogger("dualrun.io")

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)
    LOG.info(f"Saved: {path}")

def _valid_index_schema(data: Any) -> bool:
    return isinstance(data, dict) and all(k in data for k in ("train","val","test"))

def load_index_json(path: Optional[str]) -> Optional[dict]:
    if path is None: return None
    try:
        with open(path, "r") as f: data = json.load(f)
        if _valid_index_schema(data):
            LOG.info(f"Loaded index: {path}"); return data
        LOG.warning(f"Index JSON schema not recognized: {path}")
        return None
    except Exception as e:
        LOG.warning(f"Failed to read index '{path}': {e}")
        return None

def ensure_index_json(path: str, root_dir: str, T: int, **ds_kwargs) -> dict:
    data = load_index_json(path)
    if _valid_index_schema(data): return data  # type: ignore
    LOG.warning("Index missing/invalid; generating a new one (80/10/10) from --data.")
    full = DualFeaturesClipDataset(root_dir=root_dir, T=T, **ds_kwargs)
    N = len(full); idx = list(range(N)); rng = random.Random(42); rng.shuffle(idx)
    n_train = int(0.8*N); n_val = int(0.1*N)
    index = {
        "train": [full.clip_dirs[i] for i in idx[:n_train]],
        "val"  : [full.clip_dirs[i] for i in idx[n_train:n_train+n_val]],
        "test" : [full.clip_dirs[i] for i in idx[n_train+n_val:]],
    }
    try: save_json(index, path)
    except Exception as e: LOG.warning(f"Could not save generated index: {e}")
    return index
