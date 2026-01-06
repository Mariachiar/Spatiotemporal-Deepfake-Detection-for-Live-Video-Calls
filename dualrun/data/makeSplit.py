#!/usr/bin/env python3
# makeSplit_v2.py — split per-video con tracking di track e clip complete
import argparse, json, random, sys, datetime
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

# -------------------------------
# Costanti dataset
# -------------------------------
TRACK_PREFIX = "track_"
CLIP_PREFIX  = "clip_"

FFPP_BASE_TECHS  = {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}
FFPP_EXTRA_TECHS = {"FaceShifter"}

# -------------------------------
# Utility FS
# -------------------------------
def is_track_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith(TRACK_PREFIX)

def iter_tracks(root: Path) -> Iterable[Path]:
    # rglob su directory è veloce e non apre file
    for p in root.rglob("*"):
        if is_track_dir(p):
            yield p

def list_clips(track_dir: Path) -> List[Path]:
    # Cerca solo sotto la track, non ricorsivo profondo
    return [c for c in track_dir.iterdir() if c.is_dir() and c.name.startswith(CLIP_PREFIX)]

def tech_from_ffpp_root(ffpp_root: Path, p: Path) -> str:
    # tecnica = primo segmento immediato sotto il root
    rel = p.relative_to(ffpp_root)
    return rel.parts[0] if rel.parts else "unknown"

def video_key_from_track(root: Path, track_dir: Path) -> Tuple[str, str]:
    """
    Chiave video: (tech, video_id).
    video_id = percorso relativo tra <tech>/.../ e la directory che contiene la track.
    """
    rel = track_dir.relative_to(root)
    parts = rel.parts
    if len(parts) < 2:
        return ("unknown", track_dir.name)
    tech = parts[0]
    # parti tra tech e track_*
    between = parts[1:-1]
    video_id = "/".join(between) if between else track_dir.name
    return (tech, video_id)

def group_by_video(root: Path) -> Dict[Tuple[str, str], List[Path]]:
    G: Dict[Tuple[str, str], List[Path]] = {}
    for tr in iter_tracks(root):
        k = video_key_from_track(root, tr)
        G.setdefault(k, []).append(tr)
    return G

def expand_video_items(groups: Dict[Tuple[str,str], List[Path]]) -> Dict[Tuple[str,str], List[dict]]:
    """
    Restituisce, per ogni chiave video, una lista di item:
      { "track": <path>, "clips": [<path_clip1>, ...] }
    """
    out: Dict[Tuple[str,str], List[dict]] = {}
    for key, tracks in groups.items():
        items = []
        for tr in tracks:
            clips = list_clips(tr)
            # Se non ci sono clip, teniamo traccia vuota ma registriamo la track
            items.append({
                "track": str(tr),
                "clips": [str(c) for c in sorted(clips)]
            })
        out[key] = items
    return out

def sample_split(keys: List, ratios: Tuple[float,float,float]) -> Tuple[List, List, List]:
    r_train, r_val, r_test = ratios
    if abs((r_train + r_val + r_test) - 1.0) > 1e-6:
        raise ValueError("ffpp_base_ratios deve sommare a 1.0")
    n = len(keys)
    n_train = int(round(n * r_train))
    n_val   = int(round(n * r_val))
    n_test  = max(0, n - n_train - n_val)
    return keys[:n_train], keys[n_train:n_train+n_val], keys[n_train+n_val:n_train+n_val+n_test]

# -------------------------------
# Raccolta gruppi per dataset
# -------------------------------
def collect_ffpp_groups(ffpp_root: Path):
    """Ritorna due mapping per-video: base e extra."""
    raw = group_by_video(ffpp_root)
    base, extra = {}, {}
    for (tech, vid), tracks in raw.items():
        t_low = tech.lower()
        is_real = t_low in {"real", "original", "original_sequences", "youtube", "real_youtube"}
        if is_real or tech in FFPP_BASE_TECHS:
            base[(tech, vid)] = tracks
        elif tech in FFPP_EXTRA_TECHS:
            extra[(tech, vid)] = tracks
        # altri tech ignorati
    return base, extra

def collect_generic_groups(root: Path):
    """Per DFD, CelebDF, FFIW: nessun filtro di tech."""
    return group_by_video(root)

# -------------------------------
# Composizione split
# -------------------------------
def add_videos_to_split(dst_list: List[dict], video_map: Dict[Tuple[str,str], List[dict]], keys: List[Tuple[str,str]]):
    for (tech, vid) in keys:
        items = video_map[(tech, vid)]
        dst_list.append({
            "tech": tech,
            "video_id": vid,
            "tracks": items  # lista di {track, clips:[...]}
        })

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Crea split per-video includendo track e clip complete.")
    ap.add_argument("--ffpp_root", type=Path, required=True, help="Root FaceForensics++ (C23)")
    ap.add_argument("--dfd_root", type=Path, required=False, help="Root DeepFakeDetection")
    ap.add_argument("--celebdf_root", type=Path, required=False, help="Root CelebDF v2")
    ap.add_argument("--ffiw_root", type=Path, required=False, help="Root FFIW")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--ffpp_base_ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--extra_ratio_val", type=float, default=0.5,
                    help="Quota dei video extra (FaceShifter, DFD) che va in val, resto in test2")

    ap.add_argument("--cap_celeb_val", type=int, default=None)
    ap.add_argument("--cap_ffiw_val", type=int, default=None)

    ap.add_argument("--out", type=Path, required=True, help="Path file JSON di output")
    args = ap.parse_args()

    random.seed(args.seed)

    result = {
        "meta": {
            "seed": args.seed,
            "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "schema": {
                "split": {
                    "tech": "string",
                    "video_id": "string",
                    "tracks": [
                        {
                            "track": "path",
                            "clips": ["path", "..."]
                        }
                    ]
                }
            }
        },
        "splits": {"train": [], "val": [], "test": [], "test2": []}
    }

    # ---------- FF++ ----------
    ffpp_base_groups, ffpp_extra_groups = collect_ffpp_groups(args.ffpp_root)
    ffpp_base_items = expand_video_items(ffpp_base_groups)
    ffpp_extra_items = expand_video_items(ffpp_extra_groups)

    base_keys = list(ffpp_base_items.keys())
    random.shuffle(base_keys)
    tr_keys, va_keys, te_keys = sample_split(base_keys, tuple(args.ffpp_base_ratios))

    add_videos_to_split(result["splits"]["train"], ffpp_base_items, tr_keys)
    add_videos_to_split(result["splits"]["val"],   ffpp_base_items, va_keys)
    add_videos_to_split(result["splits"]["test"],  ffpp_base_items, te_keys)

    # ---------- FaceShifter extra dentro FF++ ----------
    extra_keys = list(ffpp_extra_items.keys())
    random.shuffle(extra_keys)
    n_val_extra = int(round(len(extra_keys) * args.extra_ratio_val))
    add_videos_to_split(result["splits"]["val"],   ffpp_extra_items, extra_keys[:n_val_extra])
    add_videos_to_split(result["splits"]["test2"], ffpp_extra_items, extra_keys[n_val_extra:])

    # ---------- DFD extra ----------
    if args.dfd_root:
        dfd_groups = collect_generic_groups(args.dfd_root)
        dfd_items  = expand_video_items(dfd_groups)
        dfd_keys = list(dfd_items.keys())
        random.shuffle(dfd_keys)
        n_val_dfd = int(round(len(dfd_keys) * args.extra_ratio_val))
        add_videos_to_split(result["splits"]["val"],   dfd_items, dfd_keys[:n_val_dfd])
        add_videos_to_split(result["splits"]["test2"], dfd_items, dfd_keys[n_val_dfd:])

    # ---------- CelebDF v2 in val ----------
    if args.celebdf_root:
        celeb_groups = collect_generic_groups(args.celebdf_root)
        celeb_items  = expand_video_items(celeb_groups)
        celeb_keys = list(celeb_items.keys())
        random.shuffle(celeb_keys)
        if args.cap_celeb_val is not None:
            celeb_keys = celeb_keys[:args.cap_celeb_val]
        add_videos_to_split(result["splits"]["val"], celeb_items, celeb_keys)

    # ---------- FFIW in val ----------
    if args.ffiw_root:
        ffiw_groups = collect_generic_groups(args.ffiw_root)
        ffiw_items  = expand_video_items(ffiw_groups)
        ffiw_keys = list(ffiw_items.keys())
        random.shuffle(ffiw_keys)
        if args.cap_ffiw_val is not None:
            ffiw_keys = ffiw_keys[:args.cap_ffiw_val]
        add_videos_to_split(result["splits"]["val"], ffiw_items, ffiw_keys)

    # Dedup soft e ordinamento stabile dentro ogni split
    for split_name, items in result["splits"].items():
        # dedup per (tech, video_id)
        seen = set()
        dedup = []
        for v in items:
            k = (v["tech"], v["video_id"])
            if k in seen:
                continue
            seen.add(k)
            # ordina clips per stabilità
            for t in v["tracks"]:
                t["clips"] = sorted(t["clips"])
            # ordina tracks per nome path
            v["tracks"] = sorted(v["tracks"], key=lambda x: x["track"])
            dedup.append(v)
        # ordina per tech poi video_id
        result["splits"][split_name] = sorted(dedup, key=lambda x: (x["tech"], x["video_id"]))

    # Salva JSON
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Log sintetico su stdout
    def count_units(lst: List[dict]):
        n_videos = len(lst)
        n_tracks = sum(len(v["tracks"]) for v in lst)
        n_clips  = sum(len(t["clips"]) for v in lst for t in v["tracks"])
        return {"videos": n_videos, "tracks": n_tracks, "clips": n_clips}

    stats = {k: count_units(v) for k, v in result["splits"].items()}
    print(json.dumps(stats, indent=2), file=sys.stdout)

if __name__ == "__main__":
    main()
