#!/usr/bin/env python3
import argparse, random, shutil
from pathlib import Path

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")

def iter_videos(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in VIDEO_EXTS:
            yield p

def pick_n(paths, n, seed=None):
    items = list(paths)
    if seed is not None:
        rnd = random.Random(seed)
        rnd.shuffle(items)
    else:
        random.shuffle(items)
    return items[:min(n, len(items))]

def materialize(files, out_root: Path, rel_base: Path, copy: bool):
    for f in files:
        rel = f.relative_to(rel_base)              # preserva struttura
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        if copy:
            shutil.copy2(f, dst)
        else:
            dst.symlink_to(f.resolve())

def sample_ffiw(ffiw_root: Path, out_root: Path, n=50, seed=None, copy=False):
    real_dir = ffiw_root / "real" / "test"
    fake_dir = ffiw_root / "deepfakes" / "test"
    real = pick_n(iter_videos(real_dir), n, seed=seed) if real_dir.exists() else []
    fake = pick_n(iter_videos(fake_dir), n, seed=seed) if fake_dir.exists() else []
    materialize(real, out_root, ffiw_root, copy)
    materialize(fake, out_root, ffiw_root, copy)

def sample_celebdf(celeb_root: Path, out_root: Path, n=50, seed=None, copy=False):
    real_dirs = [celeb_root / "Celeb-real", celeb_root / "YouTube-real"]
    fake_dirs = [celeb_root / "Celeb-synthesis"]
    real_pool = [p for d in real_dirs if d.exists() for p in iter_videos(d)]
    fake_pool = [p for d in fake_dirs if d.exists() for p in iter_videos(d)]
    real = pick_n(real_pool, n, seed=seed)
    fake = pick_n(fake_pool, n, seed=seed)
    materialize(real, out_root, celeb_root, copy)
    materialize(fake, out_root, celeb_root, copy)

def sample_ffpp(ffpp_root: Path, out_root: Path, n_real=50, n_fake=50, manips=None, seed=None, copy=False):
    # Struttura attesa: original/ e manipolazioni come Deepfakes, Face2Face, ...
    real_dir = ffpp_root / "original"
    real = pick_n(iter_videos(real_dir), n_real, seed=seed) if real_dir.exists() else []
    materialize(real, out_root, ffpp_root, copy)

    default = ["Deepfakes","Face2Face","FaceSwap","NeuralTextures","FaceShifter","DeepFakeDetection"]
    todo = manips or [m for m in default if (ffpp_root / m).exists()]
    for m in todo:
        mdir = ffpp_root / m
        fake = pick_n(iter_videos(mdir), n_fake, seed=seed)
        materialize(fake, out_root, ffpp_root, copy)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffiw_root", type=Path)
    ap.add_argument("--celeb_root", type=Path)
    ap.add_argument("--ffpp_root", type=Path)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_per_class", type=int, default=50)
    ap.add_argument("--ffpp_manips", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copia i file invece di creare symlink")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.ffiw_root and args.ffiw_root.exists():
        sample_ffiw(args.ffiw_root, args.out_dir / "ffiw", n=args.n_per_class, seed=args.seed, copy=args.copy)

    if args.celeb_root and args.celeb_root.exists():
        sample_celebdf(args.celeb_root, args.out_dir / "celebdf_v2", n=args.n_per_class, seed=args.seed, copy=args.copy)

    if args.ffpp_root and args.ffpp_root.exists():
        manips = [m.strip() for m in args.ffpp_manips.split(",") if m.strip()] or None
        sample_ffpp(
            args.ffpp_root,
            args.out_dir / "ffpp",
            n_real=args.n_per_class,
            n_fake=args.n_per_class,
            manips=manips,
            seed=args.seed,
            copy=args.copy,
        )

if __name__ == "__main__":
    main()
