#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, random, logging
from pathlib import Path
from collections import defaultdict

# ----------------- Logging -----------------
LOG = logging.getLogger("makeCDF_splits")
def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
# -------------------------------------------

def find_clip_dirs(root: str):
    LOG.info("Scanning processed root for clip dirs: %s", root)
    out = []
    for r, _, files in os.walk(root):
        if "au_features.npy" in files and "lmk_features.npy" in files:
            out.append(Path(r).as_posix())
    LOG.info("Found %d clip directories", len(out))
    return out

def split_A2A3C_D(all_dirs, a2_ratio=0.70, c_ratio=0.10, d_ratio=0.15, seed=123):
    """
    Divide tutto in A2 (FT), A3 (DAT), C (val), D (test) con percentuali fissate.
    """
    random.seed(seed)
    random.shuffle(all_dirs)
    n = len(all_dirs)

    nC  = int(c_ratio * n)
    nD  = int(d_ratio * n)
    nA2 = int(a2_ratio * n)

    C   = all_dirs[:nC]
    D   = all_dirs[nC:nC+nD]
    A2  = all_dirs[nC+nD:nC+nD+nA2]
    A3  = all_dirs[nC+nD+nA2:]

    LOG.info("Split sizes: A2=%d (%.1f%%), A3=%d (%.1f%%), C=%d (%.1f%%), D=%d (%.1f%%)",
             len(A2), 100*len(A2)/n, len(A3), 100*len(A3)/n,
             len(C), 100*len(C)/n, len(D), 100*len(D)/n)

    return {"A2":A2, "A3":A3, "C":C, "D":D}

def write_json(obj, path):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)
    size = len(obj) if isinstance(obj, list) else len(obj.keys())
    LOG.info("Saved %s (%d items)", path, size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", required=True, help="Root di CelebDF preprocessato (clip con au/lmk)")
    ap.add_argument("--outdir", default="split_celebdf", help="Output dir")
    ap.add_argument("--a2-ratio", type=float, default=0.70, help="Quota A2 (FT)")
    ap.add_argument("--c-ratio",  type=float, default=0.10, help="Quota C (VAL)")
    ap.add_argument("--d-ratio",  type=float, default=0.15, help="Quota D (TEST)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--verbose", action="store_true", help="Log dettagliato")
    args = ap.parse_args()

    setup_logging(args.verbose)

    LOG.info("Parameters | outdir=%s | ratios=(A2=%.2f, A3=rest, C=%.2f, D=%.2f) | seed=%d",
             args.outdir, args.a2_ratio, args.c_ratio, args.d_ratio, args.seed)

    all_dirs = find_clip_dirs(args.processed_root)
    splits = split_A2A3C_D(all_dirs,
                           a2_ratio=args.a2_ratio,
                           c_ratio=args.c_ratio,
                           d_ratio=args.d_ratio,
                           seed=args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    for k in ("A2","A3","C","D"):
        write_json(splits[k], Path(args.outdir, f"celebdf_{k}.json").as_posix())

    summary = {k: len(v) for k,v in splits.items()}
    write_json(summary, Path(args.outdir, "celebdf_summary.json").as_posix())
    LOG.info("Done. Shards saved in %s | %s", args.outdir, summary)

if __name__ == "__main__":
    main()
