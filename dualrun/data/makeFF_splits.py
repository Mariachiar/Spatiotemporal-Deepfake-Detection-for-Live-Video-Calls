#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, json, argparse, random, logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# ----------------- Configurazione logging -----------------
LOG = logging.getLogger("makeFF_splits")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
# ----------------------------------------------------------

BASE_TECHS  = ["deepfakes","face2face","faceswap","neuraltextures"]
EXTRA_TECHS = ["faceshifter","deepfakedetection"]
REAL_ALIASES = {"original","authentic","real","pristine"}

def infer_tech(path: str) -> str:
    p = path.lower().replace("\\","/")
    segs = [s for s in p.split("/") if s]
    for s in segs:
        k = re.sub(r"[-_]", "", s)
        if s in BASE_TECHS + EXTRA_TECHS: return s
        if k in BASE_TECHS + EXTRA_TECHS: return k
        if s in REAL_ALIASES or k in REAL_ALIASES: return "real"
    for t in BASE_TECHS + EXTRA_TECHS:
        if f"/{t}/" in p: return t
    for t in REAL_ALIASES:
        if f"/{t}/" in p: return "real"
    return "unknown"

def find_clip_dirs(root: str) -> List[str]:
    LOG.info("Scanning %s for preprocessed clips...", root)
    out = []
    for r,_,files in os.walk(root):
        if "au_features.npy" in files and "lmk_features.npy" in files:
            out.append(Path(r).as_posix())
    LOG.info("Found %d clip directories", len(out))
    return sorted(out)

def split_5(lst: List[str], ratios, seed: int):
    rnd = random.Random(seed)
    pool = lst[:]
    rnd.shuffle(pool)
    n = len(pool)
    nA1 = int(ratios["A1"] * n)
    nA2 = int(ratios["A2"] * n)
    nA3 = int(ratios["A3"] * n)
    nC  = int(ratios["C"]  * n)
    A1  = pool[:nA1]
    A2  = pool[nA1:nA1+nA2]
    A3  = pool[nA1+nA2:nA1+nA2+nA3]
    C   = pool[nA1+nA2+nA3:nA1+nA2+nA3+nC]
    D   = pool[nA1+nA2+nA3+nC:]
    return {"A1":A1,"A2":A2,"A3":A3,"C":C,"D":D}

def write_json(obj, path):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path,"w") as f: json.dump(obj,f,indent=2)
    LOG.info("Saved JSON %s (%d entries)", path, len(obj) if isinstance(obj, list) else len(obj.keys()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffpp-root", required=True)
    ap.add_argument("--outdir", default="split_ffpp")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ratios-base", default="0.45,0.20,0.15,0.10,0.10")
    ap.add_argument("--ratios-extra", default="0.00,0.40,0.40,0.10,0.10")
    ap.add_argument("--cap_extra_valtest", type=int, default=500)
    args = ap.parse_args()

    # parse ratios
    rB = [float(x) for x in args.ratios_base.split(",")]
    rE = [float(x) for x in args.ratios_extra.split(",")]
    ratios_base  = {"A1":rB[0],"A2":rB[1],"A3":rB[2],"C":rB[3],"D":rB[4]}
    ratios_extra = {"A1":rE[0],"A2":rE[1],"A3":rE[2],"C":rE[3],"D":rE[4]}

    LOG.info("Ratios base: %s", ratios_base)
    LOG.info("Ratios extra: %s", ratios_extra)

    all_dirs = find_clip_dirs(args.ffpp_root)
    by_tech: Dict[str, List[str]] = defaultdict(list)
    for d in all_dirs:
        t = infer_tech(d)
        by_tech[t].append(d)

    LOG.info("Techniques found: %s", {t:len(v) for t,v in by_tech.items()})

    # split per tecnica
    shards = {}
    for tech, lst in by_tech.items():
        if tech=="unknown": 
            LOG.warning("Skipping %d dirs with unknown tech", len(lst))
            continue
        if tech in BASE_TECHS or tech=="real":
            shards[tech] = split_5(lst, ratios_base, seed=args.seed)
        elif tech in EXTRA_TECHS:
            shards[tech] = split_5(lst, ratios_extra, seed=args.seed)
        LOG.info("Shard sizes for %s: %s", tech, {k:len(v) for k,v in shards[tech].items()})

    os.makedirs(args.outdir, exist_ok=True)
    write_json(shards, Path(args.outdir,"ffpp_shards_A1A2A3CD.json").as_posix())

    # costruisci i 3 index
    LOG.info("Building phase-1 index...")
    idx1 = {"train":[], "val":[], "test":[]}
    for t in BASE_TECHS + ["real"]:
        if t in shards:
            idx1["train"] += shards[t]["A1"]
            idx1["val"]   += shards[t]["C"]
            idx1["test"]  += shards[t]["D"]
    for t in EXTRA_TECHS:
        if t in shards:
            idx1["val"]  += shards[t]["C"][:args.cap_extra_valtest]
            idx1["test"] += shards[t]["D"][:args.cap_extra_valtest]
    write_json(idx1, Path(args.outdir,"ffpp_phase1_pretrain.json").as_posix())

    LOG.info("Building phase-2 index...")
    idx2 = {"train":[], "val":[], "test":[]}
    for t in BASE_TECHS + ["real"] + EXTRA_TECHS:
        if t in shards:
            idx2["train"] += shards[t]["A2"]
            idx2["val"]   += shards[t]["C"]
            idx2["test"]  += shards[t]["D"]
    write_json(idx2, Path(args.outdir,"ffpp_phase2_ft_all6.json").as_posix())

    LOG.info("Building phase-3 index...")
    idx3 = {"train":[], "val":[], "test":[]}
    for t in BASE_TECHS + ["real"] + EXTRA_TECHS:
        if t in shards:
            idx3["train"] += shards[t]["A3"]
            idx3["val"]   += shards[t]["C"]
            idx3["test"]  += shards[t]["D"]
    write_json(idx3, Path(args.outdir,"ffpp_phase3_dat_all6.json").as_posix())

    summary = {
        "counts_per_tech": {t:{k:len(v) for k,v in s.items()} for t,s in shards.items()},
        "phase1": {k:len(v) for k,v in idx1.items()},
        "phase2": {k:len(v) for k,v in idx2.items()},
        "phase3": {k:len(v) for k,v in idx3.items()},
    }
    write_json(summary, Path(args.outdir,"ffpp_summary.json").as_posix())
    LOG.info("Process completed. Shards and indexes saved in %s", args.outdir)

if __name__ == "__main__":
    main()
