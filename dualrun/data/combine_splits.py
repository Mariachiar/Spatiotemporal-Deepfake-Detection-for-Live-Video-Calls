#!/usr/bin/env python3
# combine_splits_phase2_phase3.py
import json, argparse, os, random
from pathlib import Path

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def dedup_keep_order(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def build_phase(ffpp_idx_path, celeb_A_path, celeb_C_path, celeb_D_path,
                out_path, shuffle=False, seed=123, dedup=True):
    ff = load_json(ffpp_idx_path)        # ha train/val/test
    cA = load_json(celeb_A_path)         # lista
    cC = load_json(celeb_C_path)         # lista
    cD = load_json(celeb_D_path)         # lista

    combo = {
        "train": list(ff["train"]) + list(cA),
        "val":   list(ff["val"])   + list(cC),
        "test":  list(ff["test"])  + list(cD),
    }

    if dedup:
        combo = {k: dedup_keep_order(v) for k, v in combo.items()}
    if shuffle:
        rng = random.Random(seed)
        for k in ("train","val","test"):
            rng.shuffle(combo[k])

    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combo, f, indent=2)
    print(f"✅ Saved {out_path} | sizes:",
          {k: len(v) for k, v in combo.items()})

def main():
    ap = argparse.ArgumentParser(description="Fuse FF++ and CelebDF splits for Phase 2 (FT) and Phase 3 (DAT).")
    ap.add_argument("--ffpp_dir", default="split_ffpp", help="Dir con ffpp_phase2_ft_all6.json e ffpp_phase3_dat_all6.json")
    ap.add_argument("--celeb_dir", default="split_celebdf", help="Dir con celebdf_A2.json, celebdf_A3.json, celebdf_C.json, celebdf_D.json")
    ap.add_argument("--out_dir",  default="split_combined", help="Dir di output per gli index combinati")
    ap.add_argument("--shuffle", action="store_true", help="Mescola gli elementi all'interno di ciascuna split")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no-dedup", action="store_true", help="Non rimuovere duplicati se i path compaiono in entrambe le fonti")
    args = ap.parse_args()

    ffpp_p2 = Path(args.ffpp_dir, "ffpp_phase2_ft_all6.json")
    ffpp_p3 = Path(args.ffpp_dir, "ffpp_phase3_dat_all6.json")

    celeb_A2 = Path(args.celeb_dir, "celebdf_A2.json")
    celeb_A3 = Path(args.celeb_dir, "celebdf_A3.json")
    celeb_C  = Path(args.celeb_dir, "celebdf_C.json")
    celeb_D  = Path(args.celeb_dir, "celebdf_D.json")

    out_p2 = Path(args.out_dir, "phase2_ffpp+celebdf.json")
    out_p3 = Path(args.out_dir, "phase3_ffpp+celebdf.json")

    # Phase 2: FT = FF++ (phase2) + CelebDF A2/C/D
    build_phase(ffpp_p2, celeb_A2, celeb_C, celeb_D, out_p2,
                shuffle=args.shuffle, seed=args.seed, dedup=not args.no_dedup)

    # Phase 3: DAT = FF++ (phase3) + CelebDF A3/C/D
    build_phase(ffpp_p3, celeb_A3, celeb_C, celeb_D, out_p3,
                shuffle=args.shuffle, seed=args.seed, dedup=not args.no_dedup)

if __name__ == "__main__":
    main()
