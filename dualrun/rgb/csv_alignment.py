#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, json, csv, argparse, math

# Tecniche canoniche FF++
KNOWN = {"original","Deepfakes","FaceSwap","Face2Face","NeuralTextures","FaceShifter"}

# ---------------------------
# Utils
# ---------------------------
def clamp01(x, eps=1e-6):
    return max(eps, min(1.0 - eps, float(x)))

def prob_to_logit(p, eps=1e-6):
    p = clamp01(p, eps)
    return float(math.log(p / (1.0 - p)))

def has_header_autodetect(csv_path):
    with open(csv_path, "r", newline="") as f:
        first = f.readline().lower()
    return ("video_path" in first) and ("video_score" in first)

def normsep(p: str):
    return p.replace("\\", "/")

# ---------------------------
# Key extraction
# ---------------------------
def key_from_csv_path(p: str):
    """
    CSV: .../<TECH>/<VIDEO>.mp4
    -> (tech, stem_video)
    """
    toks = normsep(p).split("/")
    tech = next((t for t in toks if t in KNOWN), None)
    if tech is None:
        # fallback per 'original' nascosto
        if any(s in "/".join(toks).lower() for s in ["original","pristine","authentic","youtube"]):
            tech = "original"
    if tech is None:
        return (None, None)
    stem = os.path.splitext(toks[-1])[0]
    return (tech, stem)

def key_from_clip_path(p: str):
    """
    Clip dir: .../<TECH>/<VIDEO>/<PERSON>/<clip*>
    -> (tech, stem_video)
    """
    toks = normsep(p).split("/")
    try:
        i = next(idx for idx,t in enumerate(toks) if t in KNOWN)
    except StopIteration:
        # stessa euristica di sopra
        if any(s in "/".join(toks).lower() for s in ["original","pristine","authentic","youtube"]):
            # prova a prendere il token successivo come video
            tech = "original"
            # cerca un candidato video tra i successivi 3 token
            cand = None
            for t in toks[-5:]:
                t0 = os.path.splitext(t)[0]
                if re.fullmatch(r"\d+(_\d+)?", t0):
                    cand = t0; break
            return (tech, cand)
        return (None, None)
    tech = toks[i]
    if i + 1 >= len(toks):
        return (None, None)
    stem = os.path.splitext(toks[i+1])[0]
    # se non sembra un id video, prova i successivi due token
    if not re.fullmatch(r"\d+(_\d+)?", stem):
        for t in toks[i+1:i+4]:
            t0 = os.path.splitext(t)[0]
            if re.fullmatch(r"\d+(_\d+)?", t0):
                stem = t0; break
    return (tech, stem)

def clip_path_from_index_entry(e):
    if isinstance(e, str): return e
    return e.get("path") or e.get("clip_dir") or e.get("dir") or e.get("p")

# ---------------------------
# CSV loading
# ---------------------------
def load_csv_map(csv_path, force_header=None, path_col=0, score_col=6, label_col=None):
    """
    Ritorna: dict[(tech, stem)] -> {"logit":..., "prob":..., "label": int|None}
    Se il CSV ha header 'video_path' e 'video_score' li usa; altrimenti usa colonne posizionali.
    """
    has_hdr = has_header_autodetect(csv_path) if force_header is None else force_header
    out = {}
    with open(csv_path, "r", newline="") as f:
        if has_hdr:
            rd = csv.DictReader(f)
            for r in rd:
                if ("video_path" not in r) or ("video_score" not in r): continue
                k = key_from_csv_path(r["video_path"])
                if None in k: continue
                try:
                    prob = float(r["video_score"])
                except:
                    continue
                lbl = None
                gl = r.get("gt_label", "")
                if str(gl).strip() != "":
                    try: lbl = int(gl)
                    except: pass
                out[k] = {"prob": clamp01(prob), "logit": prob_to_logit(prob), "label": lbl}
        else:
            rr = csv.reader(f)
            for row in rr:
                if len(row) <= max(path_col, score_col): continue
                k = key_from_csv_path(row[path_col])
                if None in k: continue
                try:
                    prob = float(row[score_col])
                except:
                    continue
                lbl = None
                if (label_col is not None) and (label_col < len(row)):
                    try: lbl = int(row[label_col])
                    except: lbl = None
                out[k] = {"prob": clamp01(prob), "logit": prob_to_logit(prob), "label": lbl}
    return out

# ---------------------------
# Processing
# ---------------------------
def process_list(L, csv_map, label_source="csv"):
    """
    label_source: 'csv' | 'index' | 'consensus' | 'skip_mismatch'
    """
    kept = 0; mism = 0; matched = 0; total = len(L)
    OUT = []
    for e in L:
        p = clip_path_from_index_entry(e)
        if not p: continue
        k = key_from_clip_path(p)
        if None in k: continue
        if k not in csv_map: continue
        matched += 1

        info = csv_map[k]
        logit = info["logit"]; prob = info["prob"]; lbl_csv = info["label"]

        # etichetta finale
        y_idx = None if isinstance(e, str) else e.get("y", e.get("label"))
        y_out = y_idx

        if label_source == "csv" and (lbl_csv is not None):
            y_out = lbl_csv
        elif label_source == "index":
            pass
        elif label_source == "consensus":
            if (y_idx is not None) and (lbl_csv is not None) and (int(y_idx) != int(lbl_csv)):
                mism += 1; continue
            y_out = int(y_idx) if y_idx is not None else (int(lbl_csv) if lbl_csv is not None else None)
        elif label_source == "skip_mismatch":
            if (y_idx is not None) and (lbl_csv is not None) and (int(y_idx) != int(lbl_csv)):
                mism += 1; continue

        if isinstance(e, str):
            OUT.append({"path": p, "y": int(y_out) if y_out is not None else 1,
                        "teacher_logit": float(logit), "teacher_prob": float(prob)})
        else:
            ee = dict(e)
            ee["y"] = int(y_out) if y_out is not None else int(ee.get("y", 1))
            ee["teacher_logit"] = float(logit)
            ee["teacher_prob"]  = float(prob)
            OUT.append(ee)
        kept += 1

    print(f"total={total} matched={matched} kept={kept} mismatches_skipped={mism}")
    return OUT

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser("Allinea teacher per-video al tuo index di clip")
    ap.add_argument("--index", required=True, help="JSON index clip-level")
    ap.add_argument("--csv",   required=True, help="CSV per-video con video_path, video_score [, gt_label]")
    ap.add_argument("--out",   required=True, help="JSON di output con teacher_logit/prob")
    ap.add_argument("--label-source", choices=["csv","index","consensus","skip_mismatch"], default="csv")

    # override CSV se senza header
    ap.add_argument("--csv-has-header", action="store_true", help="Forza uso header")
    ap.add_argument("--csv-no-header", action="store_true", help="Forza modalità posizionale")
    ap.add_argument("--csv-path-col",  type=int, default=0)
    ap.add_argument("--csv-score-col", type=int, default=6)
    ap.add_argument("--csv-label-col", type=int, default=None)
    args = ap.parse_args()

    force_header = True if args.csv_has_header else (False if args.csv_no_header else None)

    csv_map = load_csv_map(
        args.csv,
        force_header=force_header,
        path_col=args.csv_path_col,
        score_col=args.csv_score_col,
        label_col=args.csv_label_col
    )
    print(f"csv_keys={len(csv_map)}")

    J = json.load(open(args.index, "r"))
    if isinstance(J, dict):
        OUT = {}
        for split, lst in J.items():
            print(f"[split={split}]")
            OUT[split] = process_list(lst, csv_map, label_source=args.label_source)
    else:
        OUT = process_list(J, csv_map, label_source=args.label_source)

    with open(args.out, "w") as f:
        json.dump(OUT, f)
    print("written:", args.out)

if __name__ == "__main__":
    main()
