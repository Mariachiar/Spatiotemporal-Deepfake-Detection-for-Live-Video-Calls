#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uso:
  python eval_suite_auto.py --base altfreezing/tests --outdir results_suite \
      --threshold 0.04 --seeds 42,43,44,45,46

Cerca ricorsivamente:
  altfreezing/tests/{ffpp,ffiw,celebdf}/{metodo}/per_video.csv

Per ogni (dataset, metodo):
- Imposta Fake:Real (ffpp=4.0, celebdf=9.5, ffiw=dal CSV)
- Sottocampiona ratio-matched con seed specifico (1 subset per run)
- Calcola AUROC/AP sul subset
- Bootstrap stratificato (B=2000) su quel subset -> IC95% (AUROC/AP)
- Calcola F1_macro @ soglia indicata (--threshold) sullo stesso subset
- Riassume FPS/latency/memoria (per-video) dal CSV
- Salva JSON e CSV per-run
- Aggrega media±SD tra i 5 run -> summary_all.csv

Se WANDB_PROJECT è impostata, logga su Weights & Biases.
"""
import argparse, csv, json, os, sys, glob
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold

# ---------------- IO ----------------
def load_per_video(path: str):
    rows=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for d in r: rows.append(d)
    if not rows:
        raise ValueError(f"CSV vuoto: {path}")
    y_true  = np.array([int(d["gt_label"]) for d in rows], int)
    y_score = np.array([float(d["video_score"]) for d in rows], float)

    def getf(k, cast=float):
        vals=[]
        for d in rows:
            try:
                vals.append(cast(d[k]))
            except Exception:
                vals.append(np.nan)
        return np.array(vals, float)

    fps   = getf("fps")
    lat   = getf("latency_ms_clip_mean")
    gpu_a = getf("gpu_mem_alloc_peak_mb")
    gpu_r = getf("gpu_mem_reserved_peak_mb")
    cpu_m = getf("cpu_mem_peak_mb")
    return y_true, y_score, fps, lat, gpu_a, gpu_r, cpu_m

def summarize_perf(x: np.ndarray):
    x = x[np.isfinite(x)]
    if x.size==0:
        return {"mean":np.nan,"p50":np.nan,"p95":np.nan}
    return {
        "mean": float(np.mean(x)),
        "p50":  float(np.percentile(x,50)),
        "p95":  float(np.percentile(x,95))
    }

# ------------- ratio-matching -------------
def pick_counts(nR:int, nF:int, fake_per_real:float)->Tuple[int,int]:
    if fake_per_real<=0:
        return nR, 0
    rA = min(nR, int(nF / fake_per_real)); fA = int(round(rA*fake_per_real))
    fB = min(nF, int(nR * fake_per_real)); rB = int(round(fB/fake_per_real)) if fake_per_real>0 else nR
    return (rA,fA) if (rA+fA) >= (rB+fB) else (rB,fB)

def ratio_match_indices(y_true, fake_per_real, rng, frac=1.0):
    real_idx = np.where(y_true==0)[0]; fake_idx = np.where(y_true==1)[0]
    if real_idx.size==0 or fake_idx.size==0:
        raise ValueError("Classi insufficienti per ratio-matching.")
    nRmax, nFmax = pick_counts(len(real_idx), len(fake_idx), fake_per_real)
    nR = max(1, int(nRmax*frac)); nF = max(1, int(nFmax*frac))
    if nR==0 or nF==0:
        raise ValueError("Campioni insufficienti dopo ratio-matching.")
    sel_R = rng.choice(real_idx, size=nR, replace=False)
    sel_F = rng.choice(fake_idx, size=nF, replace=False)
    return np.concatenate([sel_R, sel_F])

# ------------- metriche -------------
def metrics_threshold_free(y, s):
    return {"auc": float(roc_auc_score(y, s)),
            "ap":  float(average_precision_score(y, s))}

def mean_sd(a: List[float]) -> Tuple[float,float]:
    x = np.asarray(a, float)
    return float(np.nanmean(x)), float(np.nanstd(x, ddof=1))

# ------------- config -------------
@dataclass
class EvalConfig:
    per_video: str
    dataset: str
    method: str
    fake_per_real: float
    bootstrap: int = 2000
    seed: int = 42
    outdir: str = "results_suite"
    threshold: float = 0.04
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

# ------------- run singolo -------------
def run_one(cfg: EvalConfig) -> Dict:
    y, s, fps, lat, gpu_a, gpu_r, cpu_m = load_per_video(cfg.per_video)
    rng = np.random.default_rng(cfg.seed)

    # 1) pool ratio-matched unica
    idx_pool = ratio_match_indices(y, cfg.fake_per_real, rng)
    yt_pool, st_pool = y[idx_pool], s[idx_pool]

    # 2) 5 subset disgiunti via StratifiedKFold sulla pool
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)

    aucL=[]; apL=[]; f1L=[]; per_fold=[]
    for k, (_, fold_idx) in enumerate(skf.split(yt_pool, yt_pool), start=1):
        yt, st = yt_pool[fold_idx], st_pool[fold_idx]
        auc = roc_auc_score(yt, st)
        ap  = average_precision_score(yt, st)
        yhat = (st >= cfg.threshold).astype(int)
        prec = precision_score(yt, yhat, average="macro", zero_division=0)
        rec  = recall_score(yt,  yhat, average="macro", zero_division=0)
        f1m = f1_score(yt, yhat, average="macro")
        per_fold.append({
            "fold": k, "n": int(len(fold_idx)),
            "n_real": int((yt==0).sum()), "n_fake": int((yt==1).sum()),
            "auc": auc, "ap": ap,
            "f1_macro_at_tau": f1m,
            "precision_macro_at_tau": prec,
            "recall_macro_at_tau": rec
        })

        aucL.append(auc); apL.append(ap); f1L.append(f1m)
        precL = locals().get("precL", []); recL = locals().get("recL", [])
        precL.append(prec); recL.append(rec)

    # 3) bootstrap opzionale sull’intera pool (non sul singolo fold)
    pos, neg = np.where(yt_pool==1)[0], np.where(yt_pool==0)[0]
    auc_bs=[]; ap_bs=[]
    for _ in range(cfg.bootstrap):
        bi = np.concatenate([rng.choice(pos, len(pos), True),
                             rng.choice(neg, len(neg), True)])
        auc_bs.append(roc_auc_score(yt_pool[bi], st_pool[bi]))
        ap_bs.append(average_precision_score(yt_pool[bi], st_pool[bi]))
    lo_auc, hi_auc = np.nanpercentile(auc_bs, [2.5, 97.5])
    lo_ap,  hi_ap  = np.nanpercentile(ap_bs,  [2.5, 97.5])
    prec_mean = float(np.mean(precL)); prec_sd = float(np.std(precL, ddof=1))
    rec_mean  = float(np.mean(recL));  rec_sd  = float(np.std(recL,  ddof=1))

    return {
        "config": asdict(cfg),
        "counts_available": {"real": int((y==0).sum()), "fake": int((y==1).sum()), "total": int(len(y))},
        "subset": {"n": int(len(idx_pool)), "n_real": int((yt_pool==0).sum()), "n_fake": int((yt_pool==1).sum())},
        "metrics_mean_sd": {
            "auc_mean": float(np.mean(aucL)), "auc_sd": float(np.std(aucL, ddof=1)),
            "ap_mean":  float(np.mean(apL)),  "ap_sd":  float(np.std(apL,  ddof=1)),
            "f1_macro@tau_mean": float(np.mean(f1L)), "f1_macro@tau_sd": float(np.std(f1L, ddof=1)),
            "precision_macro@tau_mean": prec_mean, "precision_macro@tau_sd": prec_sd,
            "recall_macro@tau_mean":    rec_mean,  "recall_macro@tau_sd":  rec_sd
        },
        "bootstrap_ci": {"B": cfg.bootstrap, "auc_ci95": [float(lo_auc), float(hi_auc)],
                         "ap_ci95": [float(lo_ap), float(hi_ap)]},
        "per_fold": per_fold,
        "hardware_stats": {
            "fps": summarize_perf(fps),
            "latency_ms": summarize_perf(lat),
            "gpu_alloc_mb": summarize_perf(gpu_a),
            "gpu_reserved_mb": summarize_perf(gpu_r),
            "cpu_peak_mb": summarize_perf(cpu_m),
        }
    }

# ------------- W&B -------------
def maybe_log_wandb(result: Dict):
    import os
    proj = os.getenv("WANDB_PROJECT")
    if not proj: return
    import wandb
    cfg = result["config"]
    mm = result["metrics_mean_sd"]; bs = result["bootstrap_ci"]
    run = wandb.init(project=proj, entity=os.getenv("WANDB_ENTITY", None),
                     name=f'{cfg["dataset"]}/{cfg["method"]} seed={cfg["seed"]}',
                     config=cfg, reinit=True)
    wandb.log({
        "auc_mean": mm.get("auc_mean"), "auc_sd": mm.get("auc_sd"),
        "ap_mean":  mm.get("ap_mean"),  "ap_sd":  mm.get("ap_sd"),
        "f1_macro@tau_mean": mm.get("f1_macro@tau_mean"),
        "f1_macro@tau_sd":   mm.get("f1_macro@tau_sd"),
        "auc/ci95_lo": bs["auc_ci95"][0], "auc/ci95_hi": bs["auc_ci95"][1],
        "ap/ci95_lo":  bs["ap_ci95"][0],  "ap/ci95_hi":  bs["ap_ci95"][1],
    })
    run.finish()

# ------------- discovery + main -------------
FIXED_RATIOS = {
    "ffpp": 4.0,      # Fake:Real
    "celebdf": 1.91,   # Fake:Real
    "ffiw": 1.0
}

def discover_jobs(base: str, datasets: List[str]) -> List[Tuple[str,str,str]]:
    jobs=[]
    for ds in datasets:
        ds_dir = os.path.join(base, ds)
        if not os.path.isdir(ds_dir):
            print(f"[skip] missing: {ds_dir}", file=sys.stderr); continue
        for method_dir in sorted(d for d in glob.glob(os.path.join(ds_dir, "*")) if os.path.isdir(d)):
            per_csv = os.path.join(method_dir, "per_video.csv")
            if os.path.isfile(per_csv):
                jobs.append((ds, os.path.basename(method_dir), per_csv))
    return jobs

def infer_ratio(ds: str, per_csv: str) -> float:
    if ds in FIXED_RATIOS and FIXED_RATIOS[ds] is not None:
        return FIXED_RATIOS[ds]
    y, *_ = load_per_video(per_csv)
    nR = int((y==0).sum()); nF = int((y==1).sum())
    return float(nF / max(1, nR))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=None,
                help="Radice per discovery ricorsiva se non passi i per_video espliciti.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threshold", type=float, required=True, help="Soglia decisione per F1_macro (applicata a video_score).")
    ap.add_argument("--seeds", default="42,43,44,45,46", help="Cinque seed separati da virgola.")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--per_video_celebdf", type=str, default=None)
    ap.add_argument("--per_video_ffiw",    type=str, default=None)
    ap.add_argument("--per_video_ffpp",    type=str, default=None)
    ap.add_argument("--method",            type=str, default="demo_official")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    datasets = ["ffpp","ffiw","celebdf"]
    seed_list = [int(s) for s in args.seeds.split(",")]
    if len(seed_list) != 5:
        print("[avviso] --seeds non ha 5 valori; proseguo comunque.", file=sys.stderr)

    jobs = []
    if args.per_video_celebdf:
        jobs.append(("celebdf", args.method, args.per_video_celebdf))
    if args.per_video_ffiw:
        jobs.append(("ffiw", args.method, args.per_video_ffiw))
    if args.per_video_ffpp:
        jobs.append(("ffpp", args.method, args.per_video_ffpp))
    if not jobs:  # fallback alla discovery su --base
        jobs = discover_jobs(args.base, ["ffpp","ffiw","celebdf"])
    if not jobs:
        print("Nessun per_video.csv trovato.", file=sys.stderr); sys.exit(1)


    summary_rows = [[
        "dataset","method","n_avail_real","n_avail_fake","fake_per_real",
        "runs",
        "auc_mean","auc_sd","auc_ci_lo","auc_ci_hi",
        "ap_mean","ap_sd","ap_ci_lo","ap_ci_hi",
        "f1_macro@tau_mean","f1_macro@tau_sd",
        "precision_macro@tau_mean","precision_macro@tau_sd",
        "recall_macro@tau_mean","recall_macro@tau_sd",
        "fps_mean","fps_p95","lat_p50","lat_mean","lat_p95",
        "gpu_alloc_p95","gpu_reserved_p95","cpu_peak_p95",
        "out_dir"
        ]]


    for ds, method, per_csv in jobs:
        fpr = infer_ratio(ds, per_csv)
        out_dir_ds = os.path.join(args.outdir, ds, method); os.makedirs(out_dir_ds, exist_ok=True)

        aucL=[]; apL=[]; f1L=[]; precL_all=[]; recL_all=[]

        last_result=None

        for sd in seed_list:
            cfg = EvalConfig(per_video=per_csv, dataset=ds, method=method,
                             fake_per_real=fpr, outdir=out_dir_ds,
                             seed=sd, bootstrap=args.bootstrap, threshold=args.threshold)
            result = run_one(cfg); last_result = result

            # salva per-run
            out_json = os.path.join(out_dir_ds, f"summary_seed{sd}.json")
            with open(out_json, "w") as f:
                json.dump(result, f)

            # tabellina per-run minimale
            per_row_csv = os.path.join(out_dir_ds, f"metrics_seed{sd}.csv")
            with open(per_row_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "seed","n_pool","n_real","n_fake","threshold",
                    "auc_mean","auc_sd","ap_mean","ap_sd",
                    "f1_macro@tau_mean","f1_macro@tau_sd",
                    "precision_macro@tau_mean","precision_macro@tau_sd",
                    "recall_macro@tau_mean","recall_macro@tau_sd"
                    ])
                mm = result["metrics_mean_sd"]; subs = result.get("subset", {})
                w.writerow([
                sd, subs.get("n", np.nan), subs.get("n_real", np.nan), subs.get("n_fake", np.nan),
                args.threshold,
                mm.get("auc_mean", np.nan), mm.get("auc_sd", np.nan),
                mm.get("ap_mean", np.nan),  mm.get("ap_sd", np.nan),
                mm.get("f1_macro@tau_mean", np.nan), mm.get("f1_macro@tau_sd", np.nan),
                mm.get("precision_macro@tau_mean", np.nan), mm.get("precision_macro@tau_sd", np.nan),
                mm.get("recall_macro@tau_mean",    np.nan), mm.get("recall_macro@tau_sd",    np.nan),
                ])



            aucL.append(result["metrics_mean_sd"]["auc_mean"])
            apL.append(result["metrics_mean_sd"]["ap_mean"])
            f1L.append(result["metrics_mean_sd"]["f1_macro@tau_mean"])
            precL_all.append(result["metrics_mean_sd"]["precision_macro@tau_mean"])
            recL_all.append(result["metrics_mean_sd"]["recall_macro@tau_mean"])



            maybe_log_wandb(result)

        # aggregazione su 5 run
        auc_m, auc_sd = mean_sd(aucL)
        ap_m,  ap_sd  = mean_sd(apL)
        f1_m,  f1_sd  = mean_sd(f1L)
        prec_m, prec_sd = mean_sd(precL_all)
        rec_m,  rec_sd  = mean_sd(recL_all)

        hw = last_result["hardware_stats"] if last_result else {
            "fps":{"mean":np.nan,"p50":np.nan,"p95":np.nan},
            "latency_ms":{"mean":np.nan,"p50":np.nan,"p95":np.nan},
            "gpu_alloc_mb":{"p95":np.nan},
            "gpu_reserved_mb":{"p95":np.nan},
            "cpu_peak_mb":{"p95":np.nan},
        }

        # CI95% riporto quelli dell’ultimo run (stesso protocollo, subset cambia col seed)
        bs = last_result["bootstrap_ci"] if last_result else {"auc_ci95":[np.nan,np.nan], "ap_ci95":[np.nan,np.nan]}

        summary_rows.append([
            ds, method,
            last_result["counts_available"]["real"], last_result["counts_available"]["fake"],
            fpr, len(seed_list),
            auc_m, auc_sd, bs["auc_ci95"][0], bs["auc_ci95"][1],
            ap_m,  ap_sd,  bs["ap_ci95"][0],  bs["ap_ci95"][1],
            f1_m,  f1_sd,
            prec_m, prec_sd,
            rec_m,  rec_sd,
            hw["fps"]["mean"], hw["fps"]["p95"],
            hw["latency_ms"]["p50"], hw["latency_ms"]["mean"], hw["latency_ms"]["p95"],
            hw["gpu_alloc_mb"]["p95"], hw["gpu_reserved_mb"]["p95"], hw["cpu_peak_mb"]["p95"],
            out_dir_ds
            ])


        print(f"[OK] {ds}/{method} -> {out_dir_ds}")

    summary_path = os.path.join(args.outdir, "summary_all.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f); w.writerows(summary_rows)
    print(f"[SUMMARY] {summary_path}")

if __name__ == "__main__":
    main()
