#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, json, math, argparse, subprocess, sys, time, shlex
from collections import defaultdict
import numpy as np
from datetime import datetime
from tqdm import tqdm
import wandb

# ===================== CONFIG DI DEFAULT =====================
DATASETS = [
   #{"name": "ffpp",    "list_path": None,                         "dataset_root": "datasets/raw_datasets/FaceForensics++/FaceForensics++_C23"},
    #{"name": "celebdf", "list_path": "raw_datasets/List_fixed.txt","dataset_root": None},
    {"name": "ffiw",    "list_path": None,                         "dataset_root": "raw_datasets/FFIW10K-v1-release-test"},
    #{"name": "ffpp_faceshifter_mixed", "list_path": "lists/faceshifter_with_real.txt", "dataset_root": None},
    #{"name": "ffpp_dfd_mixed",        "list_path": "lists/real_and_dfd.txt",          "dataset_root": None},
]
POOL_METHODS = ["mean", "logit_median","topk","topk_median","percentile","trimmed_mean","adaptive"]
#POOL_METHODS = ["mean","topk","topk_median","percentile","trimmed_mean","adaptive"]

BASE_ARGS = {
    "--cfg_path": "i3d_ori.yaml",
    "--ckpt_path": "altfreezing/checkpoints/model.pth",
    "--optimal_threshold": "0.4",
    "--max_frame": "400",
    "--stride": "5",
    "--batch_clips": "8",
    "--disable_penalty": "",
    "--amp": "",               # flag booleano
    "--channels_last": "",     # flag booleano
    "--q_weighting": "",       # flag booleano (QA on)
}

VIDEO_EXTS = {".mp4",".avi",".mov",".mkv",".webm"}

# ===================== UTILS =====================
def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def count_videos_list(list_path: str) -> int:
    if not list_path or not os.path.exists(list_path): return 0
    n = 0
    with open(list_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            path = line.split(",")[0].strip()
            if os.path.exists(path): n += 1
    return n

def count_videos_root(root: str) -> int:
    if not root or not os.path.exists(root): return 0
    n = 0
    for _, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                n += 1
    return n

def build_cmd(test2_entry: str, pervideo: str, summary: str, pool_method: str,
              list_path: str, dataset_root: str, base_args: dict,
              extra_pool_args: dict | None) -> list:
    cmd = [sys.executable, test2_entry,
           "--out_csv_pervideo", pervideo,
           "--out_csv_summary", summary,
           "--pool_method", pool_method]
    if list_path: cmd += ["--list_path", list_path]
    if dataset_root: cmd += ["--dataset_root", dataset_root]
    for k, v in base_args.items():
        if v == "": cmd.append(k)
        else: cmd += [k, v]
    if extra_pool_args:
        for k, v in extra_pool_args.items():
            cmd += [k, str(v)]
    return cmd

def run_test2(list_path, dataset_root, pool_method, out_dir,
              extra_pool_args=None, test2_path="altfreezing/test3.py", cwd=None, verbose=True):
    ensure_dir(out_dir)
    pervideo = os.path.join(out_dir, "per_video.csv")
    summary  = os.path.join(out_dir, "summary.csv")
    cmd = build_cmd(test2_path, pervideo, summary, pool_method, list_path, dataset_root, BASE_ARGS, extra_pool_args)
    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    if verbose:
        print(f"[{ts()}] [RUN] {cmd_str}")
        print(f"[{ts()}] [CWD] {os.getcwd() if cwd is None else cwd}")

    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
    dt = time.time() - t0

    if verbose:
        print(f"[{ts()}] [RET] code={res.returncode} elapsed={dt:.2f}s")
        if res.stdout:
            print(f"[{ts()}] [STDOUT]\n{res.stdout}")

    return pervideo, summary, (res.returncode == 0), dt

def read_per_video(pervideo_path):
    if not os.path.exists(pervideo_path):
        print(f"[{ts()}] [WARN] per_video.csv assente: {pervideo_path}")
        return []
    rows=[]
    with open(pervideo_path, newline="") as f:
        r = csv.DictReader(f)
        for d in r: rows.append(d)
    return rows

def to_float(x, default=np.nan):
    try: return float(x)
    except: return default

def to_int(x, default=None):
    try: return int(x)
    except: return default

# --- metriche base senza sklearn ---
def _roc_points(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_t = np.array(y_true, dtype=np.int32)[order]
    y_s = np.array(y_score, dtype=np.float64)[order]
    P = y_t.sum(); N = len(y_t)-P
    tps = np.cumsum(y_t)
    fps = np.cumsum(1 - y_t)
    with np.errstate(divide='ignore', invalid='ignore'):
        tpr = tps / max(1, P)
        fpr = fps / max(1, N)
    return fpr, tpr, y_s

def auc_trapz(x, y): return float(np.trapz(y, x)) if len(x) >= 2 else float('nan')

def roc_auc(y_true, y_score):
    fpr, tpr, _ = _roc_points(y_true, y_score)
    fpr = np.concatenate(([0.0], fpr, [1.0])); tpr = np.concatenate(([0.0], tpr, [1.0]))
    return auc_trapz(fpr, tpr)

def pr_points(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_t = np.array(y_true, dtype=np.int32)[order]
    tp = np.cumsum(y_t); fp = np.cumsum(1 - y_t); fn = y_t.sum() - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = tp / np.maximum(1, tp + fp)
        rec  = tp / np.maximum(1, tp + fn)
    prec = np.concatenate(([1.0], prec))
    rec  = np.concatenate(([0.0], rec))
    return rec, prec

def ap_auc(y_true, y_score):
    rec, prec = pr_points(y_true, y_score)
    return auc_trapz(rec, prec)

def confusion_at_threshold(y_true, y_score, thr):
    y_pred = (np.array(y_score) > thr).astype(np.int32)
    y_true = np.array(y_true, dtype=np.int32)
    tp = int(((y_pred==1)&(y_true==1)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    tpr = tp/ max(1, tp+fn)
    fpr = fp/ max(1, fp+tn)
    acc = (tp+tn)/ max(1, len(y_true))
    ppv = tp/ max(1, tp+fp)
    npv = tn/ max(1, tn+fn)
    f1  = (2*tp)/ max(1, 2*tp+fp+fn)
    return {"tp":tp,"tn":tn,"fp":fp,"fn":fn,"tpr":tpr,"fpr":fpr,"acc":acc,"ppv":ppv,"npv":npv,"f1":f1}

def youden_threshold(y_true, y_score):
    fpr, tpr, thr = _roc_points(y_true, y_score)
    if len(thr)==0: return 0.5
    j = tpr - fpr
    k = int(np.argmax(j)) if len(j)>0 else 0
    return float(thr[k])

def neyman_pearson_threshold(y_true, y_score, fpr_target=0.01):
    fpr, tpr, thr = _roc_points(y_true, y_score)
    if len(thr)==0: return 0.5
    ok = np.where(fpr <= fpr_target)[0]
    if ok.size == 0:
        k = int(np.argmin(fpr)); return float(thr[k])
    k = ok[-1]; return float(thr[k])

# ===================== MAIN =====================
def main():
    ap = argparse.ArgumentParser("Batch evaluation runner per TEST2.py + W&B")
    ap.add_argument("--out_dir", default="altfreezing/testFfiw", help="Cartella radice per gli output")
    ap.add_argument("--fpr_np", type=float, default=0.01, help="Vincolo FPR per Neyman-Pearson (ignorato se si usano solo metriche TEST2)")
    ap.add_argument("--topk_ratio", type=float, default=0.3)
    ap.add_argument("--percentile_p", type=float, default=80.0)
    ap.add_argument("--trim_ratio", type=float, default=0.2)
    ap.add_argument("--test2_path", default="altfreezing/test3.py", help="Entry script TEST2")
    ap.add_argument("--test2_cwd", default=None, help="CWD per TEST2; default = cwd corrente")
    # W&B
    ap.add_argument("--wandb_project", default="altfreezing-ft")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_group", default="batch-eval")
    ap.add_argument("--wandb_run", default=None)
    ap.add_argument("--wandb_mode", default="online", choices=["online","offline","disabled"])
    args = ap.parse_args()

    # W&B init
    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
    elif args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_run,
        config={
            "datasets": DATASETS,
            "pool_methods": POOL_METHODS,
            "base_args": BASE_ARGS,
            "fpr_np_target": args.fpr_np,
            "topk_ratio": args.topk_ratio,
            "percentile_p": args.percentile_p,
            "trim_ratio": args.trim_ratio,
            "test2_path": args.test2_path,
        },
    )

    root_out = args.out_dir
    ensure_dir(root_out)

    summary_rows = []
    summary_header = [
        "dataset","pool_method","threshold_type","threshold",
        "videos","accuracy","auc_roc","pr_auc","f1","tpr","fpr","ppv","npv",
        "tp","tn","fp","fn","mean_fps","mean_latency_ms_clip","elapsed_s_test2"
    ]

    print(f"[{ts()}] ===== BatchEval start =====")
    print(f"[{ts()}] out_dir={root_out}")
    print(f"[{ts()}] test2_path={args.test2_path}  test2_cwd={args.test2_cwd}")
    print(f"[{ts()}] datasets={json.dumps(DATASETS, indent=2)}")
    print(f"[{ts()}] pool_methods={POOL_METHODS}")
    print(f"[{ts()}] base_args={json.dumps(BASE_ARGS, indent=2)}")

    total_tasks = sum(1 for _ in DATASETS) * len(POOL_METHODS)
    pbar = tqdm(total=total_tasks, ncols=0, desc="BatchEval", dynamic_ncols=True)

    for ds in DATASETS:
        ds_name    = ds["name"]
        list_path  = ds.get("list_path")
        dataset_root = ds.get("dataset_root")

        n_list  = count_videos_list(list_path) if list_path else 0
        n_root  = count_videos_root(dataset_root) if dataset_root else 0
        n_videos_pre = n_list if list_path else n_root
        print(f"[{ts()}] [DATASET] {ds_name} -> input_count(pre-scan)={n_videos_pre} (list={n_list} root={n_root})")

        for pm in POOL_METHODS:
            pbar.set_description(f"{ds_name}:{pm}")
            out_dir = os.path.join(root_out, ds_name, pm)

            extra = {}
            if pm in ("topk","topk_median"):
                extra["--topk_ratio"] = args.topk_ratio
            elif pm == "percentile":
                extra["--percentile_p"] = args.percentile_p
            elif pm == "trimmed_mean":
                extra["--trim_ratio"] = args.trim_ratio

            pervideo, _summary_csv, ok, dt = run_test2(
                list_path, dataset_root, pm, out_dir,
                extra_pool_args=extra,
                test2_path=args.test2_path,
                cwd=args.test2_cwd,
                verbose=True
            )

            if not ok:
                print(f"[{ts()}] [SKIP] TEST2 non riuscito per {ds_name}/{pm}. Continuo.")
                pbar.update(1)
                continue

            pv = read_per_video(pervideo)
            if not pv:
                print(f"[{ts()}] [WARN] Nessuna riga in {pervideo}. Continuo.")
                pbar.update(1)
                continue

            # === METRICHE DI TEST2, NIENTE RICALCOLI ===
            summary_path = os.path.join(out_dir, "summary.csv")
            with open(summary_path, newline="") as f:
                r = csv.DictReader(f)
                s = next(r)

            videos = int(s["videos"])
            acc    = float(s["accuracy"])
            auc    = float(s["auc_roc"])
            ap     = float(s["pr_auc"])
            f1     = float(s["f1"])
            import json as _json
            cm     = _json.loads(s["confusion_matrix"])
            tp, tn, fp, fn = cm[1][1], cm[0][0], cm[0][1], cm[1][0]
            fps_m  = float(s["mean_fps"])
            lat_m  = float(s["mean_latency_ms_clip"])
            thr    = float(BASE_ARGS["--optimal_threshold"])

            row = [
                ds_name, pm, "from_test2", f"{thr:.6f}",
                videos, f"{acc:.6f}", f"{auc:.6f}", f"{ap:.6f}", f"{f1:.6f}",
                "nan","nan","nan","nan",
                tp, tn, fp, fn, f"{fps_m:.3f}", f"{lat_m:.3f}", f"{dt:.2f}"
            ]
            summary_rows.append(row)

            wandb.log({
                "dataset": ds_name, "pool_method": pm, "threshold_type": "from_test2",
                "threshold": thr,
                "videos": videos, "accuracy": acc, "auc_roc": auc, "pr_auc": ap, "f1": f1,
                "mean_fps": fps_m, "mean_latency_ms_clip": lat_m, "elapsed_s_test2": dt
            })

            # per-video → tabella W&B + file
            if pv:
                cols = list(pv[0].keys())
                tbl = wandb.Table(columns=cols)
                for row_pv in pv:
                    tbl.add_data(*[row_pv.get(c, "") for c in cols])
                wandb.log({f"per_video/{ds_name}/{pm}": tbl})
                try:
                    wandb.save(pervideo, policy="now")
                except Exception:
                    pass

            pbar.update(1)

    pbar.close()

    # scrivi summary_all.csv
    out_all = os.path.join(root_out, "summary_all.csv")
    ensure_dir(os.path.dirname(out_all))
    with open(out_all, "w", newline="") as f:
        w = csv.writer(f); w.writerow(summary_header); w.writerows(summary_rows)

    meta = {
        "datasets": DATASETS,
        "pool_methods": POOL_METHODS,
        "base_args": BASE_ARGS,
        "fpr_np_target": args.fpr_np
    }
    meta_path = os.path.join(root_out, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # log tabella aggregata e artifact
    tbl_all = wandb.Table(columns=summary_header)
    for r in summary_rows:
        tbl_all.add_data(*r)
    wandb.log({"summary_all_table": tbl_all})

    art = wandb.Artifact("batch_eval_outputs", type="evaluation")
    art.add_file(out_all)
    art.add_file(meta_path)
    wandb.log_artifact(art)

    print(f"[{ts()}] Wrote: {out_all}")
    print(f"[{ts()}] Meta : {meta_path}")
    print(f"[{ts()}] ===== BatchEval end =====")

if __name__ == "__main__":
    main()
