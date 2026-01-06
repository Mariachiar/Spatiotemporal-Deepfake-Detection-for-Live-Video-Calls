#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, glob, csv, argparse, time, threading
import cv2, numpy as np, torch, torch.nn.functional as F
from tqdm import tqdm

# ===== metriche =====
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, confusion_matrix
    SK_OK = True
except Exception:
    SK_OK = False

# ===== util monitor CPU/GPU (opzionale, fallisce in silenzio) =====
try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pynvml = None
    NVML_HANDLE = None

class UtilizationSampler:
    def __init__(self, period_sec=0.2):
        self.period = period_sec
        self._stop = threading.Event()
        self.cpu = []
        self.gpu = []
        self.gpu_mem = []
        self._th = None

    def _tick(self):
        while not self._stop.is_set():
            if psutil:
                try: self.cpu.append(psutil.cpu_percent(interval=None))
                except Exception: pass
            if pynvml and NVML_HANDLE is not None:
                try:
                    u = pynvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE)
                    m = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
                    self.gpu.append(u.gpu)                 # 0..100
                    self.gpu_mem.append(m.used / (1024**3))# GB
                except Exception: pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._stop.clear()
        self._th = threading.Thread(target=self._tick, daemon=True)
        self._th.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._th: self._th.join(timeout=1.0)

    def summary(self):
        import statistics as st
        def s(arr, fn, default=np.nan):
            try: return fn(arr) if arr else default
            except Exception: return default
        return {
            "cpu_mean": s(self.cpu, lambda a: float(sum(a)/len(a))),
            "gpu_mean": s(self.gpu, lambda a: float(sum(a)/len(a))),
            "gpu_mem_mean": s(self.gpu_mem, lambda a: float(sum(a)/len(a))),
            "gpu_mem_max": s(self.gpu_mem, max)
        }

# ===== import DEMO UFFICIALE (stessa pipeline) =====
from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader

# normalizzazione identica alla demo
def get_mean_std(device):
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255], device=device).view(1,3,1,1,1)
    std  = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255], device=device).view(1,3,1,1,1)
    return mean, std

# ===== RACCOLTA VIDEO (come file originale) =====
def collect_videos(root_dir, per_class=500):
    import glob, os, random
    exts = (".mp4",".avi",".mov",".mkv")
    REAL_TOK = ("/original/", "/original_sequences/", "/celeb-real/", "/youtube-real/", "/real/", "/source/")
    FAKE_TOK = ("/target/", "/manipulated_sequences/", "/deepfakes/", "/face2face/", "/faceswap/", "/neuraltextures/", "/fake/", "/celeb-synthesis/")
    DATASETS_HINT = ("ffpp","ffiw","celebdf_v2","faceforensics++","faceforensics","celebdf")
    SUBSETS_HINT = ("train","val","test","c23","c40")

    def is_video(p): return p.lower().endswith(exts)
    def classify(p):
        pl = p.replace("\\","/").lower()
        if any(t in pl for t in REAL_TOK): return 0
        if any(t in pl for t in FAKE_TOK): return 1
        return None

    def dataset_name(parts_lower):
        for s in DATASETS_HINT:
            if s in parts_lower: return s
        if any(x in parts_lower for x in ("deepfakes","face2face","faceswap","neuraltextures","original","original_sequences")):
            return "ffpp"
        return "unknown"

    def subset_name(parts_lower):
        for s in SUBSETS_HINT:
            if s in parts_lower: return s
        return "unknown"

    seen=set(); pool_real=[]; pool_fake=[]
    patterns=[]
    for e in ("*.mp4","*.avi","*.mov","*.mkv"):
        patterns += [
            os.path.join(root_dir, "**", "original_sequences", "**", "c*", "videos", e),
            os.path.join(root_dir, "**", "manipulated_sequences", "**", "c*", "videos", e),
        ]
        patterns += [os.path.join(root_dir, "**", e)]

    for pat in patterns:
        for v in glob.iglob(pat, recursive=True):
            rv=os.path.realpath(v)
            if rv in seen: continue
            seen.add(rv)
            if not os.path.exists(rv) or not is_video(rv): continue
            lab=classify(rv)
            if lab is None: continue
            parts_lower=[x.lower() for x in rv.replace("\\","/").split("/")]
            dset=dataset_name(parts_lower); subset=subset_name(parts_lower)
            item=(rv, lab, dset, subset)
            if lab==0: pool_real.append(item)
            else: pool_fake.append(item)

    rng=random.Random(0)
    rng.shuffle(pool_real); rng.shuffle(pool_fake)
    pick_real = pool_real[:per_class] if per_class else pool_real
    pick_fake = pool_fake[:per_class] if per_class else pool_fake

    out=[]
    for a,b in zip(pick_real, pick_fake):
        out.append(a); out.append(b)
    out.extend(pick_real[len(out)//2:])
    out.extend(pick_fake[len(out)//2:])
    return out  # [(video_path, label, dataset, subset)]

def collect_videos_from_list(root_dir, list_path, seed=0):
    import random, os
    real, fake = [], []
    with open(list_path, "r") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            lab, rel = line.split(maxsplit=1)
            lab = int(lab)
            abs_path = os.path.join(root_dir, rel)
            item = (abs_path, lab, "celebdf", "test")
            if lab==0: real.append(item)
            else: fake.append(item)

    return real + fake


# ===== valutazione per singolo video + timing =====
def eval_video_demo_timed(video_path, classifier, crop_align_func, device, cfg_obj, max_frame, optimal_threshold, cache_dir=None, write_vis=False, out_dir="prediction"):
    mean, std = get_mean_std(device)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.avi") if write_vis else None
    cache_file = os.path.join(cache_dir or ".", f"cache_{os.path.basename(video_path)}_{max_frame}.pth")

    # reset peak memory
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    t_detect = 0.0
    t_aligninfer = 0.0

    # frames + detection/landmark (come demo)
    if os.path.exists(cache_file):
        t1 = time.perf_counter()
        detect_res, all_lm68 = torch.load(cache_file, map_location="cpu", weights_only=False)
        frames = grab_all_frames(video_path, max_size=max_frame, cvt=True)
        n = min(len(frames), len(detect_res), len(all_lm68))
        if n == 0:
            # gestisci il caso senza frame come già fatto più sotto
            t_total = time.perf_counter() - t0
            util = {"cpu_mean": np.nan, "gpu_mean": np.nan, "gpu_mem_mean": np.nan, "gpu_mem_max": np.nan}
            peak_mem = torch.cuda.max_memory_allocated()/ (1024**3) if device=="cuda" else np.nan
            return {"video_score": 0.0, "pred_label": 0, "frames": 0, "clips": 0,
                    "t_total": t_total, "t_detect": t_detect, "t_aligninfer": 0.0,
                    "fps_end2end": 0.0, "fps_preproc": 0.0, "fps_model": 0.0,
                    "cpu_mean": util["cpu_mean"], "gpu_mean": util["gpu_mean"],
                    "gpu_mem_mean": util["gpu_mem_mean"], "gpu_mem_max": util["gpu_mem_max"],
                    "gpu_mem_peak": peak_mem}
        # tronca tutto allo stesso n
        frames     = frames[:n]
        detect_res = detect_res[:n]
        all_lm68   = all_lm68[:n]
        t_detect += time.perf_counter() - t1
    else:
        t1 = time.perf_counter()
        detect_res, all_lm68, frames = detect_all(video_path, return_frames=True, max_size=max_frame)
        t_detect += time.perf_counter() - t1
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save((detect_res, all_lm68), cache_file)

    if len(frames)==0:
        t_total = time.perf_counter() - t0
        util = {"cpu_mean": np.nan, "gpu_mean": np.nan, "gpu_mem_mean": np.nan, "gpu_mem_max": np.nan}
        peak_mem = torch.cuda.max_memory_allocated()/ (1024**3) if device=="cuda" else np.nan
        return {"video_score": 0.0, "pred_label": 0, "frames": 0, "clips": 0,
                "t_total": t_total, "t_detect": t_detect, "t_aligninfer": 0.0,
                "fps_end2end": 0.0, "fps_preproc": 0.0, "fps_model": 0.0,
                "cpu_mean": util["cpu_mean"], "gpu_mean": util["gpu_mean"],
                "gpu_mem_mean": util["gpu_mem_mean"], "gpu_mem_max": util["gpu_mem_max"],
                "gpu_mem_peak": peak_mem}

    shape = frames[0].shape[:2]
    all_detect_res = []
    assert len(all_lm68) == len(detect_res)
    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_faces.append((box, lm5, face_lm68, score))
        all_detect_res.append(new_faces)
    detect_res = all_detect_res

    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)
    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res)

    data_storage = {}
    frame_boxes = {}

    # dopo: shape = frames[0].shape[:2]
    # dopo: tracks = multiple_tracking(detect_res) ... tuples = ...
    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        assert len(detect_res[start:end]) == len(track)
        for j, (face, frame_idx) in enumerate(zip(track, range(start, end))):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)

            # shift landmarks/box nel crop
            top_left = big_box[:2][None, :]
            new_lm5  = lm5  - top_left
            new_lm68 = lm68 - top_left
            new_box  = (box.reshape(2,2) - top_left).reshape(-1)
            info = (new_box, new_lm5, new_lm68, big_box)

            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]
            if frame_idx >= len(frames):
                continue  # indice fuori range, salta


            base = f"{track_i}_{j}_"
            data_storage[f"{base}img"] = cropped
            data_storage[f"{base}ldm"] = info
            data_storage[f"{base}idx"] = frame_idx

            frame_boxes[frame_idx] = np.rint(box).astype(int)

    # --- sostituisci da: super_clips = [] ... fino alla costruzione di clips_for_video ---
    clips_for_video = []
    clip_size = cfg_obj.clip_size

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        T = len(track)
        if T == 0:
            continue  # traccia vuota: niente clip

        # indici base nella traccia [0..T-1]
        base = list(range(T))

        if T >= clip_size:
            # finestre scorrevoli classiche, tutte valide
            for s in range(0, T - clip_size + 1):
                idxs = list(range(s, s + clip_size))
                clips_for_video.append([(track_i, j) for j in idxs])
        else:
            # padding riflesso senza liste vuote
            need = clip_size - T
            # riflesso sul range [1..T-2] se possibile, altrimenti ripeti estremi
            left = base[1:T-1][::-1] if T > 2 else [base[0]] * need
            right = base[1:T-1][::-1] if T > 2 else [base[-1]] * need
            # alloca metà a sinistra e metà a destra
            l = need // 2
            r = need - l
            padded = left * ((l + len(left) - 1) // len(left)) if len(left) > 0 else [base[0]] * l
            padded = padded[:l] + base
            pad_right = right * ((r + len(right) - 1) // len(right)) if len(right) > 0 else [base[-1]] * r
            padded = padded + pad_right[:r]
            assert len(padded) == clip_size
            clips_for_video.append([(track_i, j) for j in padded])

    preds = []
    frame_res = {}

    # campionamento CPU/GPU durante align+infer
    with UtilizationSampler(period_sec=0.2) as mon:
        for clip in clips_for_video:
            images   = [data_storage[f"{i}_{j}_img"] for i, j in clip]
            landmarks= [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
            frame_ids= [data_storage[f"{i}_{j}_idx"] for i, j in clip]

            t_a0 = time.perf_counter()
            _, images_align = crop_align_func(landmarks, images)

            images_t = torch.as_tensor(images_align, dtype=torch.float32, device=device).permute(3,0,1,2)
            images_t = images_t.unsqueeze(0)
            images_t = images_t.sub(mean).div(std)

            if device == "cuda": torch.cuda.synchronize()
            t_f0 = time.perf_counter()
            with torch.no_grad():
                output = classifier(images_t)
            if device == "cuda": torch.cuda.synchronize()
            t_f1 = time.perf_counter()

            pred = float(torch.sigmoid(output["final_output"]).flatten().item())

            t_a1 = time.perf_counter()
            t_aligninfer += (t_a1 - t_a0)  # include align + forward

            for f_id in frame_ids:
                frame_res.setdefault(f_id, []).append(pred)
            preds.append(pred)
        util = mon.summary()

    # valutazione video identica alla demo
    video_score = float(np.mean(preds)) if len(preds)>0 else 0.0
    pred_label = int(video_score > optimal_threshold)

    # writer opzionale identico
    if write_vis and out_file is not None:
        scores = []
        boxes = []
        for frame_idx in range(len(frames)):
            if frame_idx in frame_res:
                pred_prob = float(np.mean(frame_res[frame_idx]))
                rect = frame_boxes[frame_idx]
            else:
                pred_prob = None
                rect = None
            scores.append(pred_prob); boxes.append(rect)
        SupplyWriter(video_path, out_file, optimal_threshold).run(frames, scores, boxes)

    # timings e FPS
    t_total = time.perf_counter() - t0
    n_frames = len(frames)
    n_clips  = len(clips_for_video)
    eps = 1e-9
    fps_end2end = n_frames / max(t_total, eps)
    fps_preproc = n_frames / max(t_detect, eps) if t_detect>0 else 0.0
    fps_model   = n_clips  / max(t_aligninfer, eps) if t_aligninfer>0 else 0.0

    peak_mem = torch.cuda.max_memory_allocated()/ (1024**3) if device=="cuda" else np.nan

    return {
        "video_score": video_score, "pred_label": pred_label,
        "frames": n_frames, "clips": n_clips,
        "t_total": t_total, "t_detect": t_detect, "t_aligninfer": t_aligninfer,
        "fps_end2end": fps_end2end, "fps_preproc": fps_preproc, "fps_model": fps_model,
        "cpu_mean": util["cpu_mean"], "gpu_mean": util["gpu_mean"],
        "gpu_mem_mean": util["gpu_mem_mean"], "gpu_mem_max": util["gpu_mem_max"],
        "gpu_mem_peak": peak_mem
    }

# ===== main: batch + metriche + timing =====
def main():
    ap = argparse.ArgumentParser("AltFreezing DEMO + valutazione e timing")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--cfg_path", default="i3d_ori.yaml")
    ap.add_argument("--ckpt_path", default="altfreezing/checkpoints/model.pth")
    ap.add_argument("--max_frame", type=int, default=400)
    ap.add_argument("--optimal_threshold", type=float, default=0.04)  # come demo
    ap.add_argument("--out_csv_pervideo", default="eval_outputs/per_video_demo1000.csv")
    ap.add_argument("--out_csv_summary", default="eval_outputs/summary_demo1000.csv")
    ap.add_argument("--cache_dir", default="eval_cache")
    ap.add_argument("--write_vis", action="store_true")
    ap.add_argument("--list_file", default=None)

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv_pervideo) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv_summary) or ".", exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # config + modello come demo
    cfg.init_with_yaml()
    cfg.update_with_yaml(args.cfg_path)
    cfg.freeze()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = PluginLoader.get_classifier(cfg.classifier_type)().to(device).eval()
    classifier.load(args.ckpt_path)
    crop_align_func = FasterCropAlignXRay(cfg.imsize)

    if args.list_file:
        vids = collect_videos_from_list(args.dataset_root, args.list_file)
    else:
        vids = collect_videos(args.dataset_root, per_class=500)

    if not vids:
        print("Nessun video trovato"); return

    header = ["video_path","gt_label","pred_label","correct","video_score","threshold","frames",
              "clips","t_total_s","t_detect_s","t_aligninfer_s",
              "fps_end2end","fps_preproc","fps_model",
              "cpu_mean","gpu_mean","gpu_mem_mean_GB","gpu_mem_max_GB","gpu_mem_peak_GB"]
    rows=[]
    y_true=[]; y_pred=[]; y_score=[]
    tp=tn=fp=fn=0

    # aggregati per summary
    agg = {k: [] for k in ["t_total","t_detect","t_aligninfer","fps_end2end","fps_preproc","fps_model",
                           "cpu_mean","gpu_mean","gpu_mem_mean","gpu_mem_max","gpu_mem_peak","frames","clips"]}

    for vpath, gt, dset, split in tqdm(vids, desc="Video", unit="vid"):
        res = eval_video_demo_timed(
            video_path=vpath,
            classifier=classifier,
            crop_align_func=crop_align_func,
            device=device,
            cfg_obj=cfg,
            max_frame=args.max_frame,
            optimal_threshold=args.optimal_threshold,
            cache_dir=args.cache_dir,
            write_vis=args.write_vis
        )
        pred = int(res["pred_label"]); score = float(res["video_score"])
        correct = int(pred==gt)
        y_true.append(gt); y_pred.append(pred); y_score.append(score)
        if   gt==1 and pred==1: tp+=1
        elif gt==0 and pred==0: tn+=1
        elif gt==0 and pred==1: fp+=1
        elif gt==1 and pred==0: fn+=1

        rows.append([
            vpath, gt, pred, correct, f"{score:.6f}", args.optimal_threshold, res["frames"],
            res["clips"], f"{res['t_total']:.6f}", f"{res['t_detect']:.6f}", f"{res['t_aligninfer']:.6f}",
            f"{res['fps_end2end']:.3f}", f"{res['fps_preproc']:.3f}", f"{res['fps_model']:.3f}",
            f"{res['cpu_mean']:.1f}" if res["cpu_mean"]==res["cpu_mean"] else "nan",
            f"{res['gpu_mean']:.1f}" if res["gpu_mean"]==res["gpu_mean"] else "nan",
            f"{res['gpu_mem_mean']:.3f}" if res["gpu_mem_mean"]==res["gpu_mem_mean"] else "nan",
            f"{res['gpu_mem_max']:.3f}" if res["gpu_mem_max"]==res["gpu_mem_max"] else "nan",
            f"{res['gpu_mem_peak']:.3f}" if res["gpu_mem_peak"]==res["gpu_mem_peak"] else "nan",
        ])

        for k in ["t_total","t_detect","t_aligninfer","fps_end2end","fps_preproc","fps_model",
                  "cpu_mean","gpu_mean","gpu_mem_mean","gpu_mem_max","gpu_mem_peak"]:
            agg[k].append(res[k if k not in ("gpu_mem_mean","gpu_mem_max","gpu_mem_peak") else k])
        agg["frames"].append(res["frames"])
        agg["clips"].append(res["clips"])

    # per-video CSV
    with open(args.out_csv_pervideo,"w",newline="") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

    # summary metriche classiche
    if SK_OK and y_true:
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score) if len(set(y_true))>1 else float('nan')
        ap  = average_precision_score(y_true, y_score)
        cm  = confusion_matrix(y_true, y_pred).tolist()
    else:
        acc=f1=auc=ap=float('nan'); cm=[[0,0],[0,0]]

    # summary timing
    import statistics as st
    def avg(x): 
        xs=[v for v in x if isinstance(v,(int,float)) and v==v]
        return float(sum(xs)/len(xs)) if xs else float('nan')
    def pctl(x, q):
        xs=sorted([v for v in x if isinstance(v,(int,float)) and v==v])
        if not xs: return float('nan')
        i=(len(xs)-1)*q; lo=int(np.floor(i)); hi=int(np.ceil(i))
        if lo==hi: return float(xs[lo])
        return float(xs[lo]*(hi-i)+xs[hi]*(i-lo))

    timing_header = ["avg_t_total_s","avg_t_detect_s","avg_t_aligninfer_s",
                     "p50_fps_end2end","p95_fps_end2end",
                     "avg_fps_preproc","avg_fps_model",
                     "avg_cpu_util","avg_gpu_util","avg_gpu_mem_GB","max_gpu_mem_GB","avg_gpu_mem_peak_GB",
                     "total_frames","total_clips"]
    timing_row = [
        f"{avg(agg['t_total']):.6f}",
        f"{avg(agg['t_detect']):.6f}",
        f"{avg(agg['t_aligninfer']):.6f}",
        f"{pctl(agg['fps_end2end'],0.50):.3f}",
        f"{pctl(agg['fps_end2end'],0.95):.3f}",
        f"{avg(agg['fps_preproc']):.3f}",
        f"{avg(agg['fps_model']):.3f}",
        f"{avg(agg['cpu_mean']):.1f}",
        f"{avg(agg['gpu_mean']):.1f}",
        f"{avg(agg['gpu_mem_mean']):.3f}",
        f"{pctl(agg['gpu_mem_max'],1.0):.3f}",
        f"{avg(agg['gpu_mem_peak']):.3f}",
        int(sum([x for x in agg['frames'] if isinstance(x,int)])),
        int(sum([x for x in agg['clips'] if isinstance(x,int)])),
    ]

    summary_header = ["videos","accuracy","auc_roc","pr_auc","f1","tp","tn","fp","fn","confusion_matrix"] + timing_header
    summary_row = [len(rows),
                   f"{acc:.6f}" if acc==acc else "nan",
                   f"{auc:.6f}" if auc==auc else "nan",
                   f"{ap:.6f}"  if ap==ap  else "nan",
                   f"{f1:.6f}"  if f1==f1  else "nan",
                   tp, tn, fp, fn, cm] + timing_row

    with open(args.out_csv_summary,"w",newline="") as f:
        w=csv.writer(f); w.writerow(summary_header); w.writerow(summary_row)

    print("Per-video CSV:", args.out_csv_pervideo)
    print("Riepilogo CSV:", args.out_csv_summary)
    print("Acc:", summary_row[1], "AUC:", summary_row[2], "PR-AUC:", summary_row[3], "F1:", summary_row[4])
    print("CM:", cm)
    print("FPS end-to-end p50/p95:", timing_row[3], timing_row[4])

if __name__ == "__main__":
    main()
