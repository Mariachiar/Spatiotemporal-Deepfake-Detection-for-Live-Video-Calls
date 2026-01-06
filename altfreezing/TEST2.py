#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AltFreezing evaluator con SINGLETON per tutti i componenti pesanti:
- Config, Classifier, CropAlign, YuNet, MediaPipe FaceMesh (coarse/refine), ByteTrack
- Una sola inizializzazione per processo. Per ogni video si fa solo reset dello stato volatile.
- Output: per-video.csv e summary.csv con Accuracy, AUC-ROC, PR-AUC, F1, CM, FPS, latenza clip, memoria, coerenza ID.

Dipendenze aggiuntive: scikit-learn, psutil
"""

import os, sys, glob, csv, math, time, argparse
import cv2, numpy as np, torch
from contextlib import nullcontext
import mediapipe as mp

# ----------------- Percorsi progetto -----------------
from tqdm import tqdm
import platform
try:
    import resource  # only on Unix
except ImportError:
    resource = None

ALT_DIR = os.path.abspath(os.path.dirname(__file__))          # .../deepfake/altfreezing
ROOT    = os.path.abspath(os.path.join(ALT_DIR, ".."))        # .../deepfake
for p in (ALT_DIR, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import config as cfg
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader
from preprocessing.ByteTrack.byte_tracker import BYTETracker, STrack
from preprocessing.ByteTrack.matching import iou_distance
from preprocessing.yunet.yunet import YuNet
from preprocessing.ByteTrack.basetrack import BaseTrack


# ----------------- Metriche opzionali -----------------
try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score, accuracy_score, confusion_matrix
    )
    SK_OK = True
except Exception:
    SK_OK = False

try:
    import psutil
    PS_OK = True
except Exception:
    PS_OK = False

# ----------------- Runtime safety -----------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
cv2.setNumThreads(1)

# ----------------- Singleton infra -----------------
class _Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        # una sola istanza per classe; i parametri della PRIMA init restano "bloccati"
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# ----------------- Helper -----------------
MP68_IDX = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,
            70,63,105,66,107,336,296,334,293,300,168,6,197,195,5,4,1,19,94,
            33,7,163,144,145,153,263,249,390,373,374,380,61,146,91,181,84,
            17,314,405,321,375,291,308,324,318,402,317,14,87,178,88]

def _opencv_has_cuda_dnn():
    try:
        dev_ok = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        dev_ok = False
    try:
        info = cv2.getBuildInformation()
        build_ok = ("CUDA:" in info and "YES" in info.split("CUDA")[1]) or ("NVIDIA CUDA:" in info and "YES" in info.split("NVIDIA CUDA")[1])
    except Exception:
        build_ok = False
    return dev_ok and build_ok

def _square_roi_from_tlbr(tlbr, roi_scale, W, H):
    x1, y1, x2, y2 = map(int, tlbr)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    side = int(max(x2-x1, y2-y1) * float(roi_scale)); side = max(2, side)
    rx1, ry1 = max(0, cx - side//2), max(0, cy - side//2)
    rx2, ry2 = min(W, rx1 + side), min(H, ry1 + side)
    side = min(rx2-rx1, ry2-ry1); rx2, ry2 = rx1 + side, ry1 + side
    return np.array([rx1, ry1, rx2, ry2], dtype=np.int32)

def facemesh_on_square_roi(frame_rgb, box_tlbr, roi_scale, mesh):
    H, W = frame_rgb.shape[:2]
    sroi = _square_roi_from_tlbr(box_tlbr, roi_scale, W, H)
    x1, y1, x2, y2 = map(int, sroi)
    roi = frame_rgb[y1:y2, x1:x2]
    if roi.size == 0: return None
    roi_c = np.ascontiguousarray(roi)
    res = mesh.process(roi_c)
    if not (res and res.multi_face_landmarks): return None
    h, w = roi.shape[:2]
    pts = np.array([[lm.x*w + x1, lm.y*h + y1] for lm in res.multi_face_landmarks[0].landmark], dtype=np.float32)
    lm68 = pts[MP68_IDX]
    lc, rc = lm68[36:42].mean(0), lm68[42:48].mean(0)
    nose, ml, mr = lm68[30], lm68[48], lm68[54]
    return {'lm5': np.vstack([lc, rc, nose, ml, mr]), 'lm68': lm68}

def variance_of_laplacian(img_rgb):
    return cv2.Laplacian(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

def human_bytes(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0: return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"



# ----------------- SINGLETON Services -----------------
class ConfigSvc(metaclass=_Singleton):
    def __init__(self, cfg_path):
        cfg.init_with_yaml(); cfg.update_with_yaml(cfg_path); cfg.freeze()
        self.cfg = cfg

class DeviceSvc(metaclass=_Singleton):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.backends.cudnn.benchmark = True

class ClassifierSvc(metaclass=_Singleton):
    def __init__(self, cfg_path, ckpt_path, amp=False, channels_last=False):
        self.device = DeviceSvc().device
        self.cfg = ConfigSvc(cfg_path).cfg
        self.model = PluginLoader.get_classifier(self.cfg.classifier_type)().to(self.device).eval()
        self.model.load(ckpt_path)
        self.amp_ctx = (lambda: torch.amp.autocast('cuda')) if (amp and self.device=='cuda') else nullcontext

        self.channels_last = channels_last
        self.mean = torch.tensor([0.485,0.456,0.406], device=self.device).view(1,3,1,1,1)*255
        self.std  = torch.tensor([0.229,0.224,0.225], device=self.device).view(1,3,1,1,1)*255


    def infer_scores(self, aligned_batch_bthwc):
    
        x = torch.as_tensor(aligned_batch_bthwc, dtype=torch.float32, device=self.device).permute(0,4,1,2,3)
        if self.channels_last and x.dim()==5:
            try: x = x.contiguous(memory_format=torch.channels_last_3d)
            except AttributeError: pass
        # niente x = x/255.0
        x = x.sub(self.mean).div(self.std)
        if not hasattr(self, "_pp_once"):
            self._pp_once = True
            xmn, xmx = float(x.min()), float(x.max())
            xm, xs = float(x.mean()), float(x.std(unbiased=False))
            print(f"[PP] after norm: mean={xm:.3f} std={xs:.3f} min={xmn:.1f} max={xmx:.1f}")


        with torch.inference_mode():
            with self.amp_ctx():
                out = self.model(x)

        # --- estrazione logits robusta ---
        def _get_logits(o):
            if torch.is_tensor(o):
                return o
            if isinstance(o, dict):
                for k in ('final_output','logits','cls','pred','y'):
                    v = o.get(k, None)
                    if torch.is_tensor(v):
                        return v
                # fallback: prima tensor nel dict
                for v in o.values():
                    if torch.is_tensor(v):
                        return v
            raise TypeError("Impossibile estrarre logits dal modello")

        t = _get_logits(out)
        if t.ndim == 1:
            t = t.unsqueeze(1)

        # probe una tantum
        if t.size(1) == 2 and not hasattr(self, "_cls_probe"):
            p = torch.softmax(t, dim=1).mean(0)
            print(f"[CLS] softmax mean col0={p[0].item():.4f} col1={p[1].item():.4f}")
            self._cls_probe = True

        # punteggi
        if t.size(1) == 1:
            scores = torch.sigmoid(t).squeeze(1).float().cpu().numpy()
        else:
            scores = torch.softmax(t, dim=1)[:, 1].float().cpu().numpy()


        self._last_scores = scores.copy()
        self._last_logits = t.detach().cpu() if t.size(1)==2 else None
        return scores


class CropAlignSvc(metaclass=_Singleton):
    def __init__(self, cfg_path):
        self.cfg = ConfigSvc(cfg_path).cfg
        self.crop_align = FasterCropAlignXRay(self.cfg.imsize)
    def __call__(self, infos, imgs):
        return self.crop_align(infos, imgs)

class YuNetSvc(metaclass=_Singleton):
    def __init__(self, detector_res=480, conf=0.8, nms=0.5, topK=5000):
        use_gpu = _opencv_has_cuda_dnn()
        backend, target = ((cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
                           if use_gpu else (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU))
        self.net = YuNet(
            modelPath=os.path.join('preprocessing','yunet','face_detection_yunet_2023mar.onnx'),
            inputSize=[detector_res, detector_res], confThreshold=conf, nmsThreshold=nms, topK=topK,
            backendId=backend, targetId=target
        )
    def set_frame_size(self, W, H):
        self.net.setInputSize((W, H))
    def infer(self, frame_bgr):
        return self.net.infer(frame_bgr)

class FaceMeshSvc(metaclass=_Singleton):
    def __init__(self, mp_min_det=0.35, mp_min_trk=0.35, refine_size=240):
        self.coarse = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                      refine_landmarks=False,
                                                      min_detection_confidence=mp_min_det,
                                                      min_tracking_confidence=mp_min_trk)
        self.refine = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                      refine_landmarks=True,
                                                      min_detection_confidence=mp_min_det,
                                                      min_tracking_confidence=mp_min_trk)
        self.refine_size = refine_size
    def pick(self, tlbr):
        bw = tlbr[2]-tlbr[0]; bh = tlbr[3]-tlbr[1]
        return self.refine if max(bw,bh) >= self.refine_size else self.coarse

class ByteTrackSvc(metaclass=_Singleton):
    def __init__(self, args_like, fps: float = 30.0):
        self.args = args_like
        self.fps = float(fps)
        self.tracker = BYTETracker(self.args, frame_rate=self.fps)
    def reset(self, fps: float | None = None):
        if fps is not None:
            self.fps = float(fps)
        self.tracker = BYTETracker(self.args, frame_rate=self.fps)
    def update(self, tracks_in, wh, wh2):
        return self.tracker.update(tracks_in, wh, wh2)



# ----------------- Runner per video (stateless a livello globale) -----------------
class VideoRunner:
    def __init__(self, args):
        self.args = args
        self.device = DeviceSvc().device
        self.cfg = ConfigSvc(args.cfg_path).cfg
        self.classifier = ClassifierSvc(args.cfg_path, args.ckpt_path, amp=args.amp, channels_last=args.channels_last)
        self.crop_align = CropAlignSvc(args.cfg_path)
        self.yunet = YuNetSvc(args.detector_res, args.conf, 0.3, 5000)
        self.facemesh = FaceMeshSvc(args.mp_min_det, args.mp_min_trk, args.refine_size)

        # placeholder: sarà aggiornato in run() col vero FPS
        self.bytetrack = ByteTrackSvc(args, fps=30.0)

        self.clip_size = int(args.clip_size) if args.clip_size > 0 else int(getattr(self.cfg, "clip_size", 32))
        print(f"[CFG] imsize={self.cfg.imsize} clip_size={self.clip_size} "
            f"crop_scale={self.args.crop_scale} roi_scale={self.args.roi_scale} "
            f"det_res={self.args.detector_res} detect_every={self.args.detect_every} mesh_every={self.args.mesh_every}")



    def _faces_valid(self, dets):
        if dets is None or len(dets)==0: return False
        for d in dets:
            x,y,w,h,score = d[:5]
            if score >= self.args.start_conf and max(w,h) >= self.args.start_min_size:
                return True
        return False

    def _frame_quality_weight(self, crop_rgb):
        if crop_rgb.size == 0: return 0.0
        h, w = crop_rgb.shape[:2]; min_side = min(h, w)
        small = crop_rgb if min_side <= 0 else cv2.resize(crop_rgb, (max(1,w//2), max(1,h//2)), interpolation=cv2.INTER_AREA)
        lap = variance_of_laplacian(small)
        if min_side < self.args.q_min_size_hard or lap < self.args.q_lap_hard: return 0.0
        if not self.args.q_weighting: return 1.0
        size_w = 1.0 if min_side >= self.args.q_min_size_soft else max(0.0, (min_side - self.args.q_min_size_hard) / max(1.0, (self.args.q_min_size_soft - self.args.q_min_size_hard)))
        lap_w  = 1.0 if lap >= self.args.q_lap_soft else max(0.0, (lap - self.args.q_lap_hard) / max(1e-6, (self.args.q_lap_soft - self.args.q_lap_hard)))
        if not hasattr(self, "_qstat"): self._qstat = []
        if len(self._qstat) < 50:
            h,w = crop_rgb.shape[:2]
            self._qstat.append((min(h,w), variance_of_laplacian(crop_rgb)))

        return float(size_w * lap_w)

    def run(self, video_path):
        # reset stato per video
        self.bytetrack.reset() 

        # reset anche del contatore globale di STrack (altrimenti i tid crescono tra video)
        if hasattr(STrack, "_count"):
                STrack._count = 0
        if hasattr(BaseTrack, "_count"):
            BaseTrack._count = 0

        cur_imgs, cur_infos, cur_w = {}, {}, {}
        track_clip_scores = {}
        batch_imgs, batch_infos, batch_tid = [], [], []
        clip_enqueue_t, clip_infer_t = {}, []

        prev_boxes = None; prev_ids = None; id_switches = 0; frames_seen = 0

        frames_processed = 0
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        fps_t0 = time.perf_counter()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None

        # FPS reale con sanity-check
        real_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if real_fps < 5 or real_fps > 240:
            real_fps = 30.0

        # aggiorna ByteTrack con FPS corretto
        self.bytetrack.reset(fps=real_fps)

        # ritmo detection: CLI se >0, altrimenti adattivo
        if getattr(self.args, "detect_every", 0) > 0:
            detect_every = int(self.args.detect_every)
        else:
            detect_every = max(1, min(8, round(real_fps / 8)))  # clamp a 8

        ok, first = cap.read()
        if not ok: return None

        H, W = first.shape[:2]
        self.yunet.set_frame_size(W, H)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        started, consec = (not self.args.smart_start), 0
        processed_after_start = 0
        last_lm = {}
        frame_idx = -1

        def enqueue_clip(tid):
            imgs = cur_imgs.get(tid, [])
            infos = cur_infos.get(tid, [])
            ws = cur_w.get(tid, [])
            if not imgs or not infos: return
            if len(imgs) < self.clip_size:
                need = self.clip_size - len(imgs)
                imgs += [imgs[-1]] * need
                infos += [infos[-1]] * need
                ws += [ws[-1]] * need
            fixed_infos = []
            for nb, lm5, lm68, big in infos:
                nb  = np.asarray(nb,  np.float32).reshape(4,)
                lm5 = np.asarray(lm5, np.float32).reshape(5,2)
                if lm68 is None or (isinstance(lm68, np.ndarray) and lm68.size == 0):
                    lm68 = np.zeros((68,2), np.float32)   # fallback unico, centralizzato
                else:
                    lm68 = np.asarray(lm68, np.float32).reshape(68,2)
                big = np.asarray(big, np.int32).reshape(4,)
                fixed_infos.append((nb, lm5, lm68, big))
            batch_imgs.append(list(imgs))
            batch_infos.append(fixed_infos)
            batch_tid.append(tid)
            # ---- finestra scorrevole + skip esplicito ----
            stride_frames = self.args.stride if getattr(self.args, "stride", 0) > 0 else max(1, self.clip_size // 2)

            if stride_frames >= self.clip_size:
                # nessun overlap
                keep_last = 0
            else:
                keep_last = self.clip_size - stride_frames

            cur_imgs[tid]  = cur_imgs[tid][-keep_last:]
            cur_infos[tid] = cur_infos[tid][-keep_last:]
            cur_w[tid]     = cur_w[tid][-keep_last:]


            clip_enqueue_t.setdefault(tid, []).append(time.perf_counter())

        def flush_and_infer():
            if not batch_imgs: return
            aligned_all, tids_used, enq_times = [], [], []
            target_T = self.clip_size
            target_HW = int(self.cfg.imsize)

            for infos, imgs, tid in zip(batch_infos, batch_imgs, batch_tid):
                try:
                    _, aligned = self.crop_align(infos, imgs)   # atteso (T,H,W,C)
                except Exception:
                    continue
                if not isinstance(aligned, np.ndarray) or aligned.ndim != 4:
                    continue

                # --- normalizza T (pad/trim) ---
                T, H, W, C = aligned.shape
                if T < target_T:
                    pad = np.repeat(aligned[-1:], target_T - T, axis=0)
                    aligned = np.concatenate([aligned, pad], axis=0)
                elif T > target_T:
                    aligned = aligned[:target_T]

                # --- normalizza H×W frame-wise ---
                if H != target_HW or W != target_HW:
                    out = np.empty((target_T, target_HW, target_HW, C), dtype=aligned.dtype)
                    for i in range(target_T):
                        out[i] = cv2.resize(aligned[i], (target_HW, target_HW), interpolation=cv2.INTER_LINEAR)
                    aligned = out

                # --- canale e contiguità ---
                if aligned.shape[-1] != 3:
                    continue
                aligned = np.ascontiguousarray(aligned, dtype=np.uint8)

                aligned_all.append(aligned); tids_used.append(tid)
                tlist = clip_enqueue_t.get(tid, []); enq_times.append(tlist.pop(0) if tlist else None)

            if not aligned_all:
                batch_imgs.clear(); batch_infos.clear(); batch_tid.clear(); return

            arr = np.stack(aligned_all, 0)  # (B,T,H,W,C) ora coerente
            scores = self.classifier.infer_scores(arr)
            for s, tid, t0c in zip(scores, tids_used, enq_times):
                track_clip_scores.setdefault(tid, []).append(float(s))
                if t0c is not None: clip_infer_t.append((time.perf_counter() - t0c) * 1000.0)

            batch_imgs.clear(); batch_infos.clear(); batch_tid.clear()


        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        default_budget = self.args.max_frame if self.args.max_frame > 0 else int(real_fps*20)

        if total_frames > 0:
            stride = self.args.stride if getattr(self.args, "stride", 0) > 0 else max(1, self.clip_size // 2)

            max_start = max(0, total_frames - self.clip_size)
            starts = list(range(0, max_start + 1, stride))

            # rispetta il budget: numero massimo di finestre processabili
            max_windows = max(1, default_budget // max(1, self.clip_size))
            if len(starts) > max_windows:
                idxs = np.linspace(0, len(starts) - 1, max_windows, dtype=int)
                starts = [starts[i] for i in idxs]

            keep_ranges = [(s, min(s + self.clip_size - 1, total_frames - 1)) for s in starts]

            def window_id(i: int) -> int:
                for k,(lo,hi) in enumerate(keep_ranges):
                    if lo <= i <= hi:
                        return k
                return -1

            last_win = {}  # tid -> indice finestra già usata


            budget_frames = sum(hi - lo + 1 for (lo, hi) in keep_ranges)


            def keep_frame(i: int) -> bool:
                return any(lo <= i <= hi for (lo, hi) in keep_ranges)
        else:
            def keep_frame(i: int) -> bool:
                return True
            budget_frames = default_budget
            # aggiungi:
            def window_id(i: int) -> int:
                return -1
            last_win = {}


        pbar = tqdm(total=total_frames, desc=f"Frames [{os.path.basename(video_path)}]", unit="f")


        while True:
            ok, fbgr = cap.read()
            if not ok: break
            frame_idx += 1
            pbar.update(1)

            if not keep_frame(frame_idx):
                continue

        
            need_detect = (
                not started
                or (frame_idx % detect_every == 0)
                or (self.bytetrack.tracker.tracked_stracks == [])
            )


            dets = self.yunet.infer(fbgr) if need_detect else None

            if not started:
                consec = consec + 1 if (dets is not None and self._faces_valid(dets)) else 0
                if consec >= self.args.start_after_n: started = True
                continue

            if processed_after_start >= budget_frames: break

            processed_after_start += 1
            frames_processed += 1

            det_tlbr, dets_np, tracks_in = [], None, []
            if dets is not None and len(dets)>0:
                dets_np = np.asarray(dets, dtype=np.float32)

                # Filtri anti micro-volti / fascia bassa
                w = dets_np[:,2]; h = dets_np[:,3]; y = dets_np[:,1]
                mask = np.ones(len(dets_np), dtype=bool)
                if self.args.min_det_side > 0:
                    mask &= (np.maximum(w, h) >= self.args.min_det_side)
                if self.args.min_det_area and self.args.min_det_area > 0:
                    mask &= ((w * h) >= self.args.min_det_area)
                if self.args.exclude_bottom_frac and self.args.exclude_bottom_frac > 0:
                    cutoff = H * (1.0 - float(self.args.exclude_bottom_frac))
                    cy = y + 0.5*h
                    mask &= (cy < cutoff)

                dets_np = dets_np[mask]
                for d in dets_np:
                    x,y,w,h,score = d[:5]
                    tlbr = np.array([x,y,x+w,y+h], dtype=np.float32)
                    tracks_in.append(STrack(tlbr, score=float(score)))
                    det_tlbr.append(tlbr)
            det_tlbr = np.asarray(det_tlbr, dtype=np.float32) if det_tlbr else None
            online_tracks = self.bytetrack.update(tracks_in, (H, W), (H, W))


            frgb = None

            # ID coherence
            cur_boxes = []; cur_ids = []
            for tr in online_tracks or []:
                bw = tr.tlbr[2] - tr.tlbr[0]; bh = tr.tlbr[3] - tr.tlbr[1]
                if max(bw, bh) < self.args.min_track_side:
                    continue

                cur_boxes.append(tr.tlbr.astype(np.float32)); cur_ids.append(tr.track_id)
            if len(cur_boxes)>0:
                cur_boxes = np.stack(cur_boxes,0)
                if prev_boxes is not None and prev_ids is not None:
                    dist = iou_distance(prev_boxes, cur_boxes)
                    for i_prev in range(prev_boxes.shape[0]):
                        j = int(np.argmin(dist[i_prev])); iou = 1.0 - float(dist[i_prev, j])
                        if iou >= 0.5 and prev_ids[i_prev] != cur_ids[j]: id_switches += 1
                prev_boxes, prev_ids = cur_boxes, cur_ids
            frames_seen += 1

            for tr in online_tracks or []:
                bw = tr.tlbr[2] - tr.tlbr[0]; bh = tr.tlbr[3] - tr.tlbr[1]
                if max(bw, bh) < self.args.min_track_side:
                    continue

                tid = tr.track_id
                if tid not in cur_imgs: cur_imgs[tid], cur_infos[tid], cur_w[tid] = [], [], []

                yunet_lm5 = None
                if det_tlbr is not None and len(det_tlbr)>0:
                    ious = 1.0 - iou_distance(np.array([tr.tlbr], dtype=np.float32), det_tlbr)[0]
                    k = int(np.argmax(ious))
                    if ious[k] >= 0.4:
                        yunet_lm5 = dets_np[k][5:15].reshape(5,2)

                fm = None
                if (frame_idx % self.args.mesh_every)==0 or (tid not in last_lm):
                    if frgb is None: frgb = cv2.cvtColor(fbgr, cv2.COLOR_BGR2RGB)
                    mesh = self.facemesh.pick(tr.tlbr)
                    fm = facemesh_on_square_roi(frgb, tr.tlbr, self.args.roi_scale, mesh)
                    if fm is None and yunet_lm5 is not None: fm = {'lm5': yunet_lm5, 'lm68': None}
                    if fm is not None: last_lm[tid] = {**fm, 'frame_idx': frame_idx}
                else:
                    cached = last_lm.get(tid)
                    if cached is not None: fm = {'lm5': cached['lm5'], 'lm68': cached['lm68']}
                    elif yunet_lm5 is not None: fm = {'lm5': yunet_lm5, 'lm68': None}

                if fm is None: 
                    continue
                use_lm68 = (fm['lm68'] is not None)

                if frgb is None: frgb = cv2.cvtColor(fbgr, cv2.COLOR_BGR2RGB)
                big = get_crop_box((H, W), tr.tlbr, scale=self.args.crop_scale)
                x1,y1,x2,y2 = map(int, big)
                if x2<=x1 or y2<=y1: continue

                crop_rgb = frgb[y1:y2, x1:x2]
                wq = self._frame_quality_weight(crop_rgb)
                if wq > 0.0:
                    top_left = np.array([[x1, y1]], dtype=np.float32)
                    new_box = (tr.tlbr.reshape(2,2).astype(np.float32) - top_left).reshape(-1)
                    lm5 = fm['lm5'].astype(np.float32) - top_left
                    lm68 = (fm['lm68'].astype(np.float32) - top_left) if use_lm68 else None
                    cur_infos[tid].append(
                        (new_box, lm5, (lm68 if lm68 is not None else np.empty((0,2), np.float32)),
                        np.array([x1,y1,x2,y2], dtype=np.int32)) )
                    cur_imgs[tid].append(crop_rgb)
                    #cur_infos[tid].append((new_box, lm5, lm68, np.array([x1,y1,x2,y2], dtype=np.int32)))
                    cur_w[tid].append(wq)
                wid = window_id(frame_idx)
                if len(cur_imgs[tid]) >= self.clip_size:
                    if wid != -1 and last_win.get(tid) != wid:
                        enqueue_clip(tid)
                        last_win[tid] = wid
                        # azzera il buffer per NON produrre più clip nella stessa finestra
                        cur_imgs[tid].clear(); cur_infos[tid].clear(); cur_w[tid].clear()
                        if len(batch_imgs) >= self.args.batch_clips:
                            flush_and_infer()


        flush_and_infer()

        cap.release()
        pbar.close()


        def score_with_stability(scores, base):
            s = np.asarray(scores, float)
            if s.size==0: return 0.0
            iqr = np.percentile(s,85)-np.percentile(s,25)
            # penalizza solo se la serie è instabile e la mediana non è già molto alta
            if iqr > 0.25 and np.median(s) < 0.85:
                return base * (0.85 ** (iqr/0.25))  # freno dolce ai falsi picchi
            return base

        def _pool_track(scores, method="median", topk_ratio=0.2, percentile_p=80.0, trim_ratio=0.2):
            s = np.asarray(scores, float)
            if s.size == 0:
                return 0.0

            if method == "mean":
                return float(np.mean(s))

            if method == "median":
                return float(np.median(s))

            if method == "logit_median":
                se = np.clip(s, 1e-6, 1-1e-6)
                med = np.median(np.log(se/(1-se)))
                return float(1/(1+np.exp(-med)))

            if method == "topk":
                k = max(1, int(np.ceil(topk_ratio * s.size)))
                return float(np.mean(np.sort(s)[-k:]))

            if method == "topk_median":
                k = max(1, int(np.ceil(topk_ratio * s.size)))
                return float(np.median(np.sort(s)[-k:]))

            if method == "percentile":
                p = float(np.clip(percentile_p, 0.0, 100.0))
                return float(np.percentile(s, p))

            if method == "trimmed_mean":
                t = float(np.clip(trim_ratio, 0.0, 0.49))
                ss = np.sort(s)
                n = ss.size
                a = int(n * t)
                b = max(a+1, n - a)
                return float(np.mean(ss[a:b]))

            if method == "adaptive":
                # se distribuzione stretta → percentile alto; altrimenti logit_median
                iqr = np.percentile(s,75) - np.percentile(s,25)
                if iqr < 0.15:
                    p = float(np.clip(percentile_p, 0.0, 100.0))
                    return float(np.percentile(s, p))
                se = np.clip(s, 1e-6, 1-1e-6)
                med = np.median(np.log(se/(1-se)))
                return float(1/(1+np.exp(-med)))

            # fallback
            return float(np.median(s))
        
        q_med_minSide = q_med_lap = None
        low_quality = False
        if hasattr(self, "_qstat") and self._qstat:
            ms = np.array(self._qstat, float)
            q_med_minSide = float(np.median(ms[:,0]))
            q_med_lap     = float(np.median(ms[:,1]))
            low_quality = (q_med_minSide < self.args.qa_min_side) or (q_med_lap < self.args.qa_min_lap)
            print(f"[QA] low_quality={low_quality} (med_minSide={q_med_minSide:.1f}, med_lap={q_med_lap:.1f}, "
                f"thr_side={self.args.qa_min_side}, thr_lap={self.args.qa_min_lap})")
            self._qstat.clear()


        POOL_METH = getattr(self.args, "pool_method", "median")
        TOPK_R    = getattr(self.args, "topk_ratio", 0.2)
        PERC_P    = getattr(self.args, "percentile_p", 80.0)
        TRIM_R    = getattr(self.args, "trim_ratio", 0.2)
        MIN_CLIPS = getattr(self.args, "min_clips", 1)

        raw_scores = {}
        per_person = {}
        for tid, scores in track_clip_scores.items():
            if len(scores) < MIN_CLIPS:
                continue
            raw = _pool_track(scores, method=POOL_METH, topk_ratio=TOPK_R,
                            percentile_p=PERC_P, trim_ratio=TRIM_R)
            # eventualmente disattiva la penalità (vedi fix 4)
            pen = score_with_stability(scores, raw) if not getattr(self.args, "disable_penalty", False) else raw

            raw_scores[tid] = float(raw)
            per_person[tid] = float(pen)

            if getattr(self.args, "verbose_pool", False):
                print(f"[POOL] tid={tid} raw={raw:.4f} penalized={pen:.4f} clips={len(scores)}")



        thr = getattr(self.args, "optimal_threshold", 0.0)
        track_quants = {}
        for tid, ss in track_clip_scores.items():
            s = np.array(ss, float)
            if s.size:
                qs = np.percentile(s, [10,25,50,75,90])
                track_quants[tid] = {"q10":qs[0], "q25":qs[1], "q50":qs[2], "q75":qs[3], "q90":qs[4]}
                print(f"[SCORE] tid={tid} q10/25/50/75/90={qs.round(4)}")


        per_person_labels_std = {tid: int((per_person.get(tid, raw_scores[tid]) > thr))
                         for tid in per_person.keys()}

        # regola extra per qualità bassa (q75/q90)
        per_person_labels_qa = {}
        if low_quality:
            for tid in per_person.keys():
                q = track_quants.get(tid, None)
                qa_hit = bool(q and (q["q75"] >= self.args.qa_q75_thr or q["q90"] >= self.args.qa_q90_thr))
                per_person_labels_qa[tid] = int(qa_hit)
        else:
            per_person_labels_qa = {tid:0 for tid in per_person.keys()}

        # fuse: positivo se supera soglia standard OPPURE regola QA
        per_person_labels = {tid: int(per_person_labels_std.get(tid,0) or per_person_labels_qa.get(tid,0))
                            for tid in per_person.keys()}

        video_fake = any(v==1 for v in per_person_labels.values())
        video_score = float(max(raw_scores.values())) if raw_scores else 0.0  # invariato (AUC)



        elapsed = time.perf_counter() - t0
        fps = frames_processed / max(1e-6, (time.perf_counter() - fps_t0))
        lat_ms = (sum(clip_infer_t)/len(clip_infer_t)) if clip_infer_t else float('nan')
        if resource and platform.system() != "Windows":
            ru = resource.getrusage(resource.RUSAGE_SELF)
            if platform.system() == "Darwin":
                cpu_peak_mb = ru.ru_maxrss / (1024*1024)   # bytes → MB
            else:
                cpu_peak_mb = ru.ru_maxrss / 1024.0
        else:
            if PS_OK:
                p = psutil.Process(os.getpid())
                mi = p.memory_info()
                # su Windows preferisci il picco se disponibile
                peak = getattr(mi, "peak_wset", None) or getattr(mi, "peak_pagefile", None)
                base = peak if peak else mi.rss  # bytes
                cpu_peak_mb = base / (1024*1024)
            else:
                cpu_peak_mb = float('nan')
        nclips = sum(len(v) for v in track_clip_scores.values())
        print(f"[CLIPS] tracks={len(track_clip_scores)} total_clips={nclips} clips_per_track={[len(v) for v in track_clip_scores.values()]}")



        id_switch_rate = (id_switches / max(1, frames_seen)) * 1000.0
        gpu_peak_alloc_mb   = (torch.cuda.max_memory_allocated()/1024/1024) if torch.cuda.is_available() else float('nan')
        gpu_peak_reserved_mb= (torch.cuda.max_memory_reserved()/1024/1024)   if torch.cuda.is_available() else float('nan')

        return {
            "video_path": video_path,
            "frames_processed": frames_processed,
            "elapsed_s": elapsed,
            "fps": fps,
            "latency_ms_clip_mean": lat_ms,
            "video_score": video_score,
            "pred_label": int(video_fake),
            "num_tracks": len(per_person),
            "id_switch_rate_per_1k_frames": id_switch_rate,
            "gpu_mem_alloc_peak_mb": gpu_peak_alloc_mb,
            "gpu_mem_reserved_peak_mb": gpu_peak_reserved_mb,
            "cpu_mem_peak_mb": cpu_peak_mb,
            "per_person_scores": per_person,
            "per_person_labels": per_person_labels

        }

# ----------------- Dataset scan + metriche -----------------
import os, glob, random

def collect_videos(root_dir, per_class=250):
    """
    Scansiona root_dir (sottocartelle incluse), segue symlink, 
    classifica per path e restituisce max `per_class` video per classe.
    Ritorna: list[(video_path, label, dataset, subset)]
    """
    exts = (".mp4",".avi",".mov",".mkv")
    REAL_TOK = ("/original/", "/original_sequences/", "/celeb-real/", "/youtube-real/", "/real/", "/source/")
    FAKE_TOK = (
        "/target/","/manipulated_sequences/","/deepfakes/","/face2face/","/faceswap/","/neuraltextures/","/fake/","/celeb-synthesis/",
        "/faceshifter/","/deepfakedetection/","/dfd/"
    )
    DATASETS_HINT = ("ffpp","ffiw","celebdf_v2","faceforensics++","faceforensics","celebdf")
    SUBSETS_HINT = ("train","val","test","c23","c40")

    def is_video(p):
        pl = p.lower()
        return pl.endswith(exts)

    def classify(p):
        pl = p.replace("\\","/").lower()
        if any(t in pl for t in REAL_TOK): return 0
        if any(t in pl for t in FAKE_TOK): return 1
        return None  # ignora se non riconoscibile

    def dataset_name(parts_lower):
        for s in DATASETS_HINT:
            if s in parts_lower: return s
        # fallback FF++
        if any(x in parts_lower for x in ("deepfakes","face2face","faceswap","neuraltextures","original","original_sequences")):
            return "ffpp"
        return "unknown"

    def subset_name(parts_lower):
        for s in SUBSETS_HINT:
            if s in parts_lower: return s
        return "unknown"

    seen = set()
    pool_real, pool_fake = [], []

    # pattern FF++ specifici + scan generico
    patterns = []
    for e in ("*.mp4","*.avi","*.mov","*.mkv"):
        # FF++ classico
        patterns += [
            os.path.join(root_dir, "**", "original_sequences", "**", "c*", "videos", e),
            os.path.join(root_dir, "**", "manipulated_sequences", "**", "c*", "videos", e),
        ]
        # generico
        patterns += [os.path.join(root_dir, "**", e)]

    for pat in patterns:
        for v in glob.iglob(pat, recursive=True):
            real_v = os.path.realpath(v)
            if real_v in seen: 
                continue
            seen.add(real_v)
            if not os.path.exists(real_v) or not is_video(real_v):
                continue
            lab = classify(real_v)
            if lab is None:
                continue
            parts_lower = [x.lower() for x in real_v.replace("\\","/").split("/")]
            dset = dataset_name(parts_lower)
            subset = subset_name(parts_lower)
            item = (real_v, lab, dset, subset)
            if lab == 0:
                pool_real.append(item)
            else:
                pool_fake.append(item)

    # campionamento deterministico
    rng = random.Random(0)
    rng.shuffle(pool_real)
    rng.shuffle(pool_fake)
    pick_real = pool_real[:per_class] if per_class else pool_real
    pick_fake = pool_fake[:per_class] if per_class else pool_fake

    # interleave per bilanciare l’ordine
    out = []
    for a, b in zip(pick_real, pick_fake):
        out.append(a); out.append(b)
    m = min(len(pick_real), len(pick_fake))
    out = [x for pair in zip(pick_real[:m], pick_fake[:m]) for x in pair]
    out += pick_real[m:]
    out += pick_fake[m:]


    return out

def _infer_label_from_path(p):
    pl = p.replace("\\","/").lower()
    REAL_TOK = ("/original/","/original_sequences/","/celeb-real/","/youtube-real/","/real/", "/source/")
    FAKE_TOK = (
        "/manipulated_sequences/","/deepfakes/","/face2face/","/faceswap/","/neuraltextures/","/fake/","/celeb-synthesis/",
        "/faceshifter/","/deepfakedetection/","/dfd/", "/target/"
    )
    if any(t in pl for t in REAL_TOK): return 0
    if any(t in pl for t in FAKE_TOK): return 1
    return None

def _dataset_subset_from_path(p):
    parts_lower = [x.lower() for x in p.replace("\\","/").split("/")]
    dsets = ("ffpp","ffiw","celebdf_v2","faceforensics++","faceforensics","celebdf")
    subs  = ("train","val","test","c23","c40")
    dset = next((s for s in dsets if s in parts_lower), "ffpp"
                if any(x in parts_lower for x in ("deepfakes","face2face","faceswap","neuraltextures","original","original_sequences"))
                else "unknown")
    subset = next((s for s in subs if s in parts_lower), "unknown")
    return dset, subset

def read_list_file(list_path, root_for_relative=None):
    out = []
    base_dir = os.path.dirname(os.path.realpath(list_path))
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # --- parsing flessibile: "path[,label]" OPPURE "label path" ---
            p, lab = line, None
            if "," in line:  # es. "YouTube-real/00170.mp4,1"
                p, lab = line.rsplit(",", 1)
                p = p.strip(); lab = lab.strip()
                lab = int(lab) if lab.isdigit() else None
            else:
                parts = line.split()
                if parts and parts[0] in {"0","1"} and len(parts) >= 2:  # es. "1 YouTube-real/00170.mp4"
                    lab = int(parts[0])
                    p = " ".join(parts[1:]).strip()
                else:
                    p = line  # es. "YouTube-real/00170.mp4"

            # --- risoluzione path relativi: prima dataset_root, poi cartella del txt ---
            candidates = [p]
            if not os.path.isabs(p):
                if root_for_relative:
                    candidates.append(os.path.join(root_for_relative, p))
                candidates.append(os.path.join(base_dir, p))

            p_real = next((os.path.realpath(c) for c in candidates if os.path.exists(c)), None)
            if not p_real:
                continue

            if lab is None:
                lab = _infer_label_from_path(p_real)
            if lab is None:
                continue

            dset, subset = _dataset_subset_from_path(p_real)
            out.append((p_real, lab, dset, subset))
    return out




def main():
    ap = argparse.ArgumentParser("AltFreezing batch eval (singleton)")
    # IO
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--out_csv_pervideo", default="eval_outputs/per_video.csv")
    ap.add_argument("--out_csv_summary", default="eval_outputs/summary.csv")
    ap.add_argument("--list_path", default=None,
                help="File con uno per riga: path[,label]. Se la label manca, è inferita dal path.")

    # Model
    ap.add_argument("--cfg_path", default="i3d_ori.yaml")
    ap.add_argument("--ckpt_path", default="altfreezing/checkpoints/ft_ffpp_e3_f6.pt")
    ap.add_argument("--optimal_threshold", type=float, default=0.4)
    # Limits
    ap.add_argument("--max_frame", type=int, default=400)
    ap.add_argument("--clip_size", type=int, default=32)
    ap.add_argument("--crop_scale", type=float, default=1.0)
    # Detect/track
    ap.add_argument("--detector_res", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--track_thresh", type=float, default=0.6)
    ap.add_argument("--track_buffer", type=int, default=2000)
    ap.add_argument("--match_thresh", type=float, default=0.6)
    ap.add_argument("--detect_every", type=int, default=3,
                help="Fissa la frequenza di detection in frame. 0 = adattivo (~fps/8).")
    ap.add_argument("--dump_clip_scores", action="store_true")
    ap.add_argument("--dump_track_scores", action="store_true")
    ap.add_argument("--stride", type=int, default=5, help="Stride fra finestre; 0 = clip_size//2")



    # Smart start
    ap.add_argument("--smart_start", action="store_true", default=False)
    ap.add_argument("--start_after_n", type=int, default=5)
    ap.add_argument("--start_conf", type=float, default=0.6)
    ap.add_argument("--start_min_size", type=int, default=20)

    # FaceMesh
    ap.add_argument("--roi_scale", type=float, default=2.0)
    ap.add_argument("--mp_min_det", type=float, default=0.35)
    ap.add_argument("--mp_min_trk", type=float, default=0.35)
    ap.add_argument("--mesh_every", type=int, default=3)
    ap.add_argument("--refine_size", type=int, default=224)
    # Quality weighting
    ap.add_argument("--q_weighting", action="store_true", default=True)
    ap.add_argument("--q_min_size_soft", type=int, default=72)
    ap.add_argument("--q_min_size_hard", type=int, default=48)
    ap.add_argument("--q_lap_soft", type=float, default=24)
    ap.add_argument("--q_lap_hard", type=float, default=8)
    # Inference
    ap.add_argument("--batch_clips", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--channels_last", action="store_true")
    ap.add_argument("--pool_method",
        choices=["median","mean","topk","topk_median","percentile","trimmed_mean","logit_median","adaptive"],
        default="median")
    ap.add_argument("--percentile_p", type=float, default=70.0,
        help="Percentile per pool_method=percentile o adaptive (default 80)")
    ap.add_argument("--trim_ratio", type=float, default=0.2,
        help="Trim per trimmed_mean, frazione tagliata in testa e coda (default 0.2)")

    ap.add_argument("--topk_ratio", type=float, default=0.3)
    ap.add_argument("--min_clips", type=int, default=1)
    ap.add_argument("--verbose_pool", action="store_true")
    ap.add_argument("--disable_penalty", action="store_true",
                help="Se attivo, non applica la penalità di stabilità sui punteggi per-persona")
    
    ap.add_argument("--qa_min_side", type=int, default=240,
                help="Se mediana(minSide) < qa_min_side ⇒ qualità bassa")
    ap.add_argument("--qa_min_lap", type=float, default=16.0,
                    help="Se mediana(Laplacian) < qa_min_lap ⇒ qualità bassa")
    ap.add_argument("--qa_q75_thr", type=float, default=0.30,
                    help="Soglia q75 per qualità bassa")
    ap.add_argument("--qa_q90_thr", type=float, default=0.90,
                    help="Soglia q90 per qualità bassa")

    ap.add_argument("--min_det_side", type=int, default=80,
                help="scarta rilevamenti con max(w,h) < X px")
    ap.add_argument("--min_det_area", type=int, default=4000,
                    help="scarta rilevamenti con area < X px^2 (0 = disattivo)")
    ap.add_argument("--exclude_bottom_frac", type=float, default=0.10,
                    help="ignora rilevamenti nel bordo inferiore: frazione di H (0=none)")
    ap.add_argument("--min_track_side", type=int, default=96,
                    help="non enqueuare clip per track con faccia troppo piccola")







    args = ap.parse_args()

    args.mot20 = False
    os.makedirs(os.path.dirname(args.out_csv_pervideo) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv_summary) or ".", exist_ok=True)

    # dimensione checkpoint
    try:
        model_size_bytes = os.path.getsize(args.ckpt_path)
    except Exception:
        model_size_bytes = 0

    videos = read_list_file(args.list_path, root_for_relative=args.dataset_root) \
          if args.list_path else collect_videos(args.dataset_root)
    if not videos:
        print("Nessun video trovato."); return

    runner = VideoRunner(args)  # usa tutti i singleton

    header = [
        "video_path","dataset","subset","gt_label","pred_label","correct",
        "video_score","threshold",
        "frames_processed","elapsed_s","fps","latency_ms_clip_mean",
        "num_tracks","id_switch_rate_per_1k_frames",
        "gpu_mem_alloc_peak_mb","gpu_mem_reserved_peak_mb","cpu_mem_peak_mb","model_size"
    ]


    y_true,y_pred,y_score = [],[],[]
    tp=tn=fp=fn=0
    rows=[]
    
    probe_scores = {"col0_real":[], "col0_fake":[], "col1_real":[], "col1_fake":[]}

    for vpath, gt, dset, split in tqdm(videos, desc="Video", unit="vid"):
        res = runner.run(vpath)
        if res is None: continue
        pred = int(res["pred_label"]); score = float(res["video_score"])
        correct = int(pred==gt)
        y_true.append(gt); y_pred.append(pred); y_score.append(score)
        if   gt==1 and pred==1: tp+=1
        elif gt==0 and pred==0: tn+=1
        elif gt==0 and pred==1: fp+=1
        elif gt==1 and pred==0: fn+=1
        rows.append([
            vpath, dset, split, gt, pred, correct,
            f"{score:.6f}", args.optimal_threshold,
            res["frames_processed"], f"{res['elapsed_s']:.3f}", f"{res['fps']:.3f}",
            (f"{res['latency_ms_clip_mean']:.3f}" if not math.isnan(res['latency_ms_clip_mean']) else "nan"),
            res["num_tracks"], f"{res['id_switch_rate_per_1k_frames']:.3f}",
            (f"{res['gpu_mem_alloc_peak_mb']:.1f}" if not math.isnan(res['gpu_mem_alloc_peak_mb']) else "nan"),
            (f"{res['gpu_mem_reserved_peak_mb']:.1f}" if not math.isnan(res['gpu_mem_reserved_peak_mb']) else "nan"),
            (f"{res['cpu_mem_peak_mb']:.1f}" if not math.isnan(res['cpu_mem_peak_mb']) else "nan"),
            human_bytes(model_size_bytes)
        ])
    
        if hasattr(runner.classifier, "_last_logits") and runner.classifier._last_logits is not None:
            # prendi la media clip->track->video del PRIMO track usato per la decisione
            logit = runner.classifier._last_logits  # salva nella infer_scores (vedi sotto)
            s = torch.softmax(logit, dim=1).mean(0).cpu().numpy()
            if gt==0:  # real
                probe_scores["col0_real"].append(float(s[0])); probe_scores["col1_real"].append(float(s[1]))
            else:
                probe_scores["col0_fake"].append(float(s[0])); probe_scores["col1_fake"].append(float(s[1]))
    def _m(a): return (sum(a)/max(1,len(a))) if a else float("nan")
    print("[DIAG] mean col0 | real/fake =", _m(probe_scores["col0_real"]), _m(probe_scores["col0_fake"]))
    print("[DIAG] mean col1 | real/fake =", _m(probe_scores["col1_real"]), _m(probe_scores["col1_fake"]))

    with open(args.out_csv_pervideo, "w", newline="") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

    # summary
    if SK_OK and y_true:
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score) if len(set(y_true))>1 else float('nan')
        ap  = average_precision_score(y_true, y_score)
        cm  = confusion_matrix(y_true, y_pred).tolist()
    else:
        acc=f1=auc=ap=float('nan'); cm=[[0,0],[0,0]]

    mean_fps = np.nanmean([float(r[10]) for r in rows]) if rows else float('nan')
    mean_lat = np.nanmean([float(r[11]) if r[11]!="nan" else np.nan for r in rows]) if rows else float('nan')

    summary_header = [
        "videos","accuracy","auc_roc","pr_auc","f1",
        "tp","tn","fp","fn","confusion_matrix","mean_fps","mean_latency_ms_clip",
        "model_size"
    ]
    summary_row = [
        len(rows),
        f"{acc:.6f}" if not math.isnan(acc) else "nan",
        f"{auc:.6f}" if not math.isnan(auc) else "nan",
        f"{ap:.6f}"  if not math.isnan(ap)  else "nan",
        f"{f1:.6f}"  if not math.isnan(f1)  else "nan",
        tp, tn, fp, fn, cm,
        f"{mean_fps:.3f}" if not math.isnan(mean_fps) else "nan",
        f"{mean_lat:.3f}" if not math.isnan(mean_lat) else "nan",
        human_bytes(model_size_bytes)
    ]
    with open(args.out_csv_summary, "w", newline="") as f:
        w=csv.writer(f); w.writerow(summary_header); w.writerow(summary_row)

    print("Per-video CSV:", args.out_csv_pervideo)
    print("Riepilogo CSV:", args.out_csv_summary)
    print("Acc:", summary_row[1], "AUC:", summary_row[2], "PR-AUC:", summary_row[3], "F1:", summary_row[4])
    print("CM:", cm)
    print("FPS medio:", summary_row[10], "Latency clip ms:", summary_row[11])

if __name__ == "__main__":
    main()
