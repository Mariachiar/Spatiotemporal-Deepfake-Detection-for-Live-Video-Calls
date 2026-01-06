#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estrazione penultimate features AltFreezing a memoria ridotta.
- Streaming frame, nessun buffer di crop full-res.
- Allineamento per-frame → si memorizzano solo frame allineati (uint8, HxW).
- Inferenza clip-per-clip, AMP opzionale, salvataggio FP16.
Output per clip: out_dir/<rel_path>/<video_name>/<video_name>_tid<id>_c<k>.npz
Contiene: feat(fp16), logits(fp16), score(fp32), y, tid, clip_idx, video_rel
"""

import os, sys, glob, argparse, collections
import numpy as np, cv2, torch, torch.nn as nn

# --------- path progetto (adatta se serve) ----------
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, ROOT):
    if p not in sys.path: sys.path.insert(0, p)

from config import config as cfg
from utils.plugin_loader import PluginLoader
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.utils import get_crop_box

from preprocessing.yunet.yunet import YuNet
from preprocessing.ByteTrack.byte_tracker import BYTETracker, STrack
from preprocessing.ByteTrack.basetrack import BaseTrack

# --------- runtime snello ----------
cv2.setNumThreads(1)
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")

VIDEO_EXTS = (".mp4",".avi",".mov",".mkv")
REAL_TOK   = ("/class0/", "/original/", "/original_sequences/", "/youtube-real/", "/celeb-real/")
FAKE_TOK   = ("/class1/", "/manipulated_sequences/", "/deepfakes/", "/face2face/", "/faceswap/", "/neuraltextures/")

def infer_label_from_path(p: str):
    q = p.replace("\\","/").lower()
    if any(t in q for t in REAL_TOK): return 0
    if any(t in q for t in FAKE_TOK): return 1
    return -1

def video_iter(root):
    for p in glob.iglob(os.path.join(root, "**", "*"), recursive=True):
        if os.path.isfile(p) and p.lower().endswith(VIDEO_EXTS):
            yield os.path.realpath(p)

def opencv_has_cuda_dnn():
    try:
        dev = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        dev = False
    try:
        info = cv2.getBuildInformation()
        built = (("CUDA:" in info and "YES" in info.split("CUDA")[1]) or
                 ("NVIDIA CUDA:" in info and "YES" in info.split("NVIDIA CUDA")[1]))
    except Exception:
        built = False
    return dev and built

class YuNetDet:
    def __init__(self, res=320, conf=0.7, nms=0.3, topK=5000):
        backend,target = ((cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
                          if opencv_has_cuda_dnn()
                          else (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU))
        self.net = YuNet(
            modelPath=os.path.join('preprocessing','yunet','face_detection_yunet_2023mar.onnx'),
            inputSize=[res, res], confThreshold=conf, nmsThreshold=nms, topK=topK,
            backendId=backend, targetId=target
        )
    def set_size(self, W, H): self.net.setInputSize((W,H))
    def infer(self, bgr): return self.net.infer(bgr)

class Aligner:
    """Allinea un singolo frame alla volta, restituisce (H,W,3) uint8 a imsize."""
    def __init__(self, imsize):
        self.imsize = int(imsize)
        self.align = FasterCropAlignXRay(self.imsize)
    def one(self, info, img_rgb):
        # usa API batch con batch=1 per minimizzare codice
        _, aligned = self.align([info], [img_rgb])
        if not isinstance(aligned, np.ndarray) or aligned.ndim!=4 or aligned.shape[0] < 1:
            return None
        fr = aligned[0]
        if fr.shape[0] != self.imsize or fr.shape[1] != self.imsize:
            fr = cv2.resize(fr, (self.imsize, self.imsize), interpolation=cv2.INTER_LINEAR)
        return fr

class AFModel:
    """Carica modello AltFreezing e cattura penultimate via hook."""
    def __init__(self, cfg_path, ckpt_path, amp=False):
        cfg.init_with_yaml(); cfg.update_with_yaml(cfg_path); cfg.freeze()
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = PluginLoader.get_classifier(cfg.classifier_type)().to(self.device).eval()
        self.model.load(ckpt_path)
        self.amp = (amp and self.device=='cuda')
        # normalizzazione su range 0..255 (AltFreezing usa clip uint8)
        self.mean = torch.tensor([0.485,0.456,0.406], device=self.device).view(1,3,1,1,1)*255
        self.std  = torch.tensor([0.229,0.224,0.225], device=self.device).view(1,3,1,1,1)*255
        # hook all’ultimo Linear
        self._feat = None; self._logits = None
        last_linear = None
        for m in self.model.modules():
            if isinstance(m, nn.Linear): last_linear = m
        if last_linear is None:
            raise RuntimeError("Layer Linear finale non trovato")
        def _hk(m, inp, out):
            self._feat = inp[0].detach()
            self._logits = out.detach()
        last_linear.register_forward_hook(_hk)

    @torch.inference_mode()
    def infer_clip(self, aligned_THWC_uint8):
        """
        aligned_THWC_uint8: np.uint8 [T,H,W,3]
        ritorna: logits[1,C], feat[1,D], score[1]
        """
        x = torch.from_numpy(aligned_THWC_uint8).to(self.device, non_blocking=True)  # [T,H,W,3], cpu→dev
        x = x.permute(3,0,1,2).unsqueeze(0).contiguous()  # [1,3,T,H,W]
        x = x.sub(self.mean).div(self.std)
        self._feat = None; self._logits = None
        ctx = torch.cuda.amp.autocast(enabled=self.amp)
        with ctx:
            out = self.model(x)
        logits = None
        if torch.is_tensor(out): logits = out
        elif isinstance(out, dict):
            for k in ('final_output','logits','cls','pred','y'):
                v = out.get(k); 
                if torch.is_tensor(v): logits = v; break
        if logits is None:  # fallback su primo tensor
            if isinstance(out, dict):
                for v in out.values():
                    if torch.is_tensor(v): logits = v; break
        if logits is None or self._feat is None:
            raise RuntimeError("Logits/feat non catturati")
        # score binario robusto
        if logits.ndim==2 and logits.size(1)==2:
            score = torch.softmax(logits, dim=1)[:,1]
        else:
            score = torch.sigmoid(logits.squeeze(1)) if logits.ndim>=1 else torch.sigmoid(logits)
        return logits.detach().cpu(), self._feat.detach().cpu(), score.detach().cpu()

def square_from_tlbr(tlbr, scale, W, H):
    x1,y1,x2,y2 = map(float, tlbr)
    cx,cy = 0.5*(x1+x2), 0.5*(y1+y2)
    side  = max(x2-x1, y2-y1) * float(scale)
    x1 = max(0, int(round(cx - side/2))); y1 = max(0, int(round(cy - side/2)))
    x2 = min(W, x1 + int(round(side)));   y2 = min(H, y1 + int(round(side)))
    side = min(x2-x1, y2-y1); x2 = x1 + side; y2 = y1 + side
    return np.array([x1,y1,x2,y2], np.int32)

def process_video(video_path, dataset_root, out_root, model, aligner,
                  clip_size=32, stride=16, det_res=320, det_conf=0.7,
                  detect_every=1, crop_scale=1.0, min_track_side=96,
                  save_fp16=True):
    # out path che preserva struttura relativa
    rel_dir = os.path.relpath(os.path.dirname(video_path), dataset_root)
    vname = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(out_root, rel_dir, vname)
    os.makedirs(out_dir, exist_ok=True)
    y = infer_label_from_path(video_path)

    # tracker + detector
    class _Args: pass
    a=_Args(); a.track_thresh=0.8; a.track_buffer=90; a.match_thresh=0.8; a.mot20=False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps<5 or fps>240: fps=30.0
    tracker = BYTETracker(a, frame_rate=fps)
    det = YuNetDet(det_res, det_conf)
    ok, first = cap.read()
    if not ok:
        cap.release(); return 0
    H,W = first.shape[:2]; det.set_size(W,H)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # buffer: SOLO frame allineati a imsize
    from collections import deque
    buf = {}  # tid -> deque([H,W,3] uint8)
    clip_counter = collections.Counter()
    saved = 0
    frame_idx = -1

    def flush(tid):
        nonlocal saved
        dq = buf[tid]
        if len(dq) < clip_size: return
        # crea clip T,H,W,3 senza copie inutili
        arr = np.empty((clip_size, aligner.imsize, aligner.imsize, 3), dtype=np.uint8)
        for i in range(clip_size): arr[i] = dq[i]
        logits, feat, score = model.infer_clip(arr)
        # salva compresso
        outp = os.path.join(out_dir, f"{vname}_tid{tid}_c{clip_counter[tid]:05d}.npz")
        np.savez_compressed(
            outp,
            feat = feat.numpy().astype(np.float16) if save_fp16 else feat.numpy().astype(np.float32),
            logits = logits.numpy().astype(np.float16) if save_fp16 else logits.numpy().astype(np.float32),
            score = float(score.squeeze().item()),
            y = np.int64(y),
            tid = np.int64(tid),
            clip_idx = np.int64(clip_counter[tid]),
            video_rel = os.path.join(rel_dir, os.path.basename(video_path))
        )
        clip_counter[tid] += 1
        # sliding window: mantieni ultimi (clip_size - stride)
        keep = max(0, clip_size - stride)
        for _ in range(clip_size - keep):
            dq.popleft()
        saved += 1

    while True:
        ok, fbgr = cap.read()
        if not ok: break
        frame_idx += 1
        need_det = (frame_idx % max(1, detect_every) == 0) or (len(tracker.tracked_stracks) == 0)
        dets = det.infer(fbgr) if need_det else None

        tracks_in=[]
        if dets is not None and len(dets)>0:
            dnp = np.asarray(dets, np.float32)
            for d in dnp:
                x,y,w,h,score = d[:5]
                if max(w,h) < 32: continue
                tlbr = np.array([x,y,x+w,y+h], np.float32)
                tracks_in.append(STrack(tlbr, float(score)))

        online = tracker.update(tracks_in, (H,W), (H,W))
        frgb = cv2.cvtColor(fbgr, cv2.COLOR_BGR2RGB)

        for tr in online or []:
            tlbr = tr.tlbr.astype(np.float32)
            if max(tlbr[2]-tlbr[0], tlbr[3]-tlbr[1]) < min_track_side: continue
            tid = tr.track_id
            big = square_from_tlbr(tlbr, crop_scale, W, H)
            x1,y1,x2,y2 = map(int, big)
            if x2<=x1 or y2<=y1: continue
            # info minimi per allineatore; landmark 5/68 assenti → usa solo box
            new_box = (tlbr.reshape(2,2).astype(np.float32) - np.array([[x1,y1]], np.float32)).reshape(-1)
            lm5  = np.zeros((5,2), np.float32)
            lm68 = np.empty((0,2), np.float32)
            info = (new_box, lm5, lm68, np.array([x1,y1,x2,y2], np.int32))
            crop = frgb[y1:y2, x1:x2]
            if crop.size == 0: continue
            aligned = aligner.one(info, crop)  # HxW x3 uint8
            if aligned is None: continue

            if tid not in buf:
                buf[tid] = deque(maxlen=clip_size*2)
            buf[tid].append(aligned)
            flush(tid)

    cap.release()
    # reset contatori globali tracker
    if hasattr(STrack,"_count"): STrack._count = 0
    if hasattr(BaseTrack,"_count"): BaseTrack._count = 0
    return saved

def main():
    ap = argparse.ArgumentParser("AF features a memoria ridotta")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cfg_path", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--clip_size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--det_res", type=int, default=320)
    ap.add_argument("--det_conf", type=float, default=0.7)
    ap.add_argument("--detect_every", type=int, default=1)
    ap.add_argument("--crop_scale", type=float, default=1.0)
    ap.add_argument("--min_track_side", type=int, default=96)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save_fp16", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # modello e aligner
    model = AFModel(args.cfg_path, args.ckpt_path, amp=args.amp)
    imsize = int(getattr(model.cfg, "imsize", 224))
    aligner = Aligner(imsize)

    total = 0
    for vpath in video_iter(args.dataset_root):
        n = process_video(
            vpath, args.dataset_root, args.out_dir, model, aligner,
            clip_size=args.clip_size, stride=args.stride,
            det_res=args.det_res, det_conf=args.det_conf, detect_every=args.detect_every,
            crop_scale=args.crop_scale, min_track_side=args.min_track_side,
            save_fp16=args.save_fp16
        )
        print(f"{os.path.relpath(vpath, args.dataset_root)} clips={n}")
        total += n
    print("Totale clip salvate:", total)

if __name__ == "__main__":
    main()
