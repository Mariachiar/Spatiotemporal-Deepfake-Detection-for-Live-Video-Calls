import os
# ---- OpenMP / BLAS: un solo runtime e pochi thread ----
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # workaround: evita crash anche se non ideale
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# ---- TensorFlow / Mediapipe logging ----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # riduce log TF/TFLite
# absl va impostato prima che mediapipe venga importato
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
# af_realtime.py
import os, numpy as np, cv2, torch, collections, sys, types, logging
import mediapipe as mp
from collections import deque, defaultdict, Counter

# Path progetto
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ALT_DIR = os.path.join(ROOT, "altfreezing")
PRE_DIR = os.path.join(ROOT, "preprocessing")
for p in (ROOT, ALT_DIR, PRE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import config as cfg
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader

from preprocessing.ByteTrack.byte_tracker import BYTETracker, STrack
from preprocessing.ByteTrack.matching import iou_distance
from preprocessing.yunet.yunet import YuNet
from types import SimpleNamespace
from contextlib import nullcontext
import platform
try:
    import resource
except ImportError:
    resource = None

from contextlib import nullcontext
class _Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        # una sola istanza per classe; i parametri della PRIMA init restano "bloccati"
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

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
        # aligned_batch_bthwc: (B,T,H,W,C) RGB 0..255
        x = torch.as_tensor(aligned_batch_bthwc, dtype=torch.float32, device=self.device).permute(0,4,1,2,3)
        if self.channels_last and x.dim()==5:
            try:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            except AttributeError:
                pass
        x = x.sub(self.mean).div(self.std)
        with torch.inference_mode():
            with self.amp_ctx():
                out = self.model(x)
        t = out
        if isinstance(t, dict):
            t = t.get("final_output", t.get("logits", next((v for v in t.values() if torch.is_tensor(v)), t)))
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if t.size(1) == 1:
            scores = torch.sigmoid(t).squeeze(1).float().cpu().numpy()
        else:
            scores = torch.softmax(t, dim=1)[:, 1].float().cpu().numpy()
        return scores


class CropAlignSvc(metaclass=_Singleton):
    def __init__(self, cfg_path):
        self.cfg = ConfigSvc(cfg_path).cfg
        self.crop_align = FasterCropAlignXRay(self.cfg.imsize)
    def __call__(self, infos, imgs):
        return self.crop_align(infos, imgs)
    
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

class YuNetSvc(metaclass=_Singleton):
    def __init__(self, detector_res=480, conf=0.8, nms=0.3, topK=5000):
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


def _square_roi_from_tlbr(tlbr, roi_scale, W, H):
    x1,y1,x2,y2 = tlbr.astype(int)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    side = int(max(x2-x1, y2-y1) * 1.0); side = max(2, side)
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



class RealtimeAF:
    def __init__(self, **kwargs):
        a = SimpleNamespace(**kwargs)
        self.args = a
        self.score_is_real = bool(getattr(a, "score_is_real", False))

        self.device = DeviceSvc().device
        self.frames_per_tid = collections.Counter()
        self.cfg = ConfigSvc(getattr(a, "cfg_path", "i3d_ori.yaml")).cfg
        self.classifier = ClassifierSvc(
            getattr(a, "cfg_path", "i3d_ori.yaml"),
            getattr(a, "ckpt_path", "altfreezing/checkpoints/model.pth"),
            amp=getattr(a, "amp", True),
            channels_last=getattr(a, "channels_last", False)
        )
        self.crop_align = CropAlignSvc(getattr(a, "cfg_path", "i3d_ori.yaml"))
        self.drop_after = int(getattr(a, "drop_after", 90))
        self.missed = Counter()
        self.clip_hist = defaultdict(lambda: deque(maxlen=5))
        self.state = {}

        # servizi
        a.mot20 = False
        self.yunet = YuNetSvc(getattr(a, "detector_res", 360), getattr(a, "conf", 0.8), 0.3, 5000)
        self.facemesh = FaceMeshSvc(getattr(a, "mp_min_det", 0.35), getattr(a, "mp_min_trk", 0.35), getattr(a, "refine_size", 320))
        self.bytetrack = ByteTrackSvc(a, fps=30.0)

        # parametri come nel batch
        self.clip_size = int(getattr(a, "clip_size", getattr(self.cfg, "clip_size", 32)))
        self.roi_scale = float(getattr(a, "roi_scale", 2.0))
        self.crop_scale = float(getattr(a, "crop_scale", 0.6))
        self.mesh_every = int(getattr(a, "mesh_every", 1))
        self.detect_every = int(getattr(a, "detect_every", 1))
        self.start_conf = float(getattr(a, "start_conf", 0.76))
        self.start_min_size = int(getattr(a, "start_min_size", 80))
        self.q_weighting = bool(getattr(a, "q_weighting", True))
        self.q_min_size_soft = int(getattr(a, "q_min_size_soft", 64))
        self.q_min_size_hard = int(getattr(a, "q_min_size_hard", 32))
        self.q_lap_soft = float(getattr(a, "q_lap_soft", 20.0))
        self.q_lap_hard = float(getattr(a, "q_lap_hard", 5.0))
        self.channels_last = bool(getattr(a, "channels_last", False))
        self.qa_min_side = int(getattr(a, "qa_min_side", 240))
        self.qa_min_lap  = float(getattr(a, "qa_min_lap", 16.0))
        self._q_hist = defaultdict(lambda: deque(maxlen=64)) 

        # finestra scorrevole
        self.stride = int(getattr(a, "stride", 52))  # ogni 8 frame emetti uno score da 32

        self.last_aligned = {}                              # tid -> ultimo frame aligned (H,W,3 RGB)
        self.running_scores = collections.defaultdict(list) # tid -> lista score clip


        # stato
        self.frame_idx = -1
        self.last_lm = {}
        self.cur_imgs = {}   # tid -> list of RGB crops
        self.cur_infos = {}  # tid -> list of (new_box,lm5,lm68,big)
        self.cur_w = {}      # tid -> list of weights
        self.batch_imgs, self.batch_infos, self.batch_tid = [], [], []
        self.WH = None
        self._since_emit = collections.Counter()  # tid -> frames da ultimo score
        self.last_boxes = {}                      # tid -> tlbr float32
        self.optimal_threshold = float(getattr(a, "optimal_threshold", 0.5))
        # escludi la self-view (di solito in basso a destra). Coordinate NORMALIZZATE [0..1]
        self.exclude_rect = getattr(a, "exclude_rect", (0.70, 0.70, 1.00, 1.00)) 

    def _frame_quality_weight(self, crop_rgb):
        if crop_rgb.size == 0: 
            return 0.0, 0.0, 0.0
        h, w = crop_rgb.shape[:2]; min_side = min(h, w)
        small = crop_rgb if min_side <= 0 else cv2.resize(crop_rgb, (max(1,w//2), max(1,h//2)), interpolation=cv2.INTER_AREA)
        lap = variance_of_laplacian(small)
        # registra sempre le metriche
        return_weight_only = 0.0
        if min_side < self.q_min_size_hard or lap < self.q_lap_hard:
            return return_weight_only, float(min_side), float(lap)
        if not self.q_weighting:
            return 1.0, float(min_side), float(lap)
        size_w = 1.0 if min_side >= self.q_min_size_soft else max(0.0, (min_side - self.q_min_size_hard) / max(1.0, (self.q_min_size_soft - self.q_min_size_hard)))
        lap_w  = 1.0 if lap >= self.q_lap_soft else max(0.0, (lap - self.q_lap_hard) / max(1e-6, (self.q_lap_soft - self.q_lap_hard)))
        return float(size_w * lap_w), float(min_side), float(lap)


    def pick_interlocutor_id(self, H, W):
        if not self.last_boxes: return None
        x1n, y1n, x2n, y2n = self.exclude_rect
        X1, Y1, X2, Y2 = x1n*W, y1n*H, x2n*W, y2n*H
        def is_in_selfview(box):
            x1,y1,x2,y2 = box
            cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
            return (X1 <= cx <= X2) and (Y1 <= cy <= Y2)
        cand = [(tid, (b[2]-b[0])*(b[3]-b[1])) 
                for tid,b in self.last_boxes.items() if not is_in_selfview(b)]
        if not cand:  # fallback: max area
            cand = [(tid, (b[2]-b[0])*(b[3]-b[1])) for tid,b in self.last_boxes.items()]
        return max(cand, key=lambda t: t[1])[0]

    def _enqueue_clip(self, tid):
        imgs = self.cur_imgs.get(tid, [])
        infos = self.cur_infos.get(tid, [])
        # realtime: no padding → niente output se non ho 32 frame pieni
        if len(imgs) != self.clip_size or len(infos) != self.clip_size:
            return False
        fixed_infos = []
        for nb, lm5, lm68, big in infos:
            nb  = np.asarray(nb, dtype=np.float32).reshape(4,)
            lm5 = np.asarray(lm5, dtype=np.float32).reshape(5,2)
            lm68= np.asarray(lm68, dtype=np.float32).reshape(68,2)
            big = np.asarray(big, dtype=np.int32).reshape(4,)
            fixed_infos.append((nb, lm5, lm68, big))
        self.batch_imgs.append(list(imgs))
        self.batch_infos.append(fixed_infos)
        self.batch_tid.append(tid)
        return True
    
    def _in_exclude(self, box, H, W):
        x1,y1,x2,y2 = box
        cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
        x1n,y1n,x2n,y2n = self.exclude_rect
        return (x1n*W <= cx <= x2n*W) and (y1n*H <= cy <= y2n*H)


    def _flush_and_infer(self):
        if not self.batch_imgs: return []
        aligned_all, tids_used = [], []
        aligned_previews = {}

        for infos, imgs, tid in zip(self.batch_infos, self.batch_imgs, self.batch_tid):
            try:
                _, aligned = self.crop_align(infos, imgs)  # (T,H,W,C) RGB
            except Exception:
                continue
            if not isinstance(aligned, np.ndarray) or aligned.ndim != 4:
                continue
            aligned_all.append(aligned); tids_used.append(tid)
            aligned_previews[tid] = aligned[-1].copy()

        self.batch_imgs.clear(); self.batch_infos.clear(); self.batch_tid.clear()
        if not aligned_all: return []

        arr = np.stack(aligned_all, 0)  # (B,T,H,W,C)


        scores = self.classifier.infer_scores(arr) 
        if self.score_is_real:
            scores = 1.0 - np.asarray(scores, dtype=float)

        results = []
        for tid, s in zip(tids_used, scores):
            s = float(s)
            self.last_aligned[tid] = aligned_previews.get(tid)
            self.running_scores[tid].append(s)
            results.append((tid, s))
            self.clip_hist[tid].append(s)

            sm = float(np.median(self.clip_hist[tid]))
            T_high, T_low = 0.75, 0.65
            st = self.state.get(tid, {"fake": False})
            if not st["fake"] and sm >= T_high:
                st["fake"] = True
            elif st["fake"] and sm < T_low:
                st["fake"] = False
            self.state[tid] = st

        return results
    
    def track_quality_ok(self, tid):
        vals = list(self._q_hist.get(tid, []))
        if not vals:
            return True
        ms = np.array(vals, float)
        med_side = float(np.median(ms[:,0]))
        med_lap  = float(np.median(ms[:,1]))
        return not (med_side < self.qa_min_side or med_lap < self.qa_min_lap)


    def step(self, frame_bgr):
        self.frame_idx += 1
        H,W = frame_bgr.shape[:2]
        if self.WH != (W,H):
            self.yunet.set_frame_size(W,H); self.WH=(W,H)

        need_det = (self.frame_idx % max(1,self.detect_every) == 0)

        dets = self.yunet.infer(frame_bgr) if need_det else None
        tracks_in = []
        if dets is not None and len(dets)>0:
            for d in dets:
                d = np.asarray(d, dtype=np.float32)
                if d[4] >= self.start_conf and max(d[2],d[3]) >= self.start_min_size:
                    tracks_in.append(STrack(d[:4], score=float(d[4])))

        online = self.bytetrack.update(tracks_in, (H,W), (H,W))

        frgb = None
        det_tlbr, dets_np = [], None
        if dets is not None and len(dets)>0:
            dets_np = np.asarray(dets, dtype=np.float32)
            for d in dets_np:
                x,y,wf,hf = d[:4]; det_tlbr.append([x,y,x+wf,y+hf])
        det_tlbr = np.asarray(det_tlbr, dtype=np.float32) if det_tlbr else None

        ready = []
        kept_boxes = {}  # <- box validi solo per questo frame

        for tr in online or []:
            # scarta miniature/self-view
            if self._in_exclude(tr.tlbr, H, W):
                continue

            tid = tr.track_id
            kept_boxes[tid] = tr.tlbr.astype(np.float32).copy()

            if tid not in self.cur_imgs:
                self.cur_imgs[tid], self.cur_infos[tid], self.cur_w[tid] = [], [], []
                self._since_emit[tid] = 0

            yunet_lm5 = None
            if det_tlbr is not None and len(det_tlbr)>0:
                ious = 1.0 - iou_distance(np.array([tr.tlbr], dtype=np.float32), det_tlbr)[0]
                k = int(np.argmax(ious))
                if ious[k] >= 0.4:
                    yunet_lm5 = dets_np[k][5:15].reshape(5,2)

            fm = None
            if (self.frame_idx % self.mesh_every)==0 or (tid not in self.last_lm):
                if frgb is None: frgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mesh = self.facemesh.pick(tr.tlbr)
                fm = facemesh_on_square_roi(frgb, tr.tlbr, self.roi_scale, mesh)
                if fm is None and yunet_lm5 is not None: fm = {'lm5': yunet_lm5, 'lm68': None}
                if fm is not None: self.last_lm[tid] = {**fm, 'frame_idx': self.frame_idx}
            else:
                cached = self.last_lm.get(tid)
                if cached is not None: fm = {'lm5': cached['lm5'], 'lm68': cached['lm68']}
                elif yunet_lm5 is not None: fm = {'lm5': yunet_lm5, 'lm68': None}
            if fm is None: 
                continue

            if frgb is None: frgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            big = get_crop_box((H, W), tr.tlbr, scale=self.crop_scale)
            x1,y1,x2,y2 = map(int, big)
            if x2<=x1 or y2<=y1: continue
            crop_rgb = frgb[y1:y2, x1:x2]
            wq, q_side, q_lap = self._frame_quality_weight(crop_rgb)
            self._q_hist[tid].append((q_side, q_lap))
            if wq <= 0.0:
                continue


            top_left = np.array([[x1, y1]], dtype=np.float32)
            new_box = (tr.tlbr.reshape(2,2).astype(np.float32) - top_left).reshape(-1)
            lm5 = fm['lm5'].astype(np.float32) - top_left
            lm68 = (fm['lm68'].astype(np.float32) - top_left) if (fm['lm68'] is not None) else np.zeros((68,2), np.float32)

            self.cur_infos[tid].append((new_box, lm5, lm68, np.array([x1,y1,x2,y2], dtype=np.int32)))
            self.cur_imgs[tid].append(crop_rgb)
            self.cur_w[tid].append(wq)
            self._since_emit[tid] += 1
            self.frames_per_tid[tid] += 1

            # finestra scorrevole: accumula e quando hai 32 emetti
            if len(self.cur_imgs[tid]) > self.clip_size:
                # mantieni finestra scorrevole
                self.cur_imgs[tid]  = self.cur_imgs[tid][-self.clip_size:]
                self.cur_infos[tid] = self.cur_infos[tid][-self.clip_size:]

            if len(self.cur_imgs[tid]) == self.clip_size and self._since_emit[tid] >= self.stride:
                self._since_emit[tid] = 0
                if self._enqueue_clip(tid):
                    ready.append(tid)


        # se ci sono clip pronte esegui inferenza
        results = []
        if ready:
            for _tid in ready: pass
            results = self._flush_and_infer()  # [(tid, score), ...]

            # mantieni una coda per overlap: tieni gli ultimi (clip_size - stride) frame
            keep_tail = max(0, self.clip_size - self.stride)
            for tid in ready:
                self.cur_imgs[tid]  = self.cur_imgs[tid][-keep_tail:]
                self.cur_infos[tid] = self.cur_infos[tid][-keep_tail:]
                self.cur_w[tid]     = self.cur_w[tid][-keep_tail:]
        alive = set(kept_boxes.keys())

        # aggiorna contatori di miss e pulisci SOLO oltre drop_after
        known_tids = set(self.cur_imgs) | set(self.missed) | set(self.last_boxes)
        for tid in known_tids:
            if tid in alive:
                self.missed[tid] = 0
            else:
                self.missed[tid] += 1
                if self.missed[tid] >= self.drop_after:
                    # purge hard
                    self.cur_imgs.pop(tid, None)
                    self.cur_infos.pop(tid, None)
                    self.cur_w.pop(tid, None)
                    self._since_emit.pop(tid, None)
                    self.last_lm.pop(tid, None)
                    self.running_scores.pop(tid, None)
                    self.last_aligned.pop(tid, None)
                    self.last_boxes.pop(tid, None)
                    self.missed.pop(tid, None)

        # aggiorna i box: vivi ora + box “ultimi noti” non ancora scaduti
        persisting = {tid: box for tid, box in self.last_boxes.items()
                    if self.missed.get(tid, 0) < self.drop_after}
        persisting.update(kept_boxes)   # i vivi vincono
        self.last_boxes = persisting


        # ritorna lista di (tid, score) per questa frame
        return results