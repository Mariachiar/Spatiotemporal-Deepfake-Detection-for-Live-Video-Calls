#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader
import os, argparse, logging, torch, sys

# consenti import dal pacchetto 'dualrun'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ----- riproducibilità identica a demo.py -----
import random, numpy as np, torch
random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ====== MediaPipe helpers (solo per ramo yunet) ======
MP_LEFT_EYE_RING  = [33,7,163,144,145,153]
MP_RIGHT_EYE_RING = [263,249,390,373,374,380]
MP68_IDX = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,
            70,63,105,66,107,336,296,334,293,300,
            168,6,197,195,5,4,1,19,94,
            33,7,163,144,145,153,263,249,390,373,374,380,
            61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88]

_mp_face_mesh = None
def _get_facemesh():
    global _mp_face_mesh
    if _mp_face_mesh is None:
        import mediapipe as mp
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=5, refine_landmarks=True, static_image_mode=False
        )
    return _mp_face_mesh

def _precrop_square(frame_bgr, box_xyxy, out=224, scale=1.2):
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(float)
    w, h = x2 - x1, y2 - y1
    side = max(1.0, scale * max(w, h))
    cx, cy = x1 + w/2.0, y1 + h/2.0
    wx1, wy1 = cx - side/2.0, cy - side/2.0
    wx2, wy2 = cx + side/2.0, cy + side/2.0

    # FIX: usa wy1 anche per iy1
    ix1, iy1 = int(np.floor(wx1)), int(np.floor(wy1))
    ix2, iy2 = int(np.ceil(wx2)),  int(np.ceil(wy2))

    pad_left   = max(0, -ix1)
    pad_top    = max(0, -iy1)
    pad_right  = max(0, ix2 - W)
    pad_bottom = max(0, iy2 - H)

    cx1, cy1 = max(0, ix1), max(0, iy1)
    cx2, cy2 = min(W, ix2), min(H, iy2)
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE)

    crop_resized = cv2.resize(crop, (out, out), interpolation=cv2.INTER_LINEAR)
    meta = dict(wx1=wx1, wy1=wy1, side=side, out=out,
                pad_left=pad_left, pad_top=pad_top, ix1=ix1, iy1=iy1)
    return crop_resized, meta


def _reproject_pts(pts_in_out, meta):
    s = meta["side"] / meta["out"]
    x = pts_in_out.astype(np.float32) * s
    x[:, 0] -= meta["pad_left"]; x[:, 1] -= meta["pad_top"]
    x[:, 0] += meta["ix1"];      x[:, 1] += meta["iy1"]
    return x

def mesh_lm5_lm68_precrop(frame_bgr, big_box):
    crop224, meta = _precrop_square(frame_bgr, big_box, out=224, scale=1.2)
    rgb = cv2.cvtColor(crop224, cv2.COLOR_BGR2RGB)
    mp_face_mesh = _get_facemesh()
    res = mp_face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None, None
    lm = res.multi_face_landmarks[0].landmark
    out = crop224.shape[0]
    pts224 = np.array([[p.x*out, p.y*out] for p in lm], dtype=np.float32)
    pts = _reproject_pts(pts224, meta)
    lm68 = pts[MP68_IDX].astype(np.float32)
    left_c  = pts[MP_LEFT_EYE_RING].mean(axis=0)
    right_c = pts[MP_RIGHT_EYE_RING].mean(axis=0)
    nose    = pts[1]; mouth_L = pts[61]; mouth_R = pts[291]
    lm5 = np.vstack([left_c, right_c, nose, mouth_L, mouth_R]).astype(np.float32)
    return lm5, lm68

# ====== YuNet loader (solo per ramo yunet) ======
HAS_YUNET = True
try:
    from preprocessing.yunet.yunet import YuNet
except Exception:
    HAS_YUNET = False

def init_yunet(yunet_res=640, conf=0.6, nms=0.3, topK=500, backend_target=0):
    be_tg = [(cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU),
             (cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA)][backend_target]
    return YuNet(modelPath=os.path.join('preprocessing','yunet','face_detection_yunet_2023mar.onnx'),
                 inputSize=[yunet_res, yunet_res], confThreshold=conf, nmsThreshold=nms, topK=topK,
                 backendId=be_tg[0], targetId=be_tg[1])

# ====== default param ======
DEF_MAX_FRAME = 32
DEF_CFG = "i3d_ori.yaml"
DEF_CKPT = "altfreezing/checkpoints/model.pth"
DEF_THR = 0.04

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", choices=["retina","yunet"], default="retina")
    p.add_argument("--video_path", default=None)
    p.add_argument("--out_dir", default="prediction")
    p.add_argument("--max_frame", type=int, default=DEF_MAX_FRAME)
    p.add_argument("--cfg_path", default=DEF_CFG)
    p.add_argument("--ckpt_path", default=DEF_CKPT)
    p.add_argument("--optimal_threshold", type=float, default=DEF_THR)
    # YuNet opts
    p.add_argument("--yunet_res", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.6)
    p.add_argument("--nms", type=float, default=0.3)
    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--backend_target", type=int, choices=[0,1], default=0)
    args = p.parse_args()

    assert args.video_path, "specifica --video_path"

    # ===== Model & preproc identicali a demo.py =====
    cfg.init_with_yaml()
    cfg.update_with_yaml(args.cfg_path); cfg.freeze()

    classifier = PluginLoader.get_classifier(cfg.classifier_type)()
    classifier.cuda(); classifier.eval(); classifier.load(args.ckpt_path)
    crop_align_func = FasterCropAlignXRay(cfg.imsize)

    os.makedirs(args.out_dir, exist_ok=True)
    basename = f"{os.path.splitext(os.path.basename(args.video_path))[0]}.avi"
    out_file = os.path.join(args.out_dir, basename)

    # ===== Frames =====
    frames = grab_all_frames(args.video_path, max_size=args.max_frame, cvt=True)
    print("number of frames:", len(frames))
    if len(frames) == 0:
        raise RuntimeError("Nessun frame letto")
    shape = frames[0].shape[:2]

    # ===== Detection + lm =====
    if args.detector == "retina":
        # IDENTICO al demo.py
        #cache_file = f"{args.video_path}_{args.max_frame}.pth"
        #if os.path.exists(cache_file):
           # detect_res, all_lm68 = torch.load(cache_file)
           # print("detection result loaded from cache")
        #else:
        print("detecting")
        detect_res, all_lm68, _frames2 = detect_all(args.video_path, return_frames=True, max_size=args.max_frame)
            #torch.save((detect_res, all_lm68), cache_file)
        print("detect finished")

        # conversione formato identica
        all_detect_res = []
        assert len(all_lm68) == len(detect_res)
        for faces, faces_lm68 in zip(detect_res, all_lm68):
            new_faces = []
            for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
                new_faces.append((box, lm5, face_lm68, score))
            all_detect_res.append(new_faces)
        detect_res_fmt = all_detect_res

    else:
        # YuNet + MediaPipe: boxes -> big_box con get_crop_box, landmarks via MediaPipe
        if not HAS_YUNET:
            raise RuntimeError("YuNet non disponibile")
        yunet = init_yunet(args.yunet_res, args.conf, args.nms, args.topk, args.backend_target)
        yunet.setInputSize((args.yunet_res, args.yunet_res))
        detect_res_fmt = []
        for frame in frames:
            H, W = frame.shape[:2]
            fy = cv2.resize(frame, (args.yunet_res, args.yunet_res))
            dets = yunet.infer(fy)
            sx, sy = W/args.yunet_res, H/args.yunet_res
            row = []
            if dets is not None and len(dets) > 0:
                dets = np.asarray(dets)
                C = dets.shape[1]
                for d in dets:
                    if C >= 15:
                        x,y,w,h = d[0:4].astype(float)
                        x1,y1,x2,y2 = x*sx, y*sy, (x+w)*sx, (y+h)*sy
                        sc = float(d[14])
                    else:
                        x1,y1,x2,y2,sc = d[:5].astype(float)
                        x1,y1,x2,y2 = x1*sx, y1*sy, x2*sx, y2*sy
                    if sc < args.conf:
                        continue
                    # uniforma al repo: usa big_box calcolato su box detector
                    big_box = get_crop_box((H, W), np.array([x1,y1,x2,y2], np.float32), scale=0.5).astype(np.int32)
                    lm5, lm68 = mesh_lm5_lm68_precrop(frame, big_box)
                    if lm5 is None:
                        continue
                    if lm68 is None:  # non usato dall'allineatore, ma per compat
                        lm68 = np.zeros((68,2), np.float32)
                    row.append((big_box.astype(np.float32), lm5.astype(np.float32), lm68.astype(np.float32), sc))
            detect_res_fmt.append(row)

    # ===== Tracking identico a demo.py =====
    print("split into super clips")
    tracks = multiple_tracking(detect_res_fmt)
    tuples = [(0, len(detect_res_fmt))] * len(tracks)
    print("full_tracks", len(tracks))
    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res_fmt)

    # ===== Packing dati e crop identico (usa SEMPRE get_crop_box nel ramo retina) =====
    data_storage = {}
    frame_boxes = {}
    super_clips = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        print(start, end)
        assert len(detect_res_fmt[start:end]) == len(track)
        super_clips.append(len(track))

        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box, lm5, lm68 = face[:3]

            if args.detector == "retina":
                # Comportamento originale demo.py
                big_box = get_crop_box(shape, box, scale=0.5)
            else:
                # Nel ramo yunet il box è già big_box
                big_box = box.astype(np.int32)

            top_left = big_box[:2][None, :]
            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left
            new_box = (box.reshape(2, 2) - top_left).reshape(-1)
            info = (new_box, new_lm5, new_lm68, big_box)

            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]
            base_key = f"{track_i}_{j}_"
            data_storage[f"{base_key}img"] = cropped
            data_storage[f"{base_key}ldm"] = info
            data_storage[f"{base_key}idx"] = frame_idx
            frame_boxes[frame_idx] = np.rint(big_box if args.detector=="yunet" else box).astype(int)

    print("sampling clips from super clips", super_clips)

    # ===== Sampling clip identico =====
    clips_for_video = []
    clip_size = cfg.clip_size
    pad_length = clip_size - 1

    for super_clip_idx, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))
        if super_clip_size < clip_size:
            post_module = inner_index[1:-1][::-1] + inner_index
            l_post = len(post_module)
            post_module = post_module * (pad_length // l_post + 1)
            post_module = post_module[:pad_length]
            pre_module = inner_index + inner_index[1:-1][::-1]
            l_pre = len(pre_module)
            pre_module = pre_module * (pad_length // l_pre + 1)
            pre_module = pre_module[-pad_length:]
            inner_index = pre_module + inner_index + post_module

        super_clip_size = len(inner_index)
        frame_range = [inner_index[i:i+clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size]
        for indices in frame_range:
            clip = [(super_clip_idx, t) for t in indices]
            clips_for_video.append(clip)

    # ===== Inference identica =====
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1, 1)
    std  = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1, 1)

    preds = []
    frame_res = {}

    for clip in tqdm(clips_for_video, desc="testing"):
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]

        _, images_align = crop_align_func(landmarks, images)

        # pack tensor
        images_t = torch.as_tensor(images_align, dtype=torch.float32).cuda().permute(3, 0, 1, 2)
        images_t = images_t.unsqueeze(0).sub(mean).div(std)

        with torch.no_grad():
            output = classifier(images_t)
        pred = float(F.sigmoid(output["final_output"]))

        for f_id in frame_ids:
            frame_res.setdefault(f_id, []).append(pred)
        preds.append(pred)

    print(np.mean(preds))

    boxes = []
    scores = []
    for frame_idx in range(len(frames)):
        if frame_idx in frame_res:
            pred_prob = np.mean(frame_res[frame_idx])
            rect = frame_boxes[frame_idx]
        else:
            pred_prob = None
            rect = None
        scores.append(pred_prob)
        boxes.append(rect)

    SupplyWriter(args.video_path, out_file, args.optimal_threshold).run(frames, scores, boxes)

if __name__ == "__main__":
    main()
