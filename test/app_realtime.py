# app_realtime.py
import ctypes, time, cv2, numpy as np
from statistics import median
from af_realtime import RealtimeAF
from win_capture import iter_window_frames, get_foreground_hwnd

ctypes.windll.user32.SetProcessDPIAware()
WIN = "AltFreezing Realtime"
VK_F8 = 0x77  # F8

def letterbox(img, dst_w=1280, dst_h=720):
    h, w = img.shape[:2]
    r = min(dst_w / w, dst_h / h)
    nw, nh = int(w * r), int(h * r)
    out = np.zeros((dst_h, dst_w, 3), np.uint8)
    x, y = (dst_w - nw) // 2, (dst_h - nh) // 2
    out[y:y+nh, x:x+nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, x, y

def draw_boxes(view, r, ox, oy, boxes, scores, thr):
    any_fake = False; n = 0
    H, W = view.shape[:2]
    for tid, box in boxes.items():
        x1, y1, x2, y2 = map(int, box)
        X1, Y1 = int(x1 * r) + ox, int(y1 * r) + oy
        X2, Y2 = int(x2 * r) + ox, int(y2 * r) + oy
        sc_list = scores.get(tid, [])
        sc = float(median(sc_list)) if sc_list else float("nan")
        is_fake = (not np.isnan(sc)) and sc >= thr
        any_fake |= is_fake; n += 1
        color = (0,0,255) if is_fake else (0,200,0)
        cv2.rectangle(view, (X1, Y1), (X2, Y2), color, 2)
        label = f"id {tid} | s={sc:.3f} " if not np.isnan(sc) else f"id {tid} | s=… "
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_top = max(0, Y1 - th - 8)
        cv2.rectangle(view, (X1, y_top), (X1 + tw + 6, Y1), color, -1)
        cv2.putText(view, label, (X1 + 3, Y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return any_fake, n


def overlay_message(lines):
    view = np.zeros((360, 800, 3), np.uint8)
    y = 110
    for i, (txt, scale) in enumerate(lines):
        (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x = (view.shape[1] - w)//2
        cv2.putText(view, txt, (x, y + i*48), cv2.FONT_HERSHEY_SIMPLEX, scale, (230,230,230), 2, cv2.LINE_AA)
    return view

def select_source_blocking():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    prev = False
    while True:
        view = overlay_message([
            ("Press F8 on the window you want to capture", 0.9),
            ("Or ESC/Q to exit", 0.9),
        ])
        cv2.imshow(WIN, view)

        k = cv2.waitKey(30) & 0xFF
        if k in (27, ord('q'), ord('Q')):
            return None

        pressed = bool(ctypes.windll.user32.GetAsyncKeyState(VK_F8) & 0x8000)
        # rising edge
        if pressed and not prev:
            hwnd = get_foreground_hwnd()
            # attende rilascio per evitare retrigger
            while bool(ctypes.windll.user32.GetAsyncKeyState(VK_F8) & 0x8000):
                cv2.waitKey(10)
            if hwnd:
                return hwnd
        prev = pressed

def decide_meeting_fake(af, min_frames=128, percentile_p=80.0):
    thr = float(getattr(af, "optimal_threshold", 0.362))
    p = float(getattr(getattr(af, "args", object()), "percentile_p", percentile_p))

    any_ready = False
    scores_by_tid = getattr(af, "running_scores", {})
    frames_per_tid = getattr(af, "frames_per_tid", {})

    for tid, sc_list in scores_by_tid.items():
        n_frames = int(frames_per_tid.get(tid, 0))
        if n_frames >= min_frames and sc_list:
            any_ready = True
            s = float(np.percentile(np.asarray(sc_list, dtype=float), p))
            if s >= thr:
                return True, True
    if any_ready:
        return True, False
    return False, False



def run_loop(af, target_hz=8.0):
    thr = getattr(af, "optimal_threshold", 0.362)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
        hwnd = select_source_blocking()
        if hwnd is None:
            return

        if hasattr(af, "bytetrack") and hasattr(af.bytetrack, "reset"):
            af.bytetrack.reset(fps=target_hz)

        try:
            frames = iter_window_frames(hwnd, target_hz=target_hz, refresh_every=120)
            prev_f8 = False
            for frame_bgr in frames:
                af.step(frame_bgr)

                view, r, ox, oy = letterbox(frame_bgr, 1280, 720)
                # disegna solo i box, niente etichette
                _, _ = draw_boxes(
                    view, r, ox, oy,
                    boxes=getattr(af, "last_boxes", {}),
                    scores=getattr(af, "running_scores", {}),
                    thr=thr
                )

                # decisione: mostrare SOLO quando pronto
                ready, is_fake = decide_meeting_fake(af, min_frames=128, percentile_p=80.0)
                if ready:
                    status =  "REAL"
                    color = (0, 0, 255) if is_fake else (0, 200, 0)
                    cv2.putText(view, status, (12, 96),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

                cv2.imshow(WIN, view)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q'), ord('Q')):
                    return

                # F8 edge-trigger
                pressed = bool(ctypes.windll.user32.GetAsyncKeyState(VK_F8) & 0x8000)
                if pressed and not prev_f8:
                    while bool(ctypes.windll.user32.GetAsyncKeyState(VK_F8) & 0x8000):
                        cv2.waitKey(10)
                    break
                prev_f8 = pressed


        except Exception:
            continue  # finestra persa → torna alla selezione


def main():
    af = RealtimeAF(
        cfg_path="i3d_ori.yaml",
        ckpt_path="altfreezing/checkpoints/model.pth",
        clip_size=32, stride=30,        # 32 frame consecutivi
        detect_every=4, mesh_every=4,  # detection ogni frame
        detector_res=320, conf=0.8,
        track_thresh=0.8, track_buffer=90, match_thresh=0.8,
        roi_scale=2.0, crop_scale=1.0,
        q_weighting=True,
        q_min_size_soft=72, q_min_size_hard=48,
        q_lap_soft=24, q_lap_hard=8,
        optimal_threshold=0.362,
        score_is_real=False, 
        start_min_size=50,
        exclude_rect=(0.70, 0.70, 1.00, 1.00)
    )
    run_loop(af, target_hz=30.0)

if __name__ == "__main__":
    main()
