# win_capture.py
import ctypes, time, cv2, numpy as np
import win32gui, win32ui, win32con
from ctypes import wintypes
ctypes.windll.user32.SetProcessDPIAware()
_user32 = ctypes.windll.user32
_user32.PrintWindow.restype = wintypes.BOOL
_user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
PW_RENDERFULLCONTENT = 0x00000002
_user32 = ctypes.windll.user32
_dwmapi = ctypes.windll.dwmapi
DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_CLOAKED = 14

def get_foreground_hwnd():
    return win32gui.GetForegroundWindow()

def _capture_hwnd(hwnd):
    # PrintWindow + BitBlt fallback
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    if right <= 0 or bottom <= 0:
        raise RuntimeError("hwnd client rect non valido")
    w, h = right - left, bottom - top
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBmp = win32ui.CreateBitmap()
    saveBmp.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBmp)
    ok = _user32.PrintWindow(int(hwnd), saveDC.GetSafeHdc(), PW_RENDERFULLCONTENT)
    if not ok:
        # fallback: BitBlt
        saveDC.BitBlt((0,0), (w,h), mfcDC, (0,0), win32con.SRCCOPY)
    bmpinfo = saveBmp.GetInfo()
    bmpstr  = saveBmp.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # cleanup
    win32gui.DeleteObject(saveBmp.GetHandle()); saveDC.DeleteDC(); mfcDC.DeleteDC(); win32gui.ReleaseDC(hwnd, hwndDC)
    return img

def iter_window_frames(hwnd, target_hz=8.0, refresh_every=120):
    dt = 1.0 / max(1e-6, target_hz)
    t0 = time.time(); k = 0
    while True:
        # se la finestra non esiste più, solleva per gestirla sopra
        if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
            raise RuntimeError("Finestra non più disponibile")
        frame = _capture_hwnd(hwnd)
        yield frame
        k += 1
        # refresh periodico per evitare drift
        if refresh_every and (k % refresh_every) == 0:
            pass
        t = time.time() - t0
        sleep = dt * k - t
        if sleep > 0:
            time.sleep(sleep)

def _is_cloaked(hwnd):
    v = wintypes.DWORD()
    _dwmapi.DwmGetWindowAttribute(int(hwnd), DWMWA_CLOAKED,
                                  ctypes.byref(v), ctypes.sizeof(v))
    return v.value != 0

def _ext_rect(hwnd):
    r = wintypes.RECT()
    _dwmapi.DwmGetWindowAttribute(int(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS,
                                  ctypes.byref(r), ctypes.sizeof(r))
    return r.left, r.top, r.right, r.bottom

def _client_rect_screen(hwnd):
    l, t, r, b = win32gui.GetClientRect(hwnd)
    x, y = win32gui.ClientToScreen(hwnd, (0, 0))
    return x, y, x + (r - l), y + (b - t)

def _find_teams_hwnd():
    cands = []
    def enum_proc(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and not _is_cloaked(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and ("Teams" in title or "Microsoft Teams" in title):
                L,T,R,B = _ext_rect(hwnd)
                area = max(0, R-L) * max(0, B-T)
                if area > 200*200:
                    cands.append((area, hwnd))
        return True
    win32gui.EnumWindows(enum_proc, None)
    if not cands:
        raise RuntimeError("Finestra Teams non trovata")
    cands.sort(reverse=True)
    return cands[0][1]

def _grab_client(hwnd):
    wl, wt, wr, wb = _ext_rect(hwnd)
    cl, ct, cr, cb = _client_rect_screen(hwnd)
    w, h = wr - wl, wb - wt

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    bmp    = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(bmp)

    _user32.PrintWindow(int(hwnd), saveDC.GetSafeHdc(), PW_RENDERFULLCONTENT)

    buf  = bmp.GetBitmapBits(True)
    img4 = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    img  = img4[:, :, :3]  # BGRA->BGR

    # ritaglio alla client-area
    x0 = max(0, cl - wl); y0 = max(0, ct - wt)
    x1 = x0 + (cr - cl);  y1 = y0 + (cb - ct)
    img = img[y0:y1, x0:x1]

    win32gui.DeleteObject(bmp.GetHandle())
    saveDC.DeleteDC(); mfcDC.DeleteDC(); win32gui.ReleaseDC(hwnd, hwndDC)
    return img

def iter_teams_frames(target_hz=8.0, refresh_every=120):
    dt = 1.0 / max(0.1, float(target_hz))
    hwnd = _find_teams_hwnd()
    n = 0
    while True:
        try:
            frame = _grab_client(hwnd)
        except Exception:
            hwnd = _find_teams_hwnd()
            frame = _grab_client(hwnd)
        yield frame
        n += 1
        if n % refresh_every == 0:
            try: hwnd = _find_teams_hwnd()
            except Exception: pass
        time.sleep(dt)
