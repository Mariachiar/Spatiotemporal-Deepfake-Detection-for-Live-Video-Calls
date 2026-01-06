# capture_tile.py
import time, ctypes, numpy as np, cv2, mss
from ctypes import wintypes
import win32gui, win32con, win32api




ctypes.windll.user32.SetProcessDPIAware()
_dwmapi = ctypes.windll.dwmapi
DWMWA_CLOAKED = 14
VK_F8 = 0x77

def is_cloaked(h):
    v = wintypes.DWORD()
    if _dwmapi.DwmGetWindowAttribute(int(h), DWMWA_CLOAKED, ctypes.byref(v), ctypes.sizeof(v)) == 0:
        return v.value != 0
    return False

def is_valid(h):
    return h and win32gui.IsWindow(h) and win32gui.IsWindowVisible(h) and not is_cloaked(h)

def rect_area(h):
    try:
        l,t,r,b = win32gui.GetWindowRect(h); return max(0,r-l)*max(0,b-t)
    except: return 0

RENDER_CLASSES = {"TeamsWebView","Chrome_WidgetWin_1","Intermediate D3D Window",
                  "Windows.UI.Core.CoreWindow","CefBrowserWindow","Xaml_WindowedPopupClass",
                  "XAML Island Host","Windows.UI.Composition.DesktopWindowContentBridge"}

def find_render_child(parent):
    best=parent; best_a=rect_area(parent); best_cls=win32gui.GetClassName(parent)
    def _enum(h,_):
        nonlocal best,best_a,best_cls
        if not is_valid(h): return
        try:
            cls=win32gui.GetClassName(h); a=rect_area(h)
            if (cls in RENDER_CLASSES or "renderer" in cls.lower() or a>best_a*0.9) and a>best_a:
                best,best_a,best_cls=h,a,cls
        except: pass
    try: win32gui.EnumChildWindows(parent,_enum,None)
    except: pass
    print(f"[target] hwnd={best} class='{best_cls}' area={best_a}")
    return best

def client_rect_screen(h):
    if not is_valid(h): return None
    try:
        rc=win32gui.GetClientRect(h); x,y=win32gui.ClientToScreen(h,(0,0))
        l,t,r,b=x,y,x+(rc[2]-rc[0]), y+(rc[3]-rc[1])
        return None if r<=l or b<=t else (l,t,r,b)
    except: return None

class LargestTilePicker:
    def __init__(self): self.prev_gray=None; self.prev_tile=None; self.cool=0
    @staticmethod
    def _valid(w,h,x,y,ww,hh):
        if ww<200 or hh<120: return False
        ar=ww/float(hh); return 1.2<=ar<=2.2 and ww*hh>=0.10*w*h
    def _tiles(self, frame):
        H,W=frame.shape[:2]
        g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); g=cv2.GaussianBlur(g,(5,5),0)
        e=cv2.Canny(g,50,150); e=cv2.dilate(e,np.ones((3,3),np.uint8),1)
        cnts,_=cv2.findContours(e,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cand=[]
        for c in cnts:
            x,y,w,h=cv2.boundingRect(c)
            if not self._valid(W,H,x,y,w,h): continue
            roi=g[y+4:y+h-4,x+4:x+w-4]
            if roi.size==0 or roi.var()<50: continue
            cand.append((w*h,(x,y,x+w,y+h)))
        if not cand: return None
        cand.sort(reverse=True); return cand[0][1]
    def _motion(self, frame):
        H,W=frame.shape[:2]
        g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None: self.prev_gray=g.copy(); return None
        diff=cv2.absdiff(g,self.prev_gray); self.prev_gray=g
        _,th=cv2.threshold(diff,16,255,cv2.THRESH_BINARY)
        th=cv2.medianBlur(th,5); th=cv2.dilate(th,np.ones((5,5),np.uint8),2)
        cnts,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        x1,y1,x2,y2=W,H,0,0
        for c in cnts:
            x,y,w,h=cv2.boundingRect(c)
            if w*h<0.01*W*H: continue
            x1=min(x1,x); y1=min(y1,y); x2=max(x2,x+w); y2=max(y2,y+h)
        if x2<=x1 or y2<=y1: return None
        ar=16/9; ww,hh=x2-x1,y2-y1
        if ww/hh>ar:
            nh=int(ww/ar); cy=(y1+y2)//2; y1=max(0,cy-nh//2); y2=min(H,y1+nh)
        else:
            nw=int(hh*ar); cx=(x1+x2)//2; x1=max(0,cx-nw//2); x2=min(W,x1+nw)
        return (x1,y1,x2,y2)
    def pick(self, frame):
        box=self._tiles(frame) or self._motion(frame)
        if box is None:
            if self.prev_tile is not None and self.cool>0:
                self.cool-=1; return self.prev_tile
            return (0,0,frame.shape[1],frame.shape[0])
        if self.prev_tile is not None:
            a=0.6
            L=int(a*self.prev_tile[0]+(1-a)*box[0])
            T=int(a*self.prev_tile[1]+(1-a)*box[1])
            R=int(a*self.prev_tile[2]+(1-a)*box[2])
            B=int(a*self.prev_tile[3]+(1-a)*box[3])
            box=(L,T,R,B)
        self.prev_tile=box; self.cool=10; return box

def _monitors():
    return win32api.EnumDisplayMonitors()

def _place_preview_away(preview_title, target_ltbr):
    """Sposta la finestra 'preview_title' fuori da target_ltbr. Se possibile su altro monitor."""
    phwnd = win32gui.FindWindow(None, preview_title)
    if not phwnd: return
    L,T,R,B = target_ltbr
    w,h = 640, 360  # dimensione suggerita
    # prova altro monitor
    mons = _monitors()
    if len(mons) > 1:
        # cerca un monitor che non intersechi
        for i,(_,_,mr) in enumerate(mons):
            ml,mt,mr_,mb = mr
            if R <= ml or mr_ <= L or B <= mt or mb <= T:
                x = ml + 20
                y = mt + 20
                win32gui.SetWindowPos(phwnd, win32con.HWND_TOPMOST, x, y, w, h, 0)
                return
    # altrimenti scegli un angolo libero sullo stesso monitor
    sw = ctypes.windll.user32.GetSystemMetrics(0)
    sh = ctypes.windll.user32.GetSystemMetrics(1)
    cand = [(10,10),(sw-w-10,10),(10,sh-h-10),(sw-w-10,sh-h-10)]
    def inter(pr):
        x,y=pr; l2,t2,r2,b2=x,y,x+w,y+h
        return not (r2<=L or R<=l2 or b2<=T or B<=t2)
    for x,y in cand:
        if not inter((x,y)):
            win32gui.SetWindowPos(phwnd, win32con.HWND_TOPMOST, x, y, w, h, 0)
            return
    # fallback: mettila subito sopra il bordo alto
    x = max(0, min(sw-w-10, L+10))
    y = max(0, T-h-20)
    win32gui.SetWindowPos(phwnd, win32con.HWND_TOPMOST, x, y, w, h, 0)

def iter_roi_frames(preview_title="Teams ROI", max_fps: float = 30.0, max_w: int = 960):
    """Generatore: premi F8 sul riquadro Teams. Evita la preview durante la cattura."""
    child = None
    picker = LargestTilePicker()
    t_last = 0.0
    min_dt = 1.0 / float(max(1.0, max_fps))

    with mss.mss() as sct:
        while True:
            if win32api.GetAsyncKeyState(VK_F8) & 1:
                mx, my = win32api.GetCursorPos()
                h = win32gui.WindowFromPoint((mx, my))
                while win32gui.GetParent(h):
                    h = win32gui.GetParent(h)
                if h and win32gui.IsIconic(h):
                    win32gui.ShowWindow(h, win32con.SW_RESTORE)
                child = find_render_child(h) if h else None

            if not child or not win32gui.IsWindow(child):
                yield None, None
                time.sleep(0.05)
                continue

            rc = win32gui.GetClientRect(child)
            if not rc:
                yield None, None
                time.sleep(0.02)
                continue

            x0, y0 = win32gui.ClientToScreen(child, (0, 0))
            L, T = x0, y0
            R, B = x0 + (rc[2] - rc[0]), y0 + (rc[3] - rc[1])

            _place_preview_away(preview_title, (L, T, R, B))

            mon = {"left": L, "top": T, "width": R - L, "height": B - T}
            try:
                shot = sct.grab(mon)
            except Exception:
                yield None, None
                time.sleep(0.01)
                continue

            frame = cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)

            now = time.time()
            dt = now - t_last
            if dt < min_dt:
                time.sleep(min_dt - dt)
                now = time.time()
            t_last = now

            if max_w > 0 and frame.shape[1] > max_w:
                new_h = int(frame.shape[0] * (max_w / frame.shape[1]))
                frame = cv2.resize(frame, (max_w, new_h), interpolation=cv2.INTER_AREA)

            l, t, r, b = picker.pick(frame)
            l = max(0, min(frame.shape[1] - 1, l))
            t = max(0, min(frame.shape[0] - 1, t))
            r = max(l + 1, min(frame.shape[1], r))
            b = max(t + 1, min(frame.shape[0], b))
            yield frame[t:b, l:r], (l, t, r, b)
