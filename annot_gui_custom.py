import os, sys, argparse, shutil, time
from pathlib import Path
try:
    from PIL import Image, ImageTk
except ImportError:
    print("[ERR] Pillow is required. Install with: pip install pillow", file=sys.stderr); sys.exit(1)
import tkinter as tk
from tkinter import messagebox, simpledialog

# ---------- helpers ----------
def read_npoints(root):
    lm = Path(root)/"tools/2_Landmarking_v1.0/LM_number.txt"
    try:
        n = int(lm.read_text(encoding="utf-8-sig").splitlines()[0].strip())
        return n if n>1 else None
    except Exception:
        return None

def list_images(png_dir):
    return sorted([str(p) for p in Path(png_dir).glob("*.png")])

def save_csv_wide(csv_path, pts):
    flat=[]
    for (x,y) in pts: flat += [f"{x:.2f}", f"{y:.2f}"]
    Path(csv_path).write_text(",".join(flat), encoding="utf-8")

def load_existing_csv(csv_path, N):
    p = Path(csv_path)
    if not p.exists(): return []
    try:
        txt = p.read_text(encoding="utf-8-sig")
        for line in [l.strip() for l in txt.splitlines() if l.strip()]:
            parts=[t.strip() for t in line.split(",") if t.strip()!=""]
            if len(parts)==2*N:
                vals=list(map(float,parts))
                return [(vals[i],vals[i+1]) for i in range(0,2*N,2)]
        return []
    except:
        return []

def move_to_bad(img_path):
    p=Path(img_path); bad=p.parent/"bad"; bad.mkdir(exist_ok=True)
    for f in [p, p.with_suffix(".csv")]:
        try:
            if f.exists(): shutil.move(str(f), str(bad/f.name))
        except: pass

# ---------- GUI ----------
class AnnotGUI(tk.Tk):
    def __init__(self, root, png_dir, n_points, start_from=None):
        super().__init__()
        self.title("GM Points Annotator - Custom")
        self.configure(background="black")
        self.root_dir = root
        self.png_dir = png_dir
        self.N = n_points or read_npoints(root) or 0
        if self.N<=1:
            messagebox.showerror("Error","LM_number.txt invalid or missing (N>1 required).")
            self.destroy(); sys.exit(2)

        # logging
        logs_dir = Path(root)/"tools/2_Landmarking_v1.0/logs"
        logs_dir.mkdir(exist_ok=True, parents=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = logs_dir/f"gui_{ts}.log"
        self.log_last = logs_dir/"annot_gui_last.log"
        self._zoom_hq_timer = None

        def _init_log():
            self._log("[BOOT] root=%s images=%s N=%s" % (root, png_dir, self.N))
        self._log = self._mk_logger()
        _init_log()

        self.images = list_images(png_dir)
        if not self.images:
            self._log("[INFO] No PNG images in %s" % png_dir)
            messagebox.showinfo("Empty","No PNG images found."); self.destroy(); sys.exit(0)

        self.idx = 0
        if start_from:
            sf = start_from.lower().replace(".png","")
            for i,fp in enumerate(self.images):
                if Path(fp).stem.lower()==sf or sf in Path(fp).stem.lower():
                    self.idx=i; break

        # widgets
        self.canvas = tk.Canvas(self, bg="gray10", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status = tk.Label(self, text="", anchor="w", bg="black", fg="white")
        self.status.pack(fill=tk.X)

        # state
        self.radius = 6
        self.zoom = 1.0
        self.offset = [0.0, 0.0]
        self.dragging_pt = None
        self.dragging_canvas = False
        self.photo = None
        self.img = None
        self.pts = []

        # bindings
        self.bind("<KeyPress>", self.on_key)
        self.bind("<KeyRelease>", self.on_key_up)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_right_press)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)  # Win
        self.bind("<Control-f>", self.search_dialog)
        self.bind("f", self.search_dialog)
        self.bind("<Control-s>", self.ctrl_save)   # manual save
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # first load (no autosave here)
        self._loaded = False
        self.load_image(self.images[self.idx])

    # ---------- logging ----------
    def _mk_logger(self):
        def log(msg):
            line = time.strftime("[%H:%M:%S] ") + msg + "\n"
            try:
                with open(self.log_file, "a", encoding="utf-8") as f: f.write(line)
                with open(self.log_last, "w", encoding="utf-8") as f: f.write(line)
            except: pass
        return log

    # ---------- coords ----------
    def d2i(self, dx, dy):  return ((dx - self.offset[0]) / self.zoom, (dy - self.offset[1]) / self.zoom)
    def i2d(self, ix, iy):  return ( ix * self.zoom + self.offset[0], iy * self.zoom + self.offset[1] )

    # ---------- image IO ----------
    def load_image(self, fp):
        self.img_path = fp
        try:
            self.img = Image.open(fp).convert("RGB")
        except Exception as e:
            self._log(f"[ERR] open {fp}: {e}"); messagebox.showerror("Error", str(e)); return
        self.zoom=1.0; self.offset=[0.0,0.0]
        csv_path = Path(fp).with_suffix(".csv")
        self.pts = load_existing_csv(csv_path, self.N)
        self.redraw(quality="hq")
        self.update_status()
        self._loaded = True
        self._log(f"[LOAD] {Path(fp).name} pts={len(self.pts)}")

    def save_current(self):
        csv_path = Path(self.img_path).with_suffix(".csv")
        try:
            if len(self.pts)>0:
                save_csv_wide(csv_path, self.pts)
                self._log(f"[SAVE] {csv_path.name} pts={len(self.pts)} OK")
            else:
                if csv_path.exists():
                    csv_path.unlink()
                    self._log(f"[SAVE] {csv_path.name} removed (0 pts)")
        except Exception as e:
            self._log(f"[ERR] save {csv_path.name}: {e}")
            messagebox.showerror("Save error", str(e))

    def auto_save_on_switch(self):
        if (len(self.pts)>1) and (len(self.pts)!=self.N):
            if not messagebox.askyesno("Mismatch",
                f"Landmark count mismatch: expected {self.N}, got {len(self.pts)}.\nContinue and save anyway?"):
                self._log("[MIS] user chose NO, stay on image")
                return False
        self.save_current()
        return True

    # ---------- UI helpers ----------
    def update_status(self):
        i = self.idx+1; T = len(self.images)
        loc = Path(self.png_dir).parent.name
        name = Path(self.img_path).name
        self.status.config(text=f"Image {i} of {T} | {name} | {loc} | +/- size {self.radius}px")
        self.title(f"GM Points Annotator - {name} [{i}/{T}]")

    def redraw(self, quality="fast"):
        self.canvas.delete("all")
        w,h = self.img.size
        if quality=="fast":
            res = Image.NEAREST
        else:
            res = Image.BILINEAR
        disp = self.img.resize((max(1,int(w*self.zoom)), max(1,int(h*self.zoom))), res)
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(self.offset[0], self.offset[1], anchor="nw", image=self.photo)
        for i,(x,y) in enumerate(self.pts, start=1):
            dx,dy = self.i2d(x,y); r = self.radius
            self.canvas.create_oval(dx-r, dy-r, dx+r, dy+r, outline="red", width=2)
            self.canvas.create_text(dx+r+6, dy, text=str(i), fill="yellow", anchor="w")

    # ---------- events ----------
    def on_left_click(self, e):
        j=self._pick_point(e.x, e.y)
        if j is not None:
            self.dragging_pt=j
        else:
            if len(self.pts) < self.N:
                ix,iy = self.d2i(e.x, e.y)
                self.pts.append((ix,iy)); self.redraw()

    def on_left_drag(self, e):
        if self.dragging_pt is not None:
            ix,iy = self.d2i(e.x, e.y)
            self.pts[self.dragging_pt]=(ix,iy); self.redraw()

    def on_left_release(self, e): self.dragging_pt=None

    def on_right_press(self, e): self.dragging_canvas=True; self._last=(e.x,e.y)
    def on_right_drag(self, e):
        if self.dragging_canvas:
            dx=e.x-self._last[0]; dy=e.y-self._last[1]
            self.offset[0]+=dx; self.offset[1]+=dy; self._last=(e.x,e.y); self.redraw(quality="fast")
    def on_right_release(self, e): self.dragging_canvas=False

    def on_wheel(self, e):
        if (e.state & 0x0004)==0:  # Ctrl must be held
            return
        factor = 1.1 if e.delta>0 else (1/1.1)
        mx,my = e.x, e.y
        ix,iy = self.d2i(mx,my)
        self.zoom = max(0.1, min(20.0, self.zoom*factor))
        dx,dy = self.i2d(ix,iy)
        self.offset[0] += (mx-dx); self.offset[1] += (my-dy)
        # fast redraw immediately, schedule HQ after 70 ms (debounce)
        self.redraw(quality="fast")
        if self._zoom_hq_timer:
            try: self.after_cancel(self._zoom_hq_timer)
            except: pass
        self._zoom_hq_timer = self.after(70, lambda: self.redraw(quality="hq"))

    def _pick_point(self, dx, dy, tol=8):
        best=None; bestd=1e9
        for j,(x,y) in enumerate(self.pts):
            px,py = self.i2d(x,y)
            d = (px-dx)**2 + (py-dy)**2
            if d<bestd and d <= (max(self.radius,tol))**2:
                best, bestd = j, d
        return best

    def on_key(self, e):
        k=e.keysym.lower()
        if k in ("plus","equal"):   self.radius=min(30,self.radius+1); self.redraw()
        elif k=="minus":            self.radius=max(2,self.radius-1); self.redraw()
        elif k=="right":            self.next_image()
        elif k=="left":             self.prev_image()
        elif k=="delete":
            x=self.winfo_pointerx()-self.winfo_rootx()
            y=self.winfo_pointery()-self.winfo_rooty()
            j=self._pick_point(x,y)
            if j is not None: self.pts.pop(j); self.redraw()
        elif k=="b" and (e.state & 0x0004):
            if messagebox.askyesno("Move to bad","Move this image (and CSV) to 'bad' folder?"):
                move_to_bad(self.img_path); self.after(10, self.next_image)
        elif k=="h":
            messagebox.showinfo("Help","Left/Right: prev/next\nLMB: add/drag\nRMB drag: pan\nCtrl+wheel: fast zoom\n+/-: size\nDelete: remove\nCtrl+B: move to bad\nCtrl+S: save\nF: search")
        self.update_status()

    def on_key_up(self, e): pass

    def ctrl_save(self, *_):
        self._log("[HOTKEY] Ctrl+S")
        self.save_current()

    def prev_image(self):
        if not self.auto_save_on_switch(): return
        if self.idx>0:
            self.idx-=1; self.load_image(self.images[self.idx])

    def next_image(self):
        if not self.auto_save_on_switch(): return
        if self.idx < len(self.images)-1:
            self.idx+=1; self.load_image(self.images[self.idx])

    def search_dialog(self, *_):
        q = simpledialog.askstring("Search","Enter filename (part, no extension):", parent=self)
        if not q: return
        q=q.lower()
        for i,fp in enumerate(self.images):
            stem=Path(fp).stem.lower()
            if (q in stem) or (stem==q):
                if not self.auto_save_on_switch(): return
                self.idx=i; self.load_image(self.images[self.idx]); return
        messagebox.showinfo("Not found", f"No file matching: {q}")

    def on_close(self):
        try:
            self._log("[CLOSE] autosave before exit")
            self.save_current()
        finally:
            self.destroy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--start-from", default=None)
    args=ap.parse_args()

    N = args.n_points or read_npoints(args.root)
    if not N or N<=1: print("[ERR] invalid N", file=sys.stderr); return 2
    app = AnnotGUI(args.root, args.images, N, start_from=args.start_from)
    app.geometry("1280x800+100+40")
    app.mainloop()
    return 0

if __name__=="__main__":
    sys.exit(main())