import os, sys, argparse, shutil, time, math, statistics
from pathlib import Path
try:
    from PIL import Image, ImageTk
except Exception as e:
    print("[ERR] Pillow is required. Install with: pip install pillow", file=sys.stderr); sys.exit(1)
try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog
except Exception as e:
    print("[ERR] Tkinter is required. Detail:", e, file=sys.stderr); sys.exit(3)

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
    flat=[];  [flat.extend([f"{x:.2f}", f"{y:.2f}"]) for (x,y) in pts]
    Path(csv_path).write_text(",".join(flat), encoding="utf-8")

def load_existing_csv(csv_path, N):
    p=Path(csv_path)
    if not p.exists(): return []
    try:
        txt = p.read_text(encoding="utf-8-sig")
        for line in [l.strip() for l in txt.splitlines() if l.strip()]:
            parts=[t.strip() for t in line.split(",") if t.strip()!=""]
            if len(parts)==2*N:
                vals=list(map(float,parts))
                return [(vals[i],vals[i+1]) for i in range(0,2*N,2)]
        return []
    except: return []

def read_any_csv_points(csv_path):
    p=Path(csv_path)
    if not p.exists(): return []
    pts=[]
    try:
        txt = p.read_text(encoding="utf-8-sig")
        # читаем первую непустую строку
        line = next((l.strip() for l in txt.splitlines() if l.strip()), "")
        parts=[t.strip() for t in line.split(",") if t.strip()!=""]
        vals=list(map(float,parts))
        for i in range(0, len(vals)-1, 2):
            pts.append((vals[i], vals[i+1]))
    except: pass
    return pts

def move_to_bad(img_path):
    p=Path(img_path); bad=p.parent/"bad"; bad.mkdir(exist_ok=True)
    for f in [p, p.with_suffix(".csv")]:
        try:
            if f.exists(): shutil.move(str(f), str(bad/f.name))
        except: pass

# ---------- GPA ----------
def center_scale(pts):
    cx=sum(x for x,_ in pts)/len(pts); cy=sum(y for _,y in pts)/len(pts)
    centered=[(x-cx,y-cy) for x,y in pts]
    cs=math.sqrt(sum(x*x+y*y for x,y in centered))
    if cs==0: cs=1.0
    return [(x/cs,y/cs) for x,y in centered]

def rotate_to(A, B):
    # find optimal rotation R for A->B (both centered+scaled), return rotated A
    sxx = sum(ax*bx + ay*by for (ax,ay),(bx,by) in zip(A,B))
    syx = sum(ay*bx - ax*by for (ax,ay),(bx,by) in zip(A,B))
    r = math.hypot(sxx, syx); c = sxx/(r or 1.0); s = syx/(r or 1.0)
    return [(c*ax - s*ay, s*ax + c*ay) for (ax,ay) in A]

def mean_shape(shapes):
    # shapes: list of lists (N points), already centered+scaled+rotated ~ mean
    N=len(shapes[0])
    mx=[0.0]*N; my=[0.0]*N
    for sh in shapes:
        for i,(x,y) in enumerate(sh):
            mx[i]+=x; my[i]+=y
    k=len(shapes)
    return [(mx[i]/k, my[i]/k) for i in range(N)]

def gpa_align(shapes, iters=10):
    # return aligned shapes and consensus mean
    aligned=[center_scale(s) for s in shapes]
    m=mean_shape(aligned)
    for _ in range(iters):
        aligned=[rotate_to(center_scale(s), m) for s in shapes]
        new_m=mean_shape(aligned)
        # check delta
        d=sum((x1-x2)**2+(y1-y2)**2 for (x1,y1),(x2,y2) in zip(m,new_m))
        m=new_m
        if d<1e-8: break
    return aligned, m

# ---------- GUI ----------
class AnnotGUI(tk.Tk):
    def __init__(self, root, png_dir, n_points, start_from=None, scale_wizard=False):
        super().__init__()
        self.title("GM Points Annotator - Custom")
        self.configure(background="black")
        self.root_dir = root
        self.png_dir = png_dir
        self.N = n_points or read_npoints(root) or 0
        if self.N<=1:
            tk.messagebox.showerror("Error","LM_number.txt invalid or missing (N>1 required).")
            self.destroy(); sys.exit(2)

        # logging (last line only — перезаписываем)
        logs_dir = Path(root)/"tools/2_Landmarking_v1.0/logs"
        logs_dir.mkdir(exist_ok=True, parents=True)
        self.log_last = logs_dir/"annot_gui_last.log"
        self._log=lambda msg: self.log_last.write_text(time.strftime("[%H:%M:%S] ")+msg+"\n", encoding="utf-8")

        self.images = list_images(png_dir)
        if not self.images:
            tk.messagebox.showinfo("Empty","No PNG images found."); self.destroy(); sys.exit(0)

        self.idx = 0
        if start_from:
            sf = start_from.lower().replace(".png","")
            for i,fp in enumerate(self.images):
                if Path(fp).stem.lower()==sf or sf in Path(fp).stem.lower():
                    self.idx=i; break

        # widgets
        self.banner = tk.Label(self, text="", fg="#00FFFF", bg="black", font=("Segoe UI", 12, "bold"))
        self.banner.pack(fill=tk.X)
        self.canvas = tk.Canvas(self, bg="gray10", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status = tk.Label(self, text="", anchor="w", bg="black", fg="white")
        self.status.pack(fill=tk.X)
        self.font_lbl=("Segoe UI", 11, "bold")  # жирные подписи

        # state
        self.radius = 6
        self.zoom = 1.0
        self.offset = [0.0, 0.0]
        self.dragging_pt = None
        self.dragging_canvas = False
        self.photo = None
        self.img = None
        self.pts = []
        self._zoom_hq_timer=None

        # scale & QC
        self.scale_mode = bool(scale_wizard)
        self.scale_pts = []
        self.qc_mode = False
        self.qc_list = []   # [(index, reasons_str), ...]
        self.qc_pos = 0

        # bindings
        self.bind("<KeyPress>", self.on_key)
        self.bind("<KeyRelease>", self.on_key_up)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_right_press)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.bind("<Control-f>", self.search_dialog)
        self.bind("f", self.search_dialog)
        self.bind("<Control-s>", self.ctrl_save)
        self.bind("x", self.hot_move_to_bad); self.bind("X", self.hot_move_to_bad)
        self.bind("<F9>", self.toggle_qc)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # first load
        self._loaded=False
        self.load_image(self.images[self.idx])
        self.apply_banner()

    # ---------- coords ----------
    def d2i(self, dx, dy):  return ((dx - self.offset[0]) / self.zoom, (dy - self.offset[1]) / self.zoom)
    def i2d(self, ix, iy):  return ( ix * self.zoom + self.offset[0], iy * self.zoom + self.offset[1] )

    # ---------- IO ----------
    def load_image(self, fp):
        self.img_path = fp
        self.img = Image.open(fp).convert("RGB")
        self.zoom=1.0; self.offset=[0.0,0.0]
        if not self.scale_mode:
            self.pts = load_existing_csv(Path(fp).with_suffix(".csv"), self.N)
        self.redraw(quality="hq")
        self.update_status()
        self._loaded = True
        self._log(f"[LOAD] {Path(fp).name} pts={len(self.pts)} scale={self.scale_mode} qc={self.qc_mode}")

    def save_current(self):
        if self.scale_mode:
            if len(self.scale_pts)==2:
                p = Path(self.img_path); scale_csv = p.with_name(p.name + ".scale.csv")
                txt = f"{self.scale_pts[0][0]:.2f},{self.scale_pts[0][1]:.2f}\n{self.scale_pts[1][0]:.2f},{self.scale_pts[1][1]:.2f}\n"
                scale_csv.write_text(txt, encoding="utf-8")
                self._log(f"[SCALE] saved {scale_csv.name}")
        else:
            csv_path = Path(self.img_path).with_suffix(".csv")
            if len(self.pts)>0:
                save_csv_wide(csv_path, self.pts)
                self._log(f"[SAVE] {csv_path.name} pts={len(self.pts)} OK")
            else:
                if csv_path.exists():
                    csv_path.unlink()
                    self._log(f"[SAVE] {csv_path.name} removed (0 pts)")

    def auto_save_on_switch(self):
        if (not self.scale_mode) and (len(self.pts)>1) and (len(self.pts)!=self.N):
            if not tk.messagebox.askyesno("Mismatch",
                f"Landmark count mismatch: expected {self.N}, got {len(self.pts)}.\nContinue and save anyway?"):
                self._log("[MIS] user chose NO, stay on image")
                return False
        self.save_current()
        return True

    # ---------- QC ----------
    def toggle_qc(self, *_):
        if self.scale_mode:
            tk.messagebox.showinfo("Scale first","Finish Scale Wizard first.")
            return
        if not self.qc_mode:
            self.run_qc()
            if not self.qc_list:
                tk.messagebox.showinfo("QC","No issues detected.")
                return
            self.qc_mode=True; self.qc_pos=0
            # перейти к ближайшему проблемному (текущий или следующий)
            current = self.idx
            idxs=[i for i,_ in self.qc_list]
            start = min((i for i in range(len(idxs)) if idxs[i] >= current), default=0)
            self.qc_pos = start
            self.idx = idxs[self.qc_pos]
            self.load_image(self.images[self.idx])
        else:
            self.qc_mode=False
            self.apply_banner()
            self.redraw(quality="hq")

    def run_qc(self):
        """Собираем список проблемных кадров self.qc_list = [(index, reason_str), ...]"""
        N=self.N
        imgs=self.images
        any_shapes=[]; shapes_idx=[]
        problems=[]

        # 1) собираем и проверяем count
        for i,fp in enumerate(imgs):
            pts_any = read_any_csv_points(Path(fp).with_suffix(".csv"))
            if len(pts_any)==0:
                continue
            if len(pts_any)!=N:
                problems.append((i, f"count!=N (got {len(pts_any)})"))
            else:
                any_shapes.append(pts_any); shapes_idx.append(i)

        # 2) если есть валидные формы — GPA и поиск swap/outlier
        if len(any_shapes)>=2:
            aligned, mean = gpa_align(any_shapes, iters=10)
            # центроиды (mean) для каждой LM
            # медианные дистанции для порога выброса
            per_k_dists = [[] for _ in range(N)]
            for sh in aligned:
                for k,(x,y) in enumerate(sh):
                    mx,my = mean[k]
                    per_k_dists[k].append(math.hypot(x-mx, y-my))
            med = [statistics.median(d) if d else 0.0 for d in per_k_dists]
            thr = [max(0.05, 4*m) for m in med]  # базовый минимум

            for sh_idx,(i_img, sh) in enumerate(zip(shapes_idx, aligned)):
                reasons=[]
                for k,(x,y) in enumerate(sh):
                    # outlier?
                    mx,my = mean[k]
                    d_self = math.hypot(x-mx, y-my)
                    if d_self > thr[k]:
                        reasons.append(f"LM{k+1} outlier")
                        continue
                    # swap? ближе к центроиду другой точки, чем к своему
                    best_other=None; best_d=1e9
                    for j,(qx,qy) in enumerate(mean):
                        if j==k: continue
                        d = math.hypot(x-qx, y-qy)
                        if d<best_d: best_d, best_other = d, j
                    if best_d + 1e-6 < d_self:
                        reasons.append(f"LM{k+1} near LM{best_other+1} (swap?)")
                if reasons:
                    problems.append((i_img, "; ".join(reasons)))

        # 3) отсортируем по индексу и уникализируем по кадру (объединим причины)
        problems.sort(key=lambda x: x[0])
        merged=[]
        last_i=-1; acc=[]
        for i,reason in problems:
            if i!=last_i:
                if acc:
                    merged.append((last_i, "; ".join(acc)))
                last_i=i; acc=[reason]
            else:
                acc.append(reason)
        if acc:
            merged.append((last_i, "; ".join(acc)))
        self.qc_list = merged
        # лог файл
        log_path = Path(self.root_dir)/"tools/2_Landmarking_v1.0/logs"/"qc_last.txt"
        lines=[f"{Path(self.images[i]).name}: {r}" for i,r in self.qc_list]
        log_path.write_text("\n".join(lines), encoding="utf-8")

    # ---------- UI helpers ----------
    def apply_banner(self):
        if self.scale_mode:
            self.banner.config(text="SCALE MODE — place TWO cyan squares on a 10 mm segment, then press Enter.", fg="#00FFFF")
            self.banner.pack_configure(fill=tk.X)
        elif self.qc_mode and self.qc_list:
            i,r = self.qc_list[self.qc_pos]
            left = len(self.qc_list)-self.qc_pos
            self.banner.config(text=f"QC MODE — issues left: {left}. {Path(self.images[i]).name}: {r}", fg="#FF5555")
            self.banner.pack_configure(fill=tk.X)
        else:
            self.banner.config(text="")
            self.banner.pack_forget()

    def update_status(self):
        i = self.idx+1; T = len(self.images)
        loc = Path(self.png_dir).parent.name
        name = Path(self.img_path).name
        extra = ""
        if self.scale_mode: extra=" | SCALE MODE (10 mm)"
        elif self.qc_mode:  extra=" | QC MODE (F9 to exit)"
        else:               extra=f" | +/- size {self.radius}px | X/Ctrl+B: move to bad"
        self.status.config(text=f"Image {i} of {T} | {name} | {loc}{extra}")
        self.title(f"GM Points Annotator - {name} [{i}/{T}]")

    def _draw_label(self, dx, dy, text, color="yellow"):
        for ox,oy in ((-1,0),(1,0),(0,-1),(0,1)):
            self.canvas.create_text(dx+ox, dy+oy, text=text, fill="black", anchor="w", font=self.font_lbl)
        self.canvas.create_text(dx, dy, text=text, fill=color, anchor="w", font=self.font_lbl)

    def redraw(self, quality="fast"):
        self.canvas.delete("all")
        w,h = self.img.size
        res = Image.NEAREST if quality=="fast" else Image.BILINEAR
        disp = self.img.resize((max(1,int(w*self.zoom)), max(1,int(h*self.zoom))), res)
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(self.offset[0], self.offset[1], anchor="nw", image=self.photo)

        if self.scale_mode:
            for i,(x,y) in enumerate(self.scale_pts, start=1):
                dx,dy=self.i2d(x,y); r=self.radius+2
                self.canvas.create_rectangle(dx-r, dy-r, dx+r, dy+r, outline="#00FFFF", width=2)
                self._draw_label(dx+r+6, dy, f"S{i}", color="#00FFFF")
        else:
            for i,(x,y) in enumerate(self.pts, start=1):
                dx,dy=self.i2d(x,y); r=self.radius
                self.canvas.create_oval(dx-r, dy-r, dx+r, dy+r, outline="red", width=2)
                self._draw_label(dx+r+6, dy, str(i))
        self.apply_banner()

    # ---------- events ----------
    def on_left_click(self, e):
        if self.scale_mode:
            if len(self.scale_pts) < 2:
                ix,iy = self.d2i(e.x, e.y)
                self.scale_pts.append((ix,iy)); self.redraw(); return
            # drag nearest in scale mode
            j=self._pick(self.scale_pts, e.x, e.y)
            if j is not None: self.dragging_pt=j; return
        j=self._pick(self.pts, e.x, e.y)
        if j is not None:
            self.dragging_pt=j
        else:
            if len(self.pts) < self.N:
                ix,iy = self.d2i(e.x, e.y)
                self.pts.append((ix,iy)); self.redraw()

    def on_left_drag(self, e):
        if self.dragging_pt is not None:
            arr = self.scale_pts if self.scale_mode else self.pts
            ix,iy = self.d2i(e.x, e.y); arr[self.dragging_pt]=(ix,iy); self.redraw()

    def on_left_release(self, e): self.dragging_pt=None
    def on_right_press(self, e): self.dragging_canvas=True; self._last=(e.x,e.y)
    def on_right_drag(self, e):
        if self.dragging_canvas:
            dx=e.x-self._last[0]; dy=e.y-self._last[1]
            self.offset[0]+=dx; self.offset[1]+=dy; self._last=(e.x,e.y); self.redraw("fast")
    def on_right_release(self, e): self.dragging_canvas=False

    def on_wheel(self, e):
        if (e.state & 0x0004)==0: return
        factor = 1.1 if e.delta>0 else (1/1.1)
        mx,my = e.x, e.y
        ix,iy = self.d2i(mx,my)
        self.zoom = max(0.1, min(20.0, self.zoom*factor))
        dx,dy = self.i2d(ix,iy)
        self.offset[0] += (mx-dx); self.offset[1] += (my-dy)
        self.redraw("fast")
        if self._zoom_hq_timer:
            try: self.after_cancel(self._zoom_hq_timer)
            except: pass
        self._zoom_hq_timer = self.after(70, lambda: self.redraw("hq"))

    def _pick(self, arr, dx, dy, tol=8):
        best=None; bestd=1e9
        for j,(x,y) in enumerate(arr):
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
        elif k=="delete" and (not self.scale_mode):
            x=self.winfo_pointerx()-self.winfo_rootx()
            y=self.winfo_pointery()-self.winfo_rooty()
            j=self._pick(self.pts, x, y)
            if j is not None: self.pts.pop(j); self.redraw()
        elif k=="b" and (e.state & 0x0004) and (not self.scale_mode):
            move_to_bad(self.img_path); self.after(10, self.next_image)
        elif k=="h":
            tk.messagebox.showinfo("Help",
                "F9: toggle QC\n"
                "Left/Right: prev/next\n"
                "LMB: add/drag point (or scale)\n"
                "RMB drag: pan\n"
                "Ctrl+wheel: zoom (fast->HQ)\n"
                "+/-: point size\n"
                "Delete: remove point\n"
                "X/Ctrl+B: move to bad (not in scale)\n"
                "Ctrl+S: save\n"
                "Enter (Scale): save scale\n"
                "F / Ctrl+F: search")
        self.update_status()

    def on_key_up(self, e): pass

    def ctrl_save(self, *_):
        self.save_current(); self._log("[HOTKEY] Ctrl+S -> saved")

    def prev_image(self):
        if not self.auto_save_on_switch(): return
        if self.qc_mode and self.qc_list:
            if self.qc_pos>0:
                self.qc_pos-=1; self.idx=self.qc_list[self.qc_pos][0]; self.load_image(self.images[self.idx]); return
        if self.idx>0:
            self.idx-=1; self.load_image(self.images[self.idx])

    def next_image(self):
        if not self.auto_save_on_switch(): return
        if self.qc_mode and self.qc_list:
            if self.qc_pos < len(self.qc_list)-1:
                self.qc_pos+=1; self.idx=self.qc_list[self.qc_pos][0]; self.load_image(self.images[self.idx]); return
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
        tk.messagebox.showinfo("Not found", f"No file matching: {q}")

    def hot_move_to_bad(self, *_):
        if not self.scale_mode:
            move_to_bad(self.img_path); self.after(10, self.next_image)

    def on_close(self):
        try:
            self.save_current()
        finally:
            self.destroy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--start-from", default=None)
    ap.add_argument("--scale-wizard", action="store_true")
    a=ap.parse_args()

    N = a.n_points or read_npoints(a.root)
    if not N or N<=1: print("[ERR] invalid N", file=sys.stderr); return 2
    app = AnnotGUI(a.root, a.images, N, start_from=a.start_from, scale_wizard=a.scale_wizard)
    app.geometry("1280x800+100+40")
    app.mainloop()
    return 0

if __name__=="__main__":
    sys.exit(main())