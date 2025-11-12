import os, sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TOOL = os.path.join(ROOT, "tools", "2_Landmarking_v1.0")
PHOTOS = os.path.join(ROOT, "photos")
LM_FILE = os.path.join(TOOL, "LM_number.txt")

def read_npoints():
    try:
        with open(LM_FILE, "r", encoding="utf-8-sig") as f:
            s = f.readline().strip()
        n = int(s)
        return n if n > 1 else None
    except:
        return None

def parse_wide_nums(line):
    vals = []
    for t in [x.strip() for x in line.split(",") if x.strip()]:
        try: vals.append(float(t))
        except: pass
    return vals

def image_is_done(img_path, N):
    base = os.path.splitext(os.path.basename(img_path))[0]
    csv_path = os.path.join(os.path.dirname(img_path), base + ".csv")
    if not os.path.exists(csv_path): 
        return False
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            rows = [r.strip() for r in f if r.strip()]
        if not rows: return False
        for line in rows:
            nums = parse_wide_nums(line)
            if N and len(nums) == 2 * N:
                return True
        return False
    except:
        return False

def has_any_scale(img_dir):
    try:
        for fn in os.listdir(img_dir):
            if fn.lower().endswith(".scale.csv"):
                return True
    except: pass
    return False

def collect_localities():
    if not os.path.isdir(PHOTOS):
        return []
    return [d for d in sorted(os.listdir(PHOTOS))
            if os.path.isdir(os.path.join(PHOTOS, d))]

def get_images_dir(loc):
    # предпочтительно <loc>\png, иначе сам <loc>
    p_png = os.path.join(PHOTOS, loc, "png")
    p_loc = os.path.join(PHOTOS, loc)
    def count_png(p):
        try:
            return len([f for f in os.listdir(p) if f.lower().endswith(".png")])
        except: return 0
    if os.path.isdir(p_png):
        if count_png(p_png) > 0: return p_png
        # даже если пусто, используем png как целевой каталог
        return p_png
    return p_loc

def list_pngs(img_dir):
    if not os.path.isdir(img_dir): return []
    try:
        return [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                if f.lower().endswith(".png")]
    except:
        return []

def compute_status(loc, N):
    img_dir = get_images_dir(loc)
    pngs = list_pngs(img_dir)
    total = len(pngs)
    done = sum(image_is_done(p, N) for p in pngs) if (N and total) else 0
    pct = int(round((done / total * 100), 0)) if total else 0
    need_scale = (total and done == total and not has_any_scale(img_dir))
    return total, done, pct, need_scale, img_dir

def main():
    args = sys.argv[1:]
    N = read_npoints()
    locs = collect_localities()

    if "--print" in args or not args:
        if not locs:
            print("No localities found in 'photos'.")
            return 0
        for i, loc in enumerate(locs, 1):
            total, done, pct, need_scale, _ = compute_status(loc, N)
            line = f"{i}) {loc} [{done}/{total}] {pct}%"
            if need_scale: line += "  Set Scale!"
            print(line)
        print("Enter a number to open locality, or Q to quit:")
        return 0

    if "--pick" in args:
        try:
            idx = args.index("--pick")
            num = int(args[idx+1])
        except Exception:
            return 2
        if 1 <= num <= len(locs):
            print(locs[num-1], end="")
            return 0
        return 2

    if "--imgpath" in args:
        try:
            idx = args.index("--imgpath")
            loc = args[idx+1]
        except Exception:
            return 2
        if loc in locs:
            print(get_images_dir(loc), end="")
            return 0
        return 2

    return 0

if __name__ == "__main__":
    sys.exit(main())