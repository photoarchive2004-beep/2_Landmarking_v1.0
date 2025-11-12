import os, sys, math

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TOOL = os.path.abspath(HERE)
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
        try:
            vals.append(float(t))
        except:
            pass
    return vals

def image_is_done(png_path, N):
    base = os.path.splitext(os.path.basename(png_path))[0]
    csv_path = os.path.join(os.path.dirname(png_path), base + ".csv")
    if not os.path.exists(csv_path): 
        return False
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            rows = [r.strip() for r in f if r.strip()]
        if not rows:
            return False
        # если есть строка с числами на любой строке — примем её за данные
        for line in rows:
            nums = parse_wide_nums(line)
            if len(nums) == 2 * N:
                return True
        return False
    except:
        return False

def has_any_scale(png_dir):
    try:
        for fn in os.listdir(png_dir):
            if fn.lower().endswith(".scale.csv"):
                return True
    except:
        pass
    return False

def main():
    if not os.path.isdir(PHOTOS):
        print("[ERR] Photos dir not found:", PHOTOS, file=sys.stderr)
        sys.exit(1)
    N = read_npoints()
    locs = []
    names = sorted([d for d in os.listdir(PHOTOS) if os.path.isdir(os.path.join(PHOTOS, d, "png"))])
    for d in names:
        png_dir = os.path.join(PHOTOS, d, "png")
        pngs = [os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.lower().endswith(".png")]
        total = len(pngs)
        done = sum(image_is_done(p, N) for p in pngs) if (N and total) else 0
        pct = int(round((done / total * 100), 0)) if total else 0
        scale_flag = ""
        if total and done == total and not has_any_scale(png_dir):
            scale_flag = "  Set Scale!"
        print(f"{len(locs)+1}) {d} [{done}/{total}] {pct}%{scale_flag}")
        locs.append((d, png_dir))
    if not locs:
        print("No localities found in 'photos'. Press Enter to exit.")
        input()
        sys.exit(0)
    print("Enter a number to open locality, or Q to quit: ", end="", flush=True)
    choice = input().strip()
    if choice.lower() == "q":
        sys.exit(0)
    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(locs)):
            raise ValueError
    except:
        sys.exit(2)
    # Печатаем ТОЛЬКО имя локальности — батник подхватит это как выбор
    print(locs[idx][0], end="")
    sys.exit(0)

if __name__ == "__main__":
    main()