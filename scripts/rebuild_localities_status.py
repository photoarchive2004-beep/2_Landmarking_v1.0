from pathlib import Path
import csv
import sys
from datetime import datetime


def find_tool_dir() -> Path:
    # scripts/ -> parent = tools/2_Landmarking_v1.0
    return Path(__file__).resolve().parent.parent


def load_base_localities(tool_dir: Path) -> Path | None:
    cfg_dir = tool_dir / "cfg"
    last_base = cfg_dir / "last_base.txt"
    if not last_base.exists():
        print(
            "[ERR] cfg/last_base.txt not found; run 1_ANNOTATOR.bat or 2_TRAIN-INFER_HRNet.bat first.",
            file=sys.stderr,
        )
        return None

    base = last_base.read_text(encoding="utf-8").strip()
    if not base:
        print("[ERR] cfg/last_base.txt is empty.", file=sys.stderr)
        return None

    path = Path(base)
    if not path.exists():
        print(f"[ERR] Localities base not found: {path}", file=sys.stderr)
        return None

    return path


def scan_localities(base_dir: Path):
    """
    Return list of (locality_name, n_images, n_labeled).

    n_labeled считается по принципу:
    для каждого img_XXXX.png считаем размеченным, если рядом есть файл img_XXXX.csv.
    Файлы *.scale.csv, *.pass2.csv и прочие дополнения не учитываются.
    """
    results = []
    for loc_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        png_dir = loc_dir / "png"
        if not png_dir.is_dir():
            continue

        images = sorted(png_dir.glob("*.png"))
        n_images = len(images)
        if n_images == 0:
            continue

        n_labeled = 0
        for img in images:
            lm = png_dir / f"{img.stem}.csv"
            if lm.exists():
                n_labeled += 1

        results.append((loc_dir.name, n_images, n_labeled))

    return results


def read_old_status(status_file: Path):
    if not status_file.exists():
        return {}
    with status_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out = {}
        for row in reader:
            loc = row.get("locality") or row.get("name")
            if not loc:
                continue
            out[loc] = row
        return out


def write_status(status_file: Path, rows):
    fieldnames = [
        "locality",
        "status",
        "auto_quality",
        "last_model_run",
        "last_update",
        "n_images",
        "n_labeled",
    ]
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with status_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    tool_dir = find_tool_dir()
    base_dir = load_base_localities(tool_dir)
    if base_dir is None:
        return 1

    status_file = tool_dir / "status" / "localities_status.csv"
    old = read_old_status(status_file)
    scanned = scan_localities(base_dir)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_rows = []
    for name, n_images, n_labeled in scanned:
        prev = old.get(name, {})
        status = prev.get("status", "")
        auto_quality = prev.get("auto_quality", "")
        last_model_run = prev.get("last_model_run", "")
        last_update = prev.get("last_update", "")

        # Новая локальность: расставлены все точки -> сразу MANUAL
        if name not in old:
            if n_images > 0 and n_labeled == n_images:
                status = "MANUAL"

        new_rows.append(
            {
                "locality": name,
                "status": status,
                "auto_quality": auto_quality,
                "last_model_run": last_model_run,
                "last_update": last_update or now,
                "n_images": str(n_images),
                "n_labeled": str(n_labeled),
            }
        )

    write_status(status_file, new_rows)
    print(f"[INFO] Rebuilt status for {len(new_rows)} localities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
