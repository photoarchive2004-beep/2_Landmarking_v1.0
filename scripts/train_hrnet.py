from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def get_root() -> Path:
    """Pipeline root = parent of scripts/."""
    return Path(__file__).resolve().parent.parent


def load_status(root: Path) -> List[Dict[str, str]]:
    """Read status/localities_status.csv. Return list of rows."""
    csv_path = root / "status" / "localities_status.csv"
    rows: List[Dict[str, str]] = []

    if not csv_path.exists():
        print("[ERR] status/localities_status.csv not found.")
        print("      First run option 2 in HRNet Trainer to rescan localities.")
        return rows

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("locality"):
                continue
            rows.append(row)
    return rows


def safe_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def main() -> None:
    root = get_root()
    print("=== GM Landmarking: HRNet training dataset (stub) ===\n")

    rows = load_status(root)
    if not rows:
        input("Press Enter to exit...")
        return

    manual_rows: List[Dict[str, str]] = []
    for r in rows:
        status = (r.get("status") or "").strip().upper()
        if status == "MANUAL":
            manual_rows.append(r)

    if not manual_rows:
        print("[INFO] No MANUAL localities in status/localities_status.csv.")
        print("       Mark some localities as MANUAL first (in annotator or status file).")
        input("Press Enter to exit...")
        return

    # Try to read last base path (where photos are stored)
    base_txt = root / "cfg" / "last_base.txt"
    base_dir: Path | None = None
    if base_txt.exists():
        try:
            txt = base_txt.read_text(encoding="utf-8").strip()
            if txt:
                base_dir = Path(txt)
        except Exception:
            base_dir = None

    print("[INFO] Localities used for training (status = MANUAL):")
    dataset_lines: List[str] = []
    for r in manual_rows:
        loc = r.get("locality", "").strip()
        n_images = safe_int((r.get("n_images") or "").strip() or "0")
        n_labeled = safe_int((r.get("n_labeled") or "").strip() or "0")
        # защита от процентов > 100
        if n_labeled > n_images:
            n_eff = n_images
        else:
            n_eff = n_labeled
        percent = int(round(100.0 * n_eff / n_images)) if n_images > 0 else 0
        line = f"{loc}: images={n_images}, labeled={n_labeled}, {percent}%"
        dataset_lines.append(line)
        print("  -", line)

    # Save simple summary to logs
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = logs_dir / f"train_dataset_{ts}.txt"
    header = [
        "HRNet training dataset summary",
        f"root: {root}",
        f"base photos dir (from cfg/last_base.txt): {base_dir!s}",
        "",
    ]
    out_path.write_text("\n".join(header + dataset_lines) + "\n", encoding="utf-8")

    print()
    print(f"[INFO] Dataset summary saved to: {out_path}")
    print()
    print("NOTE: this script currently PREPARES the dataset only.")
    print("      Integration with real HRNet / MMPose training will be added later,")
    print("      after we finish all details of the training environment and config.")
    print()
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
