from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def get_landmark_root(arg_root: Optional[str]) -> Path:
    if arg_root:
        return Path(arg_root).resolve()
    return Path(__file__).resolve().parent.parent


def read_last_base(landmark_root: Path) -> Optional[Path]:
    cfg_path = landmark_root / "cfg" / "last_base.txt"
    if not cfg_path.exists():
        return None
    text = cfg_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return Path(text)


def read_lm_number(landmark_root: Path) -> int:
    """
    Read number of landmarks from LM_number.txt.
    Format: first non-empty line contains integer, comments (# ...) are ignored.
    """
    lm_path = landmark_root / "LM_number.txt"
    if not lm_path.exists():
        return 0
    for line in lm_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            return int(line)
        except ValueError:
            continue
    return 0


def write_dummy_csv(csv_path: Path, n_landmarks: int) -> None:
    """
    Create simple CSV with n_landmarks rows and columns x,y.
    All coordinates set to 0.0. This is a placeholder for real HRNet predictions.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for _ in range(max(n_landmarks, 1)):
            writer.writerow([0.0, 0.0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HRNet inference stub: create dummy CSV landmarks for each PNG."
    )
    parser.add_argument("--landmark-root", dest="root", default=None)
    parser.add_argument("--base-localities", dest="base", default=None)
    parser.add_argument("--locality", required=True)
    args = parser.parse_args()

    landmark_root = get_landmark_root(args.root)

    base_localities: Optional[Path]
    if args.base:
        base_localities = Path(args.base)
    else:
        base_localities = read_last_base(landmark_root)

    if base_localities is None:
        print("Base localities folder is not set (cfg/last_base.txt is empty).")
        return 1

    locality_name = args.locality
    png_dir = base_localities / locality_name / "png"
    if not png_dir.exists():
        print(f"Folder not found: {png_dir}")
        return 1

    png_list = sorted(png_dir.glob("*.png"))
    if not png_list:
        print(f"No *.png images found in {png_dir}")
        return 1

    n_landmarks = read_lm_number(landmark_root)
    if n_landmarks <= 0:
        print("LM_number.txt not found or invalid. Using 1 dummy landmark per image.")
        n_landmarks = 1

    print(
        f"Creating dummy landmark CSV files for locality \"{locality_name}\" "
        f"({len(png_list)} images, {n_landmarks} landmarks each)."
    )

    for img_path in png_list:
        csv_path = img_path.with_suffix(".csv")
        write_dummy_csv(csv_path, n_landmarks)

    print("Dummy autolabel finished (no real HRNet model yet).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
