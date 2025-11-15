from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


STATUS_FILE = "status/localities_status.csv"


@dataclass
class Locality:
    locality: str
    status: str
    auto_quality: str
    last_model_run: str
    last_update: str
    n_images: int
    n_labeled: int


def get_landmark_root(arg_root: Optional[str]) -> Path:
    if arg_root:
        return Path(arg_root).resolve()
    # tools/2_Landmarking_v1.0/scripts/trainer_menu.py -> tools/2_Landmarking_v1.0
    return Path(__file__).resolve().parent.parent


def read_last_base(landmark_root: Path) -> Optional[Path]:
    cfg_path = landmark_root / "cfg" / "last_base.txt"
    if not cfg_path.exists():
        return None
    text = cfg_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return Path(text)


def load_localities(landmark_root: Path) -> List[Locality]:
    status_path = landmark_root / STATUS_FILE
    if not status_path.exists():
        return []
    rows: List[Locality] = []
    with status_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n_images = int(row.get("n_images", "") or 0)
            except ValueError:
                n_images = 0
            try:
                n_labeled = int(row.get("n_labeled", "") or 0)
            except ValueError:
                n_labeled = 0
            rows.append(
                Locality(
                    locality=row.get("locality", ""),
                    status=row.get("status", ""),
                    auto_quality=row.get("auto_quality", ""),
                    last_model_run=row.get("last_model_run", ""),
                    last_update=row.get("last_update", ""),
                    n_images=n_images,
                    n_labeled=n_labeled,
                )
            )
    return rows


def save_localities(landmark_root: Path, localities: List[Locality]) -> None:
    status_path = landmark_root / STATUS_FILE
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "locality",
            "status",
            "auto_quality",
            "last_model_run",
            "last_update",
            "n_images",
            "n_labeled",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for loc in localities:
            writer.writerow(
                {
                    "locality": loc.locality,
                    "status": loc.status,
                    "auto_quality": loc.auto_quality,
                    "last_model_run": loc.last_model_run,
                    "last_update": loc.last_update,
                    "n_images": loc.n_images,
                    "n_labeled": loc.n_labeled,
                }
            )


def format_status(loc: Locality) -> str:
    if loc.status == "MANUAL":
        return "MANUAL"
    if loc.status == "AUTO":
        if loc.auto_quality:
            return f"AUTO {loc.auto_quality}"
        return "AUTO"
    return ""


def calc_percent(loc: Locality) -> int:
    if loc.n_images <= 0:
        return 0
    return int(round(100.0 * loc.n_labeled / max(1, loc.n_images)))


def print_localities_block(localities: List[Locality]) -> None:
    print()
    print("Localities (photos/<locality>/png):")
    print()
    if not localities:
        print("  (no localities found)")
        return

    # Выравнивание: аккуратные столбцы имени, статуса и счётчиков
    max_name = max((len(loc.locality) for loc in localities), default=0)
    name_width = max(max_name + 2, 24)  # минимум 24 символа
    count_width = 16

    for idx, loc in enumerate(localities, start=1):
        percent = calc_percent(loc)
        status_text = format_status(loc)
        # [N] + пробелы
        prefix = f"[{idx}] "
        name_part = f"{prefix}{loc.locality}".ljust(len(prefix) + name_width)
        count_part = f"[{loc.n_labeled}/{loc.n_images}] {percent:3d}%".ljust(count_width)
        # статус в отдельном столбце
        status_col = status_text if status_text else ""
        line = f"{name_part}{status_col:10s}{count_part}"
        print(line)


def run_training(landmark_root: Path) -> None:
    localities = load_localities(landmark_root)
    manual_localities = [loc for loc in localities if loc.status == "MANUAL"]
    if not manual_localities:
        print()
        print("No MANUAL localities found. Nothing to train on.")
        return

    train_script = landmark_root / "scripts" / "train_hrnet.py"
    if not train_script.exists():
        print()
        print("[ERR] train_hrnet.py not found. Cannot start training.")
        return

    print()
    print("=== HRNet training started (action 1: MANUAL localities) ===")
    print()
    cmd = [sys.executable, str(train_script)]
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] Training failed: {exc}")
        return
    print("=== HRNet training finished ===")


def read_quality(landmark_root: Path) -> Dict[str, Any]:
    quality_path = landmark_root / "models" / "current" / "quality.json"
    if not quality_path.exists():
        return {}
    try:
        data = json.loads(quality_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def choose_autolabel_locality(localities: List[Locality]) -> Optional[Locality]:
    # Только локальности без статуса или с AUTO
    candidates: List[Locality] = [
        loc for loc in localities if loc.status in ("", "AUTO")
    ]
    print()
    print("Localities available for autolabel:")
    print()
    if not candidates:
        print("  (no localities available – only MANUAL localities found)")
        return None

    max_name = max((len(loc.locality) for loc in candidates), default=0)
    name_width = max(max_name + 2, 24)

    for idx, loc in enumerate(candidates, start=1):
        status_text = format_status(loc)
        status_block = status_text if status_text else ""
        print(
            f"[{idx}] {loc.locality.ljust(name_width)}  "
            f"{status_block:8s}  ({loc.n_images} imgs, {loc.n_labeled} csv)"
        )

    print()
    try:
        choice = input("Select locality number (or 0 to cancel): ").strip()
    except EOFError:
        return None
    if not choice:
        return None
    if not choice.isdigit():
        print("Please enter a number.")
        return None
    idx = int(choice)
    if idx <= 0:
        return None
    if idx > len(candidates):
        print("No such locality.")
        return None
    return candidates[idx - 1]


def run_autolabel(landmark_root: Path, base_localities: Optional[Path]) -> None:
    localities = load_localities(landmark_root)
    if not localities:
        print()
        print("No localities found in status/localities_status.csv.")
        return

    # Проверяем, есть ли обученная модель
    model_path = landmark_root / "models" / "current" / "hrnet_best.pth"
    if not model_path.exists():
        print()
        print("No trained model found (models/current/hrnet_best.pth is missing).")
        print("Please run action 1 (Train) first.")
        return

    loc = choose_autolabel_locality(localities)
    if not loc:
        return

    if base_localities is None:
        base_localities = read_last_base(landmark_root)
    if base_localities is None:
        print()
        print("Base localities folder is not set (cfg/last_base.txt is empty).")
        return

    png_dir = base_localities / loc.locality / "png"
    if not png_dir.exists():
        print()
        print(f"Folder not found: {png_dir}")
        return

    png_list = sorted(png_dir.glob("*.png"))
    if not png_list:
        print()
        print(f"No *.png images found in {png_dir}")
        return

    infer_script = landmark_root / "scripts" / "infer_hrnet.py"
    if not infer_script.exists():
        print()
        print("[ERR] infer_hrnet.py not found. Cannot run autolabel.")
        return

    print()
    print(f"=== Autolabel started for locality \"{loc.locality}\" ===")
    print()

    cmd = [
        sys.executable,
        str(infer_script),
        "--landmark-root",
        str(landmark_root),
        "--base-localities",
        str(base_localities),
        "--locality",
        loc.locality,
    ]
    try:
        result = subprocess.run(cmd, check=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] Autolabel failed: {exc}")
        return

    if result.returncode != 0:
        print(f"[ERR] Autolabel script returned code {result.returncode}.")
        return

    # Пересчитываем n_images / n_labeled для выбранной локальности
    png_list = sorted(png_dir.glob("*.png"))
    n_images = len(png_list)
    n_labeled = 0
    for img_path in png_list:
        csv_path = img_path.with_suffix(".csv")
        if csv_path.exists():
            n_labeled += 1

    loc.n_images = n_images
    loc.n_labeled = n_labeled

    # Читаем quality.json
    quality = read_quality(landmark_root)
    auto_quality = ""
    run_id = ""
    if quality:
        try:
            pck_percent = int(round(float(quality.get("pck_r_percent", 0))))
            auto_quality = str(pck_percent)
        except Exception:
            auto_quality = ""
        run_id = str(quality.get("run_id", "") or "")

    loc.status = "AUTO"
    loc.auto_quality = auto_quality
    loc.last_model_run = run_id
    loc.last_update = datetime.now().isoformat(timespec="seconds")

    save_localities(landmark_root, localities)

    if auto_quality:
        print(f'Autolabel done for locality "{loc.locality}".')
        print(f"Status: AUTO {auto_quality}")
    else:
        print(f'Autolabel done for locality "{loc.locality}".')
        print("Status: AUTO")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", dest="root", default=None)
    parser.add_argument("--base", dest="base", default=None)
    parser.add_argument("--base-localities", dest="base_localities", default=None)
    args = parser.parse_args()

    landmark_root = get_landmark_root(args.root)
    base_localities: Optional[Path] = None
    base_arg = args.base_localities or args.base
    if base_arg:
        base_localities = Path(base_arg)

    localities = load_localities(landmark_root)

    print("=== GM Landmarking: HRNet Trainer (v1.0) ===")
    print()
    print("1) Train / Finetune model on MANUAL localities")
    print("2) Autolabel locality with current model")
    print("3) Review AUTO locality in annotator (set MANUAL by button)")
    print("4) Info about current model / metrics")
    print("5) Model settings")
    print()
    print("0) Quit")
    print()
    choice = input("Select action: ").strip()

    # После выбора действия показываем список локальностей (для информации)
    print_localities_block(localities)

    if choice == "1":
        run_training(landmark_root)
    elif choice == "2":
        run_autolabel(landmark_root, base_localities)
    elif choice == "3":
        print()
        print("Action 3 (Review AUTO) is not implemented yet in this version.")
    elif choice == "4":
        print()
        quality = read_quality(landmark_root)
        if not quality:
            print("Model is not trained yet (models/current/quality.json not found).")
        else:
            print("Current model info:")
            print()
            print(f"Run id: {quality.get('run_id', '')}")
            print("Model: HRNet-W32 (18 keypoints)")
            n_train = quality.get("n_train_images", 0)
            n_val = quality.get("n_val_images", 0)
            train_share = quality.get("train_share", 0)
            val_share = quality.get("val_share", 0)
            try:
                train_percent = int(round(float(train_share) * 100))
            except Exception:
                train_percent = 0
            try:
                val_percent = int(round(float(val_share) * 100))
            except Exception:
                val_percent = 0
            print(f"Train images: {n_train} ({train_percent}%)")
            print(f"Val images:   {n_val} ({val_percent}%)")
            pck = quality.get("pck_r_percent", None)
            if pck is not None:
                try:
                    pck_int = int(round(float(pck)))
                except Exception:
                    pck_int = 0
                print()
                print(f"PCK@R (validation): {pck_int} %")
    elif choice == "5":
        print()
        print("Model settings are stored in config/hrnet_config.yaml.")
        print("Edit this file with a text editor to change training parameters.")
    else:
        # 0 или любая другая клавиша – просто выходим
        return


if __name__ == "__main__":
    main()
