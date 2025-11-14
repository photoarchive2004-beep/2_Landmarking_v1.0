from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def get_landmark_root() -> Path:
    """Landmarking root = parent of scripts/."""
    return Path(__file__).resolve().parent.parent


def load_localities_status(root: Path) -> Tuple[List[Dict[str, str]], Path]:
    """
    Load status/localities_status.csv, create empty header if missing.

    Наполнение делает rebuild_localities_status.py (сканирует папки
    локальностей и обновляет n_images, n_labeled и т.п.).
    """
    status_dir = root / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    csv_path = status_dir / "localities_status.csv"

    rows: List[Dict[str, str]] = []
    if not csv_path.exists():
        # Только заголовок, если файла ещё нет.
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "locality",
                    "status",
                    "auto_quality",
                    "last_model_run",
                    "last_update",
                    "n_images",
                    "n_labeled",
                ]
            )
        return rows, csv_path

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("locality"):
                continue
            rows.append(row)
    return rows, csv_path


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def format_localities_block(rows: List[Dict[str, str]]) -> str:
    """
    Вернуть текстовый блок со списком локальностей, аккуратно выровненный.

    Это информационный список для главного меню тренера
    (Localities (photos/<locality>/png): ...).
    """
    if not rows:
        return "Localities: (none found)\n"

    # Выравниваем имя и статус по максимальной длине
    name_width = max(len(r["locality"]) for r in rows)

    status_labels: List[str] = []
    for r in rows:
        status = (r.get("status") or "").strip()
        auto_q = (r.get("auto_quality") or "").strip()
        if status.upper() == "AUTO" and auto_q:
            status_labels.append(f"AUTO {auto_q}")
        else:
            status_labels.append(status or "")

    status_width = max((len(s) for s in status_labels), default=0)
    if status_width < 6:
        status_width = 6

    lines: List[str] = []
    lines.append("Localities (photos/<locality>/png):\n")

    for idx, (r, s_label) in enumerate(zip(rows, status_labels), start=1):
        locality = r["locality"]
        n_img = _safe_int((r.get("n_images") or "").strip() or "0")
        n_lab = _safe_int((r.get("n_labeled") or "").strip() or "0")

        # На всякий случай не даём проценту перевалить за 100
        if n_lab > n_img:
            n_lab_eff = n_img
        else:
            n_lab_eff = n_lab

        if n_img > 0:
            percent = int(round(100.0 * n_lab_eff / float(n_img)))
        else:
            percent = 0

        name_part = locality.ljust(name_width)
        status_part = s_label.ljust(status_width)
        progress_part = f"[{n_lab}/{n_img}] {percent:3d}%"

        # Между колонками ровно по 3 пробела
        line = f"[{idx:2d}] {name_part}   {status_part}   {progress_part}"
        lines.append(line)

    return "\n".join(lines) + "\n"


def _parse_simple_yaml(path: Path) -> Dict[str, object]:
    """
    Very small YAML reader for simple "key: value" lines.
    No nesting, no lists, comments (# ...) allowed.
    """
    if not path.exists():
        return {}

    result: Dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue

        key, value_raw = line.split(":", 1)
        key = key.strip()
        value_str = value_raw.strip()

        # Strip quotes
        if (
            (value_str.startswith('"') and value_str.endswith('"'))
            or (value_str.startswith("'") and value_str.endswith("'"))
        ):
            value_str = value_str[1:-1]

        # Convert bool / int / float if possible
        lower = value_str.lower()
        if lower == "true":
            value: object = True
        elif lower == "false":
            value = False
        else:
            try:
                if "." in value_str:
                    value = float(value_str)
                else:
                    value = int(value_str)
            except Exception:
                value = value_str

        result[key] = value

    return result


def show_model_settings(root: Path) -> None:
    """
    Пункт меню 5: показать настройки из config/hrnet_config.yaml
    с простыми английскими пояснениями (как в ТЗ).
    """
    cfg_dir = root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "hrnet_config.yaml"

    # Если файла нет – создаём дефолтный, близкий к примеру из ТЗ
    if not cfg_path.exists():
        default_text = """# HRNet training config for GM Landmarking
model_type: "hrnet_w32"
input_size: 256  # 0 = do not resize (see resize_mode)
resize_mode: "resize"  # "resize" or "original"
keep_aspect_ratio: true
batch_size: 8
learning_rate: 0.0005
max_epochs: 100
train_val_split: 0.9
flip_augmentation: true
rotation_augmentation_deg: 15
scale_augmentation: 0.3
weight_decay: 0.0001
"""
        cfg_path.write_text(default_text, encoding="utf-8")

    conf = _parse_simple_yaml(cfg_path)

    print("=== Model settings (config/hrnet_config.yaml) ===\n")

    def show_param(name: str, explanation_lines):
        val = conf.get(name, "")
        print(f"{name} = {val!r}")
        for line in explanation_lines:
            print(f"  - {line}")
        print()

    show_param(
        "model_type",
        [
            "HRNet backbone type.",
            'For example: "hrnet_w32" is a good default.',
        ],
    )
    show_param(
        "input_size",
        [
            "Target input size for the network in pixels.",
            "If resize_mode = 'resize': images are scaled to this size.",
            "If input_size = 0 and resize_mode = 'original': use original image size.",
        ],
    )
    show_param(
        "resize_mode",
        [
            "How to change image size before training.",
            "'resize' - scale images to input_size.",
            "'original' - keep original resolution (only safe padding/downscale if needed).",
        ],
    )
    show_param(
        "keep_aspect_ratio",
        [
            "If True: keep fish shape, no stretching by one axis.",
        ],
    )
    show_param(
        "batch_size",
        [
            "How many images are processed in one training step.",
        ],
    )
    show_param(
        "learning_rate",
        [
            "How fast the model learns.",
            "Too high -> unstable, too low -> very slow.",
        ],
    )
    show_param(
        "max_epochs",
        [
            "Maximum number of passes through the training data.",
        ],
    )
    show_param(
        "train_val_split",
        [
            "Part of data used for training.",
            "0.9 = 90% train, 10% validation.",
        ],
    )
    show_param(
        "flip_augmentation",
        [
            "Random horizontal flip of images during training.",
        ],
    )
    show_param(
        "rotation_augmentation_deg",
        [
            "Maximum random rotation in degrees during training.",
        ],
    )
    show_param(
        "scale_augmentation",
        [
            "Random zoom in/out (for example 0.3 = up to 30%).",
        ],
    )
    show_param(
        "weight_decay",
        [
            "Regularization to reduce overfitting.",
        ],
    )

    print("To change these values:")
    print('1) Open file "config/hrnet_config.yaml" with a text editor (for example Notepad).')
    print("2) Change numbers or true/false values.")
    print("3) Save the file.")
    print("New training runs will automatically use the new settings.")
    print("Do not change the parameter names, only their values.")
    print()


def show_current_model_info(root: Path) -> None:
    """
    Пункт меню 4: информация о текущей модели по models/current/quality.json.

    Если файла нет – выводим понятное сообщение.
    Формат вывода соответствует примеру из ТЗ.
    """
    q_path = root / "models" / "current" / "quality.json"
    print("Current model info:\n")

    if not q_path.exists():
        print("Model is not trained yet (models/current/quality.json not found).")
        print()
        return

    try:
        data = json.loads(q_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("[ERR] Cannot read models/current/quality.json:")
        print(f"      {exc}")
        print()
        return

    run_id = data.get("run_id", "?")
    # В quality.json модель может не быть прописана – подставляем дефолт.
    model = data.get("model_type", "HRNet-W32 (18 keypoints)")

    n_train = int(data.get("n_train_images", 0) or 0)
    n_val = int(data.get("n_val_images", 0) or 0)

    train_share = float(data.get("train_share", 0.0) or 0.0)
    val_share = float(data.get("val_share", 0.0) or 0.0)

    # PCK в процентах – стараемся сначала взять pck_r_percent.
    pck_percent = data.get("pck_r_percent")
    if pck_percent is None:
        # Если есть только pck_r в виде доли 0.xx
        pck_raw = float(data.get("pck_r", 0.0) or 0.0)
        pck_percent = int(round(100.0 * pck_raw))
    else:
        pck_percent = int(pck_percent)

    n_manual = int(data.get("n_manual_localities", 0) or 0)

    print(f"Run id: {run_id}")
    print(f"Model: {model}")
    print()
    print(f"Train images: {n_train} ({int(round(train_share * 100))}%)")
    print(f"Val images:   {n_val} ({int(round(val_share * 100))}%)")
    print()
    print(f"PCK@R (validation): {pck_percent} %")
    print()
    print(f"Used MANUAL localities: {n_manual}")
    print()


def run_train_manual(root: Path) -> None:
    """
    Пункт меню 1: запустить scripts/train_hrnet.py.

    На этом шаге train_hrnet.py пока только готовит датасет и выводит
    сводку (как сейчас реализовано). Реальное обучение HRNet/MMPose
    будет добавлено отдельным шагом, чтобы не усложнять сразу всё.
    """
    script = root / "scripts" / "train_hrnet.py"
    if not script.exists():
        print("[ERR] scripts/train_hrnet.py not found.")
        print("Please check repository contents.")
        print()
        return

    try:
        rc = subprocess.call([sys.executable, str(script)])
        if rc != 0:
            print(f"[WARN] train_hrnet.py exited with code {rc}.")
            print()
    except Exception as exc:
        print("[ERR] Cannot start train_hrnet.py:")
        print(f"      {exc}")
        print()


def main() -> None:
    root = get_landmark_root()
    rows, _ = load_localities_status(root)

    print("=== GM Landmarking: HRNet Trainer (v1.0) ===\n")
    print("1) Train / Finetune model on MANUAL localities")
    print("2) Autolabel locality with current model")
    print("3) Review AUTO locality in annotator (set MANUAL by button)")
    print("4) Info about current model / metrics")
    print("5) Model settings\n")
    print("0) Quit\n")

    choice = input("Select action: ").strip()
    print()  # spacer

    # Информационный список локальностей под меню
    print(format_localities_block(rows))

    if choice == "0" or choice.upper() == "Q":
        return
    elif choice == "1":
        run_train_manual(root)
    elif choice == "4":
        show_current_model_info(root)
        input("Press Enter to exit...")
    elif choice == "5":
        show_model_settings(root)
        input("Press Enter to exit...")
    else:
        print("This action is not implemented yet.")
        print("Use option 5 to view settings.")
        input("Press Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
