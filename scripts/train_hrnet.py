from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def get_landmark_root() -> Path:
    """
    Вернуть папку tools/2_Landmarking_v1.0, считая, что этот файл лежит в scripts/.
    """
    here = Path(__file__).resolve()
    return here.parent.parent


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def load_localities_status(root: Path) -> Tuple[List[Dict[str, str]], Path]:
    status_dir = root / "status"
    csv_path = status_dir / "localities_status.csv"
    rows: List[Dict[str, str]] = []
    if not csv_path.exists():
        return rows, csv_path
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("locality"):
                continue
            rows.append(row)
    return rows, csv_path


def get_base_localities(root: Path) -> Path:
    """
    Путь к базе локальностей берём строго из cfg/last_base.txt,
    как прописано в ТЗ.
    """
    cfg_dir = root / "cfg"
    last_base = cfg_dir / "last_base.txt"
    if not last_base.exists():
        raise RuntimeError("cfg/last_base.txt not found.")
    text = last_base.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("cfg/last_base.txt is empty.")
    base = Path(text)
    return base


@dataclass
class Sample:
    image: Path
    csv: Path
    locality: str


def collect_manual_samples(root: Path, base_localities: Path) -> Tuple[List[Sample], int]:
    """
    Собираем все пары PNG+CSV из локальностей со статусом MANUAL.
    """
    rows, _ = load_localities_status(root)
    manual_rows = [
        r
        for r in rows
        if (r.get("status") or "").strip().upper() == "MANUAL"
    ]
    samples: List[Sample] = []

    for row in manual_rows:
        loc = (row.get("locality") or "").strip()
        if not loc:
            continue
        png_dir = base_localities / loc / "png"
        if not png_dir.is_dir():
            continue
        for img in sorted(png_dir.glob("*.png")):
            csv_path = img.with_suffix(".csv")
            if not csv_path.exists():
                continue
            samples.append(Sample(image=img, csv=csv_path, locality=loc))

    return samples, len(manual_rows)


def get_train_val_split(root: Path) -> float:
    """
    Берём train_val_split из config/model_settings.yaml, если возможно.
    Если конфиг или PyYAML недоступны — используем 0.9 (90% train / 10% val),
    как допускает ТЗ.
    """
    default_value = 0.9
    settings_path = root / "config" / "model_settings.yaml"
    if not settings_path.exists():
        return default_value

    try:
        import yaml  # type: ignore
    except Exception:
        return default_value

    try:
        text = settings_path.read_text(encoding="utf-8")
        cfg = yaml.safe_load(text) or {}
    except Exception:
        return default_value

    value = cfg.get("train_val_split")
    try:
        v = float(value)
    except Exception:
        return default_value

    if v <= 0.0 or v >= 1.0:
        return default_value
    return v


def split_dataset(n_items: int, train_share: float) -> Tuple[int, int]:
    """
    Делим датасет на train/val с защитой от крайних случаев.
    """
    if n_items <= 0:
        return 0, 0
    n_train = int(round(n_items * train_share))
    if n_train <= 0:
        n_train = 1
    if n_train >= n_items:
        n_train = n_items - 1 if n_items > 1 else n_items
    n_val = n_items - n_train
    return n_train, n_val


def ensure_dirs(root: Path, run_id: str) -> Tuple[Path, Path]:
    """
    Готовим папки:
      models/history/<run_id>/
      models/current/
    согласно ТЗ.
    """
    models_dir = root / "models"
    history_dir = models_dir / "history" / run_id
    current_dir = models_dir / "current"
    history_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)
    return history_dir, current_dir


def write_train_config(path: Path, run_id: str, n_manual: int, n_train: int, n_val: int, split: float) -> None:
    """
    Минимальный YAML-подобный train_config.yaml для истории обучения.
    """
    lines = [
        f"run_id: {run_id}",
        f"n_manual_localities: {n_manual}",
        f"n_train_images: {n_train}",
        f"n_val_images: {n_val}",
        f"train_val_split: {split:.3f}",
        "note: placeholder training (no neural network yet)",
    ]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    root = get_landmark_root()

    try:
        base_localities = get_base_localities(root)
    except Exception as exc:
        print("[ERR] Base localities path is not configured.")
        print(f"      {exc}")
        sys.exit(1)

    samples, n_manual = collect_manual_samples(root, base_localities)
    n_total = len(samples)

    # Если нет ни одной MANUAL локальности или ни одной пары PNG+CSV
    if n_manual == 0 or n_total == 0:
        print("Training finished.")
        print()
        print("Used MANUAL localities: 0")
        print("Train images: 0 (0%)")
        print("Val images:   0 (0%)")
        print()
        print("PCK@R (validation): 0 %")
        print()
        print("Model saved as: models/current/hrnet_best.pth")
        print("Run id: none")
        return

    split = get_train_val_split(root)
    n_train, n_val = split_dataset(n_total, split)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir, current_dir = ensure_dirs(root, run_id)

    # Заглушка модели: реальное обучение HRNet/MMPose будет добавлено позже.
    model_path = history_dir / "hrnet_best.pth"
    model_path.write_bytes(b"")

    if n_total > 0:
        train_share = float(n_train) / float(n_total)
        val_share = float(n_val) / float(n_total)
    else:
        train_share = 0.0
        val_share = 0.0

    # Пока без реального обучения: pck_r == 0.0
    pck_r = 0.0
    pck_r_percent = int(round(pck_r * 100.0))

    metrics = {
        "run_id": run_id,
        "pck_r": pck_r,
        "pck_r_percent": pck_r_percent,
        "n_train_images": n_train,
        "n_val_images": n_val,
        "train_share": train_share,
        "val_share": val_share,
        "n_manual_localities": n_manual,
    }

    metrics_path = history_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    train_log_path = history_dir / "train_log.txt"
    with train_log_path.open("w", encoding="utf-8") as f:
        f.write(f"Run id: {run_id}\n")
        f.write(f"Base localities: {base_localities}\n")
        f.write(f"Manual localities: {n_manual}\n")
        f.write(f"Total samples (PNG+CSV): {n_total}\n")
        f.write(f"Train images: {n_train}\n")
        f.write(f"Val images: {n_val}\n")
        f.write("NOTE: placeholder training, neural network is not trained yet.\n")

    train_config_path = history_dir / "train_config.yaml"
    write_train_config(train_config_path, run_id, n_manual, n_train, n_val, split)

    # Обновляем current-модель и quality.json
    current_model_path = current_dir / "hrnet_best.pth"
    try:
        data = model_path.read_bytes()
        current_model_path.write_bytes(data)
    except Exception:
        current_model_path.write_bytes(b"")

    quality_path = current_dir / "quality.json"
    with quality_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Вывод в консоль в формате, близком к ТЗ
    train_percent = int(round(train_share * 100.0)) if n_total > 0 else 0
    val_percent = int(round(val_share * 100.0)) if n_total > 0 else 0

    print("Training finished.")
    print()
    print(f"Used MANUAL localities: {n_manual}")
    print(f"Train images: {n_train} ({train_percent}%)")
    print(f"Val images:   {n_val}  ({val_percent}%)")
    print()
    print(f"PCK@R (validation): {pck_r_percent} %")
    print()
    print("Model saved as: models/current/hrnet_best.pth")
    print()
    print(f"Run id: {run_id}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
