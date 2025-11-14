from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def get_landmark_root() -> Path:
    """Return tools/2_Landmarking_v1.0 folder (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def get_base_localities(root: Path) -> Path:
    """
    Read base_localities from cfg/last_base.txt.

    This file is created by 1_ANNOTATOR.bat / 2_TRAIN-INFER_HRNet.bat,
    as described in ТЗ_1.0.
    """
    cfg_dir = root / "cfg"
    last_base = cfg_dir / "last_base.txt"
    if not last_base.exists():
        raise RuntimeError("cfg/last_base.txt not found.")
    text = last_base.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("cfg/last_base.txt is empty.")
    return Path(text)


def load_localities_status(root: Path) -> Tuple[List[Dict[str, str]], Path]:
    """Load status/localities_status.csv into a list of dicts."""
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


@dataclass
class Sample:
    image: Path
    csv: Path
    locality: str


def collect_manual_samples(root: Path, base_localities: Path) -> Tuple[List[Sample], int]:
    """
    Collect all (png, csv) pairs for localities with status == 'MANUAL'.
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


def read_hrnet_config(root: Path) -> Tuple[float, str]:
    """
    Read train_val_split and model_type from config/hrnet_config.yaml.

    Very small parser: only simple 'key: value' lines are supported.
    """
    cfg_path = root / "config" / "hrnet_config.yaml"
    default_split = 0.9
    default_model = "HRNet-W32 (18 keypoints)"

    if not cfg_path.exists():
        return default_split, default_model

    train_split = default_split
    model_name = default_model

    for raw in cfg_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value_raw = line.split(":", 1)
        key = key.strip()
        value_str = value_raw.strip()

        # strip quotes
        if (
            (value_str.startswith('"') and value_str.endswith('"'))
            or (value_str.startswith("'") and value_str.endswith("'"))
        ):
            value_str = value_str[1:-1]

        if key == "train_val_split":
            try:
                v = float(value_str)
                if 0.0 < v < 1.0:
                    train_split = v
            except Exception:
                pass
        elif key == "model_type":
            if value_str:
                model_name = value_str

    return train_split, model_name


def split_dataset(n_items: int, train_share: float) -> Tuple[int, int]:
    """
    Compute train/val sizes with basic protection against edge cases.
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


def write_dataset_lists(
    datasets_root: Path,
    run_id: str,
    train_samples: List[Sample],
    val_samples: List[Sample],
) -> None:
    """
    Save simple CSV lists for train/val inside datasets/<run_id>/.

    Format: locality,image,csv
    where image/csv are file names (no absolute paths).
    """
    run_dir = datasets_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, samples: List[Sample]) -> None:
        path = run_dir / f"{name}_list.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["locality", "image", "csv"])
            for s in samples:
                writer.writerow([s.locality, s.image.name, s.csv.name])

    _write("train", train_samples)
    _write("val", val_samples)


def ensure_model_dirs(root: Path, run_id: str) -> Tuple[Path, Path]:
    """
    Create models/history/<run_id>/ and models/current/ folders.
    """
    models_dir = root / "models"
    history_dir = models_dir / "history" / run_id
    current_dir = models_dir / "current"
    history_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)
    return history_dir, current_dir


def write_train_config(
    path: Path,
    run_id: str,
    n_manual: int,
    n_train: int,
    n_val: int,
    split: float,
    model_name: str,
) -> None:
    """
    Very small YAML-like train_config.yaml for history.
    """
    lines = [
        f"run_id: {run_id}",
        f"model_type: {model_name}",
        f"n_manual_localities: {n_manual}",
        f"n_train_images: {n_train}",
        f"n_val_images: {n_val}",
        f"train_val_split: {split:.3f}",
        "note: placeholder training (neural network is not implemented yet)",
    ]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    root = get_landmark_root()
    try:
        base_localities = get_base_localities(root)
    except Exception as exc:
        print("[ERR] Base localities path is not configured.")
        print(f"      {exc}")
        return 1

    samples, n_manual = collect_manual_samples(root, base_localities)
    n_total = len(samples)

    if n_manual == 0 or n_total == 0:
        print("No MANUAL localities with PNG+CSV pairs found.")
        print("Nothing to train.")
        return 0

    train_split, model_name = read_hrnet_config(root)

    # deterministic split: sort by (locality, image name)
    samples_sorted = sorted(samples, key=lambda s: (s.locality, s.image.name))
    n_train, n_val = split_dataset(len(samples_sorted), train_split)
    train_samples = samples_sorted[:n_train]
    val_samples = samples_sorted[n_train:]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    history_dir, current_dir = ensure_model_dirs(root, run_id)

    datasets_root = root / "datasets"
    write_dataset_lists(datasets_root, run_id, train_samples, val_samples)

    # placeholder model file
    model_path = history_dir / "hrnet_best.pth"
    model_path.write_bytes(b"")

    if n_total > 0:
        train_share = float(n_train) / float(n_total)
        val_share = float(n_val) / float(n_total)
    else:
        train_share = 0.0
        val_share = 0.0

    pck_r = 0.0
    pck_r_percent = int(round(pck_r * 100.0))

    metrics = {
        "run_id": run_id,
        "model_type": model_name,
        "pck_r": pck_r,
        "pck_r_percent": pck_r_percent,
        "n_train_images": n_train,
        "n_val_images": n_val,
        "train_share": train_share,
        "val_share": val_share,
        "n_manual_localities": n_manual,
    }

    metrics_path = history_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    train_log_path = history_dir / "train_log.txt"
    with train_log_path.open("w", encoding="utf-8") as f:
        f.write(f"Run id: {run_id}\n")
        f.write(f"Base localities: {base_localities}\n")
        f.write(f"Manual localities: {n_manual}\n")
        f.write(f"Total samples (PNG+CSV): {n_total}\n")
        f.write(f"Train images: {n_train}\n")
        f.write(f"Val images: {n_val}\n")
        f.write("NOTE: placeholder training, neural network is not implemented yet.\n")

    train_config_path = history_dir / "train_config.yaml"
    write_train_config(
        train_config_path,
        run_id,
        n_manual,
        n_train,
        n_val,
        train_split,
        model_name,
    )

    current_model_path = current_dir / "hrnet_best.pth"
    try:
        data = model_path.read_bytes()
        current_model_path.write_bytes(data)
    except Exception:
        current_model_path.write_bytes(b"")

    quality_path = current_dir / "quality.json"
    quality_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Detailed summary is printed by trainer_menu.run_train_manual().
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(1)

from pathlib import Path
import json


def _gm_print_training_summary_from_quality():
    """
    Print training summary after HRNet training, as described in ТЗ_1.0.
    Uses models/current/quality.json.
    """
    root = Path(__file__).resolve().parent.parent
    q_path = root / "models" / "current" / "quality.json"

    if not q_path.exists():
        print("[WARN] models/current/quality.json not found, cannot print training summary.")
        return

    try:
        data = json.loads(q_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("[ERR] Cannot read models/current/quality.json:")
        print(f"      {exc}")
        return

    run_id = data.get("run_id", "?")
    n_train = int(data.get("n_train_images", 0) or 0)
    n_val = int(data.get("n_val_images", 0) or 0)
    train_share = float(data.get("train_share", 0.0) or 0.0)
    val_share = float(data.get("val_share", 0.0) or 0.0)
    pck_percent = int(data.get("pck_r_percent", 0) or 0)
    n_loc = int(data.get("n_manual_localities", 0) or 0)

    train_pct = int(round(train_share * 100.0)) if train_share > 0.0 else 0
    val_pct = int(round(val_share * 100.0)) if val_share > 0.0 else 0

    print()
    print("Training finished.")
    print()
    print(f"Used MANUAL localities: {n_loc}")
    print(f"Train images: {n_train} ({train_pct}%)")
    print(f"Val images:   {n_val:4d} ({val_pct}%)")
    print()
    print(f"PCK@R (validation): {pck_percent} %")
    print()
    print("Model saved as: models/current/hrnet_best.pth")
    print(f"Run id: {run_id}")


if __name__ == "__main__":
    try:
        _gm_print_training_summary_from_quality()
    except Exception as exc:
        print("[WARN] Failed to print training summary:")
        print(f"       {exc}")
