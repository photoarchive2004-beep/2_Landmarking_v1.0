"""
HRNet training script for GM Landmarking.

Implements ТЗ_1.0, действие 1 (обучение по MANUAL локальностям),
но без реального обучения нейросети: создаётся структура файлов,
датасет и заглушечные метрики, чтобы конвейер не падал.
"""

import csv
import json
import sys
import random
from pathlib import Path
from datetime import datetime

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # will be checked in main()


LANDMARK_ROOT = Path(__file__).resolve().parent.parent
STATUS_CSV = LANDMARK_ROOT / "status" / "localities_status.csv"
CFG_LAST_BASE = LANDMARK_ROOT / "cfg" / "last_base.txt"
CONFIG_YAML = LANDMARK_ROOT / "config" / "hrnet_config.yaml"
MODELS_CURRENT = LANDMARK_ROOT / "models" / "current"
MODELS_HISTORY = LANDMARK_ROOT / "models" / "history"
DATASETS_ROOT = LANDMARK_ROOT / "datasets"
LOGS_ROOT = LANDMARK_ROOT / "logs"
TRAIN_LOG_LAST = LOGS_ROOT / "train_hrnet_last.log"


def ensure_dirs() -> None:
    """Make sure standard folders exist (defensive, init_structure.py should already do this)."""
    for p in (MODELS_CURRENT, MODELS_HISTORY, DATASETS_ROOT, LOGS_ROOT):
        p.mkdir(parents=True, exist_ok=True)


def log_line(message: str, *, also_print: bool = True) -> None:
    """Append message to logs/train_hrnet_last.log and optionally print to console."""
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    with TRAIN_LOG_LAST.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    if also_print:
        print(message)


def load_config():
    """Load hrnet_config.yaml, return dict with defaults if something is missing."""
    if yaml is None:
        log_line("ERROR: PyYAML is not installed. Please install 'pyyaml' in the landmarking environment.")
        sys.exit(1)

    if not CONFIG_YAML.exists():
        log_line("ERROR: config/hrnet_config.yaml not found. Please run 2_TRAIN-INFER_HRNet.bat again.")
        sys.exit(1)

    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Default values from ТЗ_1.0
    cfg_defaults = {
        "model_type": "hrnet_w32",
        "input_size": 256,
        "resize_mode": "resize",
        "keep_aspect_ratio": True,
        "batch_size": 8,
        "learning_rate": 0.0005,
        "max_epochs": 100,
        "train_val_split": 0.9,
        "flip_augmentation": True,
        "rotation_augmentation_deg": 15,
        "scale_augmentation": 0.3,
        "weight_decay": 0.0001,
    }

    for k, v in cfg_defaults.items():
        cfg.setdefault(k, v)

    return cfg


def read_base_localities() -> Path:
    """Read base localities path from cfg/last_base.txt."""
    if not CFG_LAST_BASE.exists():
        log_line("ERROR: cfg/last_base.txt not found. Please select base folder in 2_TRAIN-INFER_HRNet.bat.")
        sys.exit(1)
    text = CFG_LAST_BASE.read_text(encoding="utf-8").strip()
    if not text:
        log_line("ERROR: cfg/last_base.txt is empty. Please re-run 2_TRAIN-INFER_HRNet.bat and choose folder with localities.")
        sys.exit(1)
    base = Path(text)
    if not base.exists():
        log_line(f"ERROR: Base localities folder does not exist: {base}")
        sys.exit(1)
    return base


def read_localities_status():
    """Read status/localities_status.csv and return list of dict rows."""
    if not STATUS_CSV.exists():
        log_line("ERROR: status/localities_status.csv not found. Please run annotator or trainer menu again.")
        sys.exit(1)

    rows = []
    with STATUS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize keys just in case
            row = { (k or "").strip(): (v or "").strip() for k, v in row.items() }
            if not row.get("locality"):
                continue
            rows.append(row)
    return rows


def collect_manual_pairs(base_localities: Path, rows):
    """
    From MANUAL localities collect (img_path, csv_path, locality) pairs.

    Use only images which have corresponding *.csv.
    """
    manual_localities = [r for r in rows if r.get("status") == "MANUAL"]
    if not manual_localities:
        log_line("No MANUAL localities found in localities_status.csv. Nothing to train.")
        return [], 0

    pairs = []
    skipped_localities = []
    for r in manual_localities:
        loc_name = r["locality"]
        png_dir = base_localities / loc_name / "png"
        if not png_dir.exists():
            skipped_localities.append(loc_name)
            log_line(f"WARNING: Locality folder not found on disk, skipping: {png_dir}", also_print=False)
            continue

        # Collect *.png (и при необходимости *.jpg)
        images = list(png_dir.glob("*.png")) + list(png_dir.glob("*.jpg"))
        used_here = 0
        for img_path in images:
            csv_path = img_path.with_suffix(".csv")
            if not csv_path.exists():
                continue
            pairs.append((img_path, csv_path, loc_name))
            used_here += 1

        log_line(f"Locality '{loc_name}': {used_here} labeled images used for training.", also_print=False)

    if skipped_localities:
        log_line("Some MANUAL localities were skipped because folders were not found. See log for details.", also_print=True)

    n_manual = len(manual_localities) - len(skipped_localities)
    if not pairs:
        log_line("No labeled images found for MANUAL localities. Nothing to train.")
        return [], n_manual

    return pairs, n_manual


def split_dataset(pairs, train_val_split: float, run_id: str):
    """Shuffle pairs and split into train/val according to train_val_split."""
    total = len(pairs)
    if total == 0:
        return [], [], 0, 0, 0.0, 0.0

    random.shuffle(pairs)
    n_train = max(1, int(round(total * float(train_val_split))))
    if n_train >= total and total > 1:
        n_train = total - 1
    n_val = total - n_train

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    train_share = n_train / total
    val_share = n_val / total

    # Save dataset description to datasets/
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    train_list_path = DATASETS_ROOT / f"hrnet_train_{run_id}.txt"
    val_list_path = DATASETS_ROOT / f"hrnet_val_{run_id}.txt"

    def write_list(path, items):
        with path.open("w", encoding="utf-8") as f:
            for img_path, csv_path, locality in items:
                f.write(f"{img_path};{csv_path};{locality}\n")

    write_list(train_list_path, train_pairs)
    write_list(val_list_path, val_pairs)

    log_line(f"Dataset saved: {train_list_path.name} ({n_train} images), {val_list_path.name} ({n_val} images).", also_print=False)

    return train_pairs, val_pairs, n_train, n_val, train_share, val_share


def write_history_and_current(run_id: str,
                              cfg: dict,
                              n_manual_localities: int,
                              n_train_images: int,
                              n_val_images: int,
                              train_share: float,
                              val_share: float):
    """Create history folder, dummy model, metrics and current model info."""
    MODELS_HISTORY.mkdir(parents=True, exist_ok=True)
    MODELS_CURRENT.mkdir(parents=True, exist_ok=True)

    history_dir = MODELS_HISTORY / run_id
    history_dir.mkdir(parents=True, exist_ok=True)

    # Dummy model file
    hrnet_best_history = history_dir / "hrnet_best.pth"
    with hrnet_best_history.open("wb") as f:
        f.write(b"This is a placeholder HRNet model file. Replace with real trained weights.\n")

    # Metrics (placeholder PCK@R = 0.0)
    pck_r = 0.0
    pck_r_percent = int(round(pck_r * 100))

    metrics = {
        "run_id": run_id,
        "pck_r": pck_r,
        "pck_r_percent": pck_r_percent,
        "n_train_images": n_train_images,
        "n_val_images": n_val_images,
        "train_share": round(train_share, 4),
        "val_share": round(val_share, 4),
        "n_manual_localities": n_manual_localities,
        "model_type": cfg.get("model_type", "hrnet_w32"),
    }
    metrics_path = history_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save training config snapshot
    train_config_path = history_dir / "train_config.yaml"
    try:
        if yaml is not None:
            with train_config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        else:  # pragma: no cover
            with train_config_path.open("w", encoding="utf-8") as f:
                f.write("# hrnet_config snapshot not available (PyYAML missing)\n")
    except Exception as exc:  # pragma: no cover
        log_line(f"WARNING: Failed to write train_config.yaml: {exc}", also_print=False)

    # Training log inside history
    train_log_path = history_dir / "train_log.txt"
    with train_log_path.open("w", encoding="utf-8") as f:
        f.write(f"HRNet training run (placeholder implementation)\n")
        f.write(f"Run id: {run_id}\n")
        f.write(f"Model type: {cfg.get('model_type', 'hrnet_w32')}\n")
        f.write(f"Train images: {n_train_images}\n")
        f.write(f"Val images: {n_val_images}\n")
        f.write(f"Train share: {train_share:.3f}\n")
        f.write(f"Val share: {val_share:.3f}\n")
        f.write(f"MANUAL localities used: {n_manual_localities}\n")
        f.write("\n")
        f.write("NOTE: This is a placeholder. Real HRNet training is not implemented yet.\n")

    # Copy model to models/current/hrnet_best.pth
    hrnet_best_current = MODELS_CURRENT / "hrnet_best.pth"
    hrnet_best_current.write_bytes(hrnet_best_history.read_bytes())

    quality = {
        "run_id": run_id,
        "pck_r": pck_r,
        "pck_r_percent": pck_r_percent,
        "n_train_images": n_train_images,
        "n_val_images": n_val_images,
        "train_share": round(train_share, 4),
        "val_share": round(val_share, 4),
        "n_manual_localities": n_manual_localities,
    }
    quality_path = MODELS_CURRENT / "quality.json"
    with quality_path.open("w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    return pck_r_percent


def main():
    ensure_dirs()
    log_line("=== HRNet training started (action 1: MANUAL localities) ===")

    cfg = load_config()
    train_val_split = float(cfg.get("train_val_split", 0.9))

    base_localities = read_base_localities()
    log_line(f"Base localities folder: {base_localities}", also_print=False)

    rows = read_localities_status()
    pairs, n_manual_localities = collect_manual_pairs(base_localities, rows)
    if not pairs:
        # Message already logged by collect_manual_pairs
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_pairs, val_pairs, n_train, n_val, train_share, val_share = split_dataset(
        pairs, train_val_split, run_id
    )
    total = len(pairs)

    # Summary to log
    log_line(f"Total labeled images: {total}", also_print=False)
    log_line(f"Train images: {n_train} ({train_share*100:.1f}%)", also_print=False)
    log_line(f"Val images: {n_val} ({val_share*100:.1f}%)", also_print=False)

    # Here should be real HRNet/MMPose training.
    # Placeholder: just write files with dummy metrics so that pipeline is functional.
    pck_r_percent = write_history_and_current(
        run_id,
        cfg,
        n_manual_localities=n_manual_localities,
        n_train_images=n_train,
        n_val_images=n_val,
        train_share=train_share,
        val_share=val_share,
    )

    # Console output in strict format from ТЗ_1.0
    print()
    print("Training finished.")
    print()
    print(f"Used MANUAL localities: {n_manual_localities}")
    print(f"Train images: {n_train} ({int(round(train_share*100))}%)")
    print(f"Val images:   {n_val}  ({int(round(val_share*100))}%)")
    print()
    print(f"PCK@R (validation): {pck_r_percent} %")
    print()
    print("Model saved as: models/current/hrnet_best.pth")
    print(f"Run id: {run_id}")

    log_line("=== HRNet training finished ===")


if __name__ == "__main__":  # pragma: no cover
    main()
