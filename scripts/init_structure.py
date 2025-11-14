from pathlib import Path

def get_root() -> Path:
    """Return repository root (two levels up from scripts/)."""
    return Path(__file__).resolve().parent.parent


def ensure_dirs(root: Path) -> None:
    """Create base folders required by the module."""
    # All paths are relative to repo root, no absolute paths.
    (root / "status").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "models" / "current").mkdir(parents=True, exist_ok=True)
    (root / "models" / "history").mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir(parents=True, exist_ok=True)


def ensure_hrnet_config(root: Path) -> None:
    """Create default config/hrnet_config.yaml if missing."""
    cfg_dir = root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "hrnet_config.yaml"

    if cfg_path.exists():
        print(f"[INFO] Config already exists: {cfg_path}")
        return

    text = """model_type: "hrnet_w32"
input_size: 256        # 0 = do not resize (see resize_mode)
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
    cfg_path.write_text(text, encoding="utf-8")
    print(f"[INFO] Created default config: {cfg_path}")


def ensure_localities_status(root: Path) -> None:
    """Create empty status/localities_status.csv with header if missing."""
    status_dir = root / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / "localities_status.csv"

    if status_path.exists():
        print(f"[INFO] Status file already exists: {status_path}")
        return

    header = "locality,status,auto_quality,last_model_run,last_update,n_images,n_labeled\n"
    status_path.write_text(header, encoding="utf-8")
    print(f"[INFO] Created empty status file: {status_path}")


def main() -> None:
    root = get_root()
    print(f"[INFO] Landmarking root: {root}")
    ensure_dirs(root)
    ensure_hrnet_config(root)
    ensure_localities_status(root)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERR] init_structure.py failed:", e)
        raise SystemExit(1)
