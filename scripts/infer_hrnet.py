from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class HRNetConfig:
    # Те же поля, что и в train_hrnet.py
    model_type: str = "hrnet_w32"
    input_size: int = 256
    resize_mode: str = "resize"
    keep_aspect_ratio: bool = True
    batch_size: int = 8
    learning_rate: float = 5e-4
    max_epochs: int = 100
    train_val_split: float = 0.9
    flip_augmentation: bool = True
    rotation_augmentation_deg: float = 15.0
    scale_augmentation: float = 0.3
    weight_decay: float = 1e-4


def get_landmark_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_yaml_config(cfg_path: Path) -> HRNetConfig:
    cfg = HRNetConfig()
    if yaml is None or not cfg_path.is_file():
        return cfg
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for field in cfg.__dataclass_fields__.keys():  # type: ignore[attr-defined]
        if field in data:
            setattr(cfg, field, data[field])
    return cfg


def read_lm_number(root: Path) -> int:
    lm_file = root / "LM_number.txt"
    try:
        txt = lm_file.read_text(encoding="utf-8").strip()
        return int(txt)
    except Exception:
        return 0


def read_last_base(root: Path) -> Optional[Path]:
    cfg_file = root / "cfg" / "last_base.txt"
    if not cfg_file.is_file():
        return None
    txt = cfg_file.read_text(encoding="utf-8").strip()
    if not txt:
        return None
    p = Path(txt)
    return p if p.is_dir() else None


def _resize_and_pad(img: Image.Image, cfg: HRNetConfig) -> Tuple[Image.Image, float, float, float]:
    """
    Та же логика ресайза, что и в train_hrnet.py: даунскейл + паддинг.
    Возвращаем:
    - изображение,
    - scale (во сколько раз уменьшили),
    - offset_x, offset_y (сколько пикселей добавили слева/сверху).
    """
    input_size = int(cfg.input_size)
    if input_size <= 0:
        w, h = img.size
        return img, 1.0, 0.0, 0.0

    w, h = img.size
    if max(w, h) == 0:
        return img, 1.0, 0.0, 0.0

    scale = min(1.0, float(input_size) / float(max(w, h)))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_resized = img
    if scale != 1.0:
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
    offset_x = (input_size - new_w) // 2
    offset_y = (input_size - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))
    return canvas, scale, float(offset_x), float(offset_y)


class SimpleHRNet(nn.Module):
    """
    Точно такой же бэкбон, как в train_hrnet.py (SimpleHRNet),
    чтобы успешно загрузить hrnet_best.pth.
    """

    def __init__(self, num_keypoints: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        blocks = []
        in_channels = 64
        for _ in range(3):
            block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Conv2d(64, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            residual = x
            out = block(x)
            x = self.relu(out + residual)
        heatmaps = self.head(x)
        return heatmaps


def heatmaps_to_keypoints(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Из теплокарт получаем координаты максимумов.
    """
    B, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(B, K, -1)
    idx = heatmaps_reshaped.argmax(dim=2)  # (B, K)
    ys = (idx // W).float()
    xs = (idx % W).float()
    kps = torch.stack([xs, ys], dim=2)
    return kps


def infer_for_locality(
    root: Path,
    base_dir: Path,
    locality: str,
    cfg: HRNetConfig,
    num_keypoints: int,
    model_path: Path,
    log_fn,
) -> None:
    """
    Автолейблинг одной локальности: пробегаем все PNG и сохраняем CSV с (x,y).
    """
    loc_dir = base_dir / locality / "png"
    if not loc_dir.is_dir():
        log_fn(f"[WARN] Не найдена папка локальности: {loc_dir}")
        return

    if torch is None:
        log_fn("[ERR] PyTorch не установлен. Инференс невозможен.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_fn(f"[INFO] Устройство: {device} для локальности {locality}")

    model = SimpleHRNet(num_keypoints=num_keypoints)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    img_paths = sorted(loc_dir.glob("*.png"))
    if not img_paths:
        log_fn(f"[INFO] В {loc_dir} нет PNG-изображений.")
        return

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        img_resized, scale, offset_x, offset_y = _resize_and_pad(img, cfg)

        np_img = np.asarray(img_resized, dtype=np.float32) / 255.0
        np_img = np.transpose(np_img, (2, 0, 1))  # CHW
        tensor = torch.from_numpy(np_img).unsqueeze(0).to(device)

        with torch.no_grad():
            heatmaps = model(tensor)
            kps_resized = heatmaps_to_keypoints(heatmaps)[0]  # (K, 2)

        # Возвращаемся к координатам оригинального изображения
        if scale > 0:
            kps_original = kps_resized.clone()
            # Убираем паддинг
            kps_original[:, 0] -= offset_x
            kps_original[:, 1] -= offset_y
            # Убираем масштабирование
            kps_original /= scale
        else:
            kps_original = kps_resized

        # Гарантируем ровно num_keypoints строк
        if kps_original.shape[0] < num_keypoints:
            pad = torch.zeros((num_keypoints - kps_original.shape[0], 2), dtype=kps_original.dtype)
            kps_original = torch.cat([kps_original, pad], dim=0)
        elif kps_original.shape[0] > num_keypoints:
            kps_original = kps_original[:num_keypoints]

        out_csv = img_path.with_suffix(".csv")
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for i in range(num_keypoints):
                x = float(kps_original[i, 0].item())
                y = float(kps_original[i, 1].item())
                writer.writerow([x, y])

        log_fn(f"[OK] CSV сохранён для {img_path.name}")


def main(argv: Optional[List[str]] = None) -> int:
    """
    Точка входа для действия 2) Autolabel locality with current model.
    Варианты вызова:
    - без аргументов: берём base_dir из cfg\\last_base.txt и все локальности из localities_status.csv, у которых есть png-папка;
    - с аргументами: если первый аргумент — существующая папка, считаем её base_dir, а остальное — имена локальностей;
      иначе все аргументы считаем локальностями, а base_dir берём из cfg\\last_base.txt.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Autolabel locality with current HRNet model")
    parser.add_argument(
        "args",
        nargs="*",
        help="Необязательный base_dir, затем имена локальностей.",
    )
    args = parser.parse_args(argv)

    root = get_landmark_root()
    cfg_path = root / "config" / "hrnet_config.yaml"
    cfg = load_yaml_config(cfg_path)

    lm_number = read_lm_number(root)
    if lm_number <= 0:
        print("[ERR] LM_number.txt не найден или содержит неверное значение.")
        return 1

    base_dir: Optional[Path] = None
    localities: List[str] = []

    # Если первый аргумент — существующая директория, считаем её base_dir
    if args.args:
        first = Path(args.args[0])
        if first.is_dir():
            base_dir = first
            localities = [a for a in args.args[1:] if a.strip()]
        else:
            localities = [a for a in args.args if a.strip()]

    if base_dir is None:
        base_dir = read_last_base(root)
    if base_dir is None:
        print("[ERR] Базовая папка локальностей не определена (нет cfg\\last_base.txt).")
        return 1

    if not localities:
        # Fallback: все локальности, которые есть в localities_status.csv и имеют подпапку png
        status_file = root / "status" / "localities_status.csv"
        if status_file.is_file():
            with status_file.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    loc = row.get("locality", "").strip()
                    if not loc:
                        continue
                    loc_dir = base_dir / loc / "png"
                    if loc_dir.is_dir():
                        localities.append(loc)
        localities = sorted(set(localities))

    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "infer_hrnet_last.log"
    log_file = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    model_path = root / "models" / "current" / "hrnet_best.pth"
    if not model_path.is_file():
        log("[ERR] models/current/hrnet_best.pth не найден. Сначала нужно обучить модель (действие 1).")
        log_file.close()
        return 1

    log(f"[INFO] Базовая папка: {base_dir}")
    log(f"[INFO] Локальности для автолейблинга: {localities}")

    for loc in localities:
        infer_for_locality(root, base_dir, loc, cfg, lm_number, model_path, log)

    log_file.close()
    print("[INFO] Autolabel finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
