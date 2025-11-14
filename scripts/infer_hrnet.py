from __future__ import annotations

import sys
from pathlib import Path


def get_landmark_root() -> Path:
    """tools/2_Landmarking_v1.0 folder (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def get_base_localities(root: Path) -> Path:
    """
    Read base_localities from cfg/last_base.txt.

    Должно совпадать с тем, что использует аннотатор и train_hrnet.py.
    """
    cfg_dir = root / "cfg"
    last_base = cfg_dir / "last_base.txt"
    if not last_base.exists():
        raise RuntimeError("cfg/last_base.txt not found.")
    text = last_base.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("cfg/last_base.txt is empty.")
    return Path(text)


def parse_args(argv: list[str]) -> str | None:
    """
    Very small parser: expect --locality NAME.
    """
    locality = None
    for i, arg in enumerate(argv):
        if arg == "--locality" and i + 1 < len(argv):
            locality = argv[i + 1]
    if not locality:
        print("[ERR] --locality argument is required.")
        return None
    return locality


def main(argv: list[str] | None = None) -> int:
    """
    Каркас инференса по ТЗ_1.0:

    - читает cfg/last_base.txt;
    - проверяет, что есть папка локальности и PNG;
    - проверяет, что модель models/current/hrnet_best.pth существует.

    Сейчас это ЗАГЛУШКА: нейросеть HRNet/MMPose ещё не подключена.
    CSV с ландмарками не изменяются.
    """
    from pathlib import Path  # локальный импорт на всякий случай

    if argv is None:
        argv = sys.argv[1:]

    locality = parse_args(argv)
    if locality is None:
        return 1

    root = get_landmark_root()
    try:
        base_localities = get_base_localities(root)
    except Exception as exc:
        print("[ERR] Cannot read cfg/last_base.txt:")
        print(f"      {exc}")
        return 1

    loc_dir = base_localities / locality / "png"
    if not loc_dir.is_dir():
        print(f"[ERR] Locality folder not found: {loc_dir}")
        return 2

    images = sorted(loc_dir.glob("*.png"))
    if not images:
        print(f"[ERR] No PNG files found for locality \"{locality}\".")
        return 3

    model_path = root / "models" / "current" / "hrnet_best.pth"
    if not model_path.exists():
        print("[ERR] Current model not found: models/current/hrnet_best.pth")
        print("Run action 1 (Train) before autolabel.")
        return 4

    # Здесь будет реальный вызов HRNet/MMPose.
    # Пока делаем безопасную заглушку, чтобы не портить CSV:
    print(f"[INFO] Autolabel placeholder for locality \"{locality}\".")
    print("       Neural network inference is not implemented yet.")
    print("       No CSV landmark files were changed.")
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
