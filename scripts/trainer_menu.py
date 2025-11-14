from pathlib import Path
import csv
import subprocess
import sys

def find_tool_dir() -> Path:
    return Path(__file__).resolve().parent.parent

def rebuild_status(tool_dir: Path) -> int:
    script = tool_dir / "scripts" / "rebuild_localities_status.py"
    cmd = [sys.executable, str(script)]
    result = subprocess.run(cmd)
    return result.returncode

def load_status(tool_dir: Path):
    status_file = tool_dir / "status" / "localities_status.csv"
    if not status_file.exists():
        return []
    with status_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def format_localities(rows):
    if not rows:
        return []
    max_name = max(len(r["locality"]) for r in rows)
    lines = []
    for idx, r in enumerate(sorted(rows, key=lambda x: x["locality"]), start=1):
        name = r["locality"]
        n_images = r.get("n_images", "") or "0"
        n_labeled = r.get("n_labeled", "") or "0"
        try:
            ni = int(n_images)
            nl = int(n_labeled)
            pct = int(round(100.0 * nl / ni)) if ni > 0 else 0
        except ValueError:
            pct = 0
        status = r.get("status", "") or "(no status)"
        auto_quality = r.get("auto_quality", "")
        if status.startswith("AUTO") and auto_quality:
            status = f"AUTO {auto_quality}"
        name_pad = name.ljust(max_name + 2)
        cnt = f"[{n_labeled}/{n_images}]".ljust(12)
        pct_str = f"{pct:3d}%"
        line = f"[{idx}] {name_pad}{cnt} {pct_str}  {status}"
        lines.append(line)
    return lines

def main() -> int:
    tool_dir = find_tool_dir()
    print("== GM Landmarking: HRNet Trainer v1.0 ==")

    rc = rebuild_status(tool_dir)
    if rc != 0:
        print("[ERR] Could not rebuild status, aborting.")
        return rc

    rows = load_status(tool_dir)
    lines = format_localities(rows)

    print()
    print("Localities:")
    if not lines:
        print("  (no localities found)")
    else:
        for line in lines:
            print("  " + line)

    print()
    print("Actions:")
    print("  1) Train / finetune model on MANUAL localities   [STUB]")
    print("  2) Autolabel locality with current model         [STUB]")
    print("  3) Review AUTO locality in annotator             [STUB]")
    print("  4) Show current model info                       [STUB]")
    print("  5) Settings / config                             [STUB]")
    print("  Q) Quit")
    choice = input("Select action: ").strip()
    if choice.upper() == "Q":
        return 0

    print()
    print("This action is not implemented yet. Trainer menu is a skeleton for now.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
