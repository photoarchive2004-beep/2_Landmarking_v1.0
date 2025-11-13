import os, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(r'D:\GM')
IMGS = Path(r'G:\Research\Morphology\Photos\Phoxinus\balkh-alakol_Ayagoz_KZ_25\png')
LOGS = HERE.parent / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)
gl = LOGS / 'annot_gui_last.log'

def log_line(msg):
    with open(gl, 'a', encoding='utf-8') as f:
        f.write(time.strftime('[%H:%M:%S] ') + msg + '\n')

try:
    log_line('STAGE0: wrapper start')
    import importlib.util
    spec = importlib.util.spec_from_file_location('annot_gui_custom', str(HERE.parent / 'annot_gui_custom.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log_line('STAGE1: module loaded')
    sys.argv = ['annot_gui_custom.py', '--root', str(ROOT), '--images', str(IMGS)]
    rc = mod.main() if hasattr(mod, 'main') else 0
    log_line(f'STAGE2: main finished rc={rc}')
    sys.exit(int(rc) if isinstance(rc, int) else 0)
except SystemExit as e:
    log_line(f'EXIT(SystemExit) code={e.code}')
    raise
except Exception as e:
    log_line('FATAL: ' + repr(e))
    with open(gl, 'a', encoding='utf-8') as f:
        traceback.print_exc(file=f)
    raise
