import importlib.util, sys, traceback, os
from pathlib import Path
p = Path(r'D:\GM\tools\2_Landmarking_v1.0')/'annot_gui_custom.py'
print('IMPORT', p)
try:
    spec = importlib.util.spec_from_file_location('annot_gui_custom', str(p))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print('IMPORT: OK; has main=', hasattr(mod,'main'))
except Exception:
    print('IMPORT: FAIL')
    traceback.print_exc()
