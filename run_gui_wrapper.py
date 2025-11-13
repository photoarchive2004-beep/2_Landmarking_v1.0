import os,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
LOGS=(HERE/'logs'); LOGS.mkdir(parents=True,exist_ok=True)
gl=LOGS/'annot_gui_last.log'
def w(s): 
    with open(gl,'a',encoding='utf-8') as f: f.write(time.strftime('[%H:%M:%S] ')+s+'\n')
try:
    w('STAGE0: import module')
    import importlib.util
    spec=importlib.util.spec_from_file_location('annot_gui_custom', str(HERE.parent/'annot_gui_custom.py'))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    w('STAGE1: module loaded')
    sys.argv=['annot_gui_custom.py','--root',r'D:\GM','--images',r'G:\Research\Morphology\Photos\Phoxinus\balkh-alakol_Ayagoz_KZ_25\png']
    rc=mod.main() if hasattr(mod,'main') else 0
    w(f'STAGE2: main finished rc={rc}')
    sys.exit(int(rc) if isinstance(rc,int) else 0)
except SystemExit as e:
    w(f'EXIT(SystemExit) code={e.code}'); raise
except Exception as e:
    w('FATAL: '+repr(e)); traceback.print_exc(file=open(gl,'a',encoding='utf-8')); raise

