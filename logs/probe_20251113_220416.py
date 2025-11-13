import sys
print('PY', sys.version)
try:
    import tkinter as tk
    print('tk:', tk.TkVersion)
except Exception as e:
    print('ERR tkinter:', repr(e))
try:
    import PIL, PIL.Image
    print('Pillow:', PIL.__version__)
except Exception as e:
    print('ERR Pillow:', repr(e))