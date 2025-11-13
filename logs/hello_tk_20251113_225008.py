import sys
print('PY', sys.version)
try:
    import tkinter as tk, _tkinter, platform, os
    print('tkinter OK, TkVersion=', getattr(tk,'TkVersion',None))
    print('_tkinter:', getattr(_tkinter,'__file__',None))
    r = tk.Tk(); r.withdraw(); r.update_idletasks(); r.destroy()
    print('TK HELLO: PASS')
except Exception as e:
    print('TK HELLO: FAIL', repr(e))
