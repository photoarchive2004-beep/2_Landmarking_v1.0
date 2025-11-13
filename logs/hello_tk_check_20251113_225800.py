import sys
print('PY', sys.version)
try:
    import tkinter as tk, _tkinter
    r=tk.Tk(); r.withdraw(); r.update_idletasks(); r.destroy()
    print('TK HELLO: PASS')
except Exception as e:
    print('TK HELLO: FAIL', repr(e))
