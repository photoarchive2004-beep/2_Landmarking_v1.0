import re, sys, ast, pathlib
p = pathlib.Path(r'D:\GM\tools\2_Landmarking_v1.0\annot_gui_custom.py')
src = p.read_text(encoding='utf-8', errors='ignore')
mods = set()
try:
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                mods.add(a.name.split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                mods.add(n.module.split('.')[0])
except Exception:
    pass

# вероятные внешние (простая фильтрация stdlib)
stdlib_like = {'sys','os','time','math','re','json','pathlib','typing','itertools','functools','subprocess','traceback','tkinter','collections'}
mods = [m for m in mods if m not in stdlib_like]
print('\n'.join(sorted(mods)))
