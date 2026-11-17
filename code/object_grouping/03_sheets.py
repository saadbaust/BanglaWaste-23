"""Contact sheets showing proposed object groups (one sheet per class).
Usage: python 03_sheets.py <work_dir> <groups_json> <out_dir> [thumb_px] [per_row] [class]
"""
import os, sys, json, re
from PIL import Image, ImageDraw, ImageFont
work, gj, out = sys.argv[1:4]; s = int(sys.argv[4]) if len(sys.argv) > 4 else 110; n = int(sys.argv[5]) if len(sys.argv) > 5 else 12
only = sys.argv[6] if len(sys.argv) > 6 else None
G = json.load(open(gj)); os.makedirs(out, exist_ok=True)
FINE = json.load(open(os.environ['FINE'])) if os.environ.get('FINE') else None
REVIEWED = {k for k in json.load(open(os.environ['EDITS'])) if not k.startswith('_')} if os.environ.get('EDITS') else set()
def sub_letters(m, fm):
    # letter for each file = order of its finer sub-group inside its proposed object; '' if the object has one sub-group
    seen, out_ = {}, {}
    for f in sorted(m, key=lambda f: int(re.search(r'_(\d{5})\.jpg$', f).group(1))):
        seen.setdefault(m[f], [])
        if fm[f] not in seen[m[f]]: seen[m[f]].append(fm[f])
        out_[f] = 'abcdefghijklmnopqrstuvwxyz'[min(seen[m[f]].index(fm[f]), 25)]
    multi = {g for g, v in seen.items() if len(v) > 1}
    return {f: (l if m[f] in multi else '') for f, l in out_.items()}
COL = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#9A6324', '#000075']
try: font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf', max(10, s // 9))
except Exception: font = ImageFont.load_default()
for c, m in G.items():
    if only and c != only: continue
    files = sorted(m, key=lambda f: int(re.search(r'_(\d{5})\.jpg$', f).group(1)))
    L = sub_letters(m, FINE[c]) if FINE and c not in REVIEWED else {f: '' for f in files}
    rows = (len(files) + n - 1) // n; pad = 4; lab = s // 6
    sheet = Image.new('RGB', (n * (s + pad) + pad, rows * (s + pad + lab) + pad + 30), 'white'); d = ImageDraw.Draw(sheet)
    title = (f'{c}: {len(files)} images, {len(set(m.values()))} objects AFTER TEAM REVIEW (colour = object; label = file number / object number)' if c in REVIEWED
             else f'{c}: {len(files)} images, {len(set(m.values()))} proposed objects (colour = object; label = file number / object number' + ('; a, b, c = finer split suggestion)' if FINE else ')'))
    d.text((pad, 6), title, fill='black', font=font)
    for k, f in enumerate(files):
        x = pad + (k % n) * (s + pad); y = 30 + pad + (k // n) * (s + pad + lab); g = m[f]; col = COL[(g - 1) % len(COL)]
        d.rectangle([x - 3, y - 3, x + s + 2, y + s + 2], fill=col)
        sheet.paste(Image.open(os.path.join(work, 'thumbs', c, f)).resize((s - 2, s - 2)), (x + 1, y + 1))
        num = int(re.search(r'_(\d{5})\.jpg$', f).group(1))
        d.text((x, y + s + 2), f'{num:03d} / obj{g:02d}{L[f]}', fill=col, font=font)
    sheet.save(os.path.join(out, f'{c}.jpg'), quality=82)
print('ok')
