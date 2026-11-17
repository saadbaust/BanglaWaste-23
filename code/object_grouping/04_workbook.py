"""Build the object-ID review workbook from proposed groups. Usage: python 04_workbook.py <groups_json> <out.xlsx> [finer_groups_json|-] [final_groups_json|-] [environment_json]"""
import sys, json, re
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import FormulaRule
gj, out = sys.argv[1:3]
G = json.load(open(gj)); classes = sorted(G, key=str.lower)
FINE = json.load(open(sys.argv[3])) if len(sys.argv) > 3 and sys.argv[3] != '-' else None
FINAL = json.load(open(sys.argv[4])) if len(sys.argv) > 4 and sys.argv[4] != '-' else G   # current object IDs after team edits
ENV = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else {}   # environment values per object (+ '_default')
def sub_letters(m, fm):
    seen, res = {}, {}
    for f in sorted(m, key=lambda f: int(re.search(r'_(\d{5})\.jpg$', f).group(1))):
        seen.setdefault(m[f], [])
        if fm[f] not in seen[m[f]]: seen[m[f]].append(fm[f])
        res[f] = 'abcdefghijklmnopqrstuvwxyz'[min(seen[m[f]].index(fm[f]), 25)]
    multi = {g for g, v in seen.items() if len(v) > 1}
    return {f: (l if m[f] in multi else '') for f, l in res.items()}
num = lambda f: int(re.search(r'_(\d{5})\.jpg$', f).group(1))
F = 'Arial'; hdr_font = Font(name=F, bold=True, color='FFFFFF'); hdr_fill = PatternFill('solid', fgColor='305496')
body = Font(name=F, size=10); yellow = PatternFill('solid', fgColor='FFF2CC'); thin = Side(style='thin', color='BFBFBF')
wb = Workbook(); wb.calculation.fullCalcOnLoad = True
def header(ws, cols, widths):
    for i, (c, w) in enumerate(zip(cols, widths), 1):
        cell = ws.cell(row=1, column=i, value=c); cell.font = hdr_font; cell.fill = hdr_fill
        cell.alignment = Alignment(wrap_text=True, vertical='center'); ws.column_dimensions[cell.column_letter].width = w
    ws.freeze_panes = 'A2'; ws.row_dimensions[1].height = 32
# ---------- Instructions
ws = wb.active; ws.title = 'Instructions'; ws.column_dimensions['A'].width = 120
lines = [
 ('BanglaWaste-23: object-ID review', True),
 ('Purpose: Data in Brief requires an object ID for every image (which physical item it shows) and the environmental conditions of each object.', False),
 ('The object IDs in this file are PROPOSALS made automatically from visual similarity of images taken in sequence (5-10 objects per class). Please check and correct them.', False),
 ('', False),
 ('Cells to edit are shaded yellow. Do not edit other cells.', True),
 ('', False),
 ('Step 1 — Sheet "Images": open the contact sheet for the class (folder contact_sheets/<class>.jpg). Each colour block is one proposed object.', False),
 ('   If two neighbouring proposed objects are the same physical item (or the same arrangement of items), give them the same object_id in column D.', False),
 ('   If one proposed object contains two different items, give the images of the second item a new ID (the next unused number in that class). Keep the format <class folder>_objNN.', False),
 ('Step 2 — Sheet "Objects": one row per object. Fill indoor_outdoor, lighting and background from the drop-down lists (notes is free text).', False),
 ('   Pre-filled: indoor_outdoor = Indoor for all. lighting = Natural, except where phone-flash light was seen (Flash = most images, Mixed = some images). Please confirm the Flash and Mixed rows.', False),
 ('   background is pre-filled from a visual check; Mixed = several backgrounds, listed in notes. Objects on uneven ground (soil, stone, grass) are marked Outdoor.', False),
 ('   If you created a new object ID in Step 1, add a row for it at the bottom. If an object no longer has any images (n_images = 0), delete its row.', False),
 ('Step 3 — Sheet "Summary": check that every class shows 0 in "images without object row" and 0 in "objects with 0 images", and that all objects have environment filled.', False),
 ('', False),
 ('What counts as one object: one physical item (e.g. one bottle) or one set of items photographed together (e.g. one pile of peanut shells), even when it was rearranged or photographed on another background.', True),
 ('Example of a completed Objects row (illustration only): Bottle_obj07 | Bottle | 25 | Bottle_img_00176.jpg | Indoor | Natural | Tile/concrete floor | (notes optional)', False),
]
for r, (t, b) in enumerate(lines, 1):
    c = ws.cell(row=r, column=1, value=t); c.font = Font(name=F, size=12 if r == 1 else 10, bold=b); c.alignment = Alignment(wrap_text=True)
# ---------- Images
im = wb.create_sheet('Images'); header(im, ['class_folder', 'file_name', 'proposed_object_id', 'object_id (edit)', 'note (optional)'] + (['finer split hint'] if FINE else []), [24, 34, 26, 26, 40, 16])
r = 2
for c in classes:
    L = sub_letters(G[c], FINE[c]) if FINE else {}
    for f in sorted(G[c], key=num):
        pid = f'{c}_obj{G[c][f]:02d}'
        vals = [c, f, pid, f'{c}_obj{FINAL[c][f]:02d}', None] + ([(f'obj{G[c][f]:02d}' + L[f]) if L.get(f) else None] if FINE else [])
        for j, v in enumerate(vals, 1):
            cell = im.cell(row=r, column=j, value=v); cell.font = body
            if j in (4, 5): cell.fill = yellow
        r += 1
last_img = r - 1
im.auto_filter.ref = f'A1:{"F" if FINE else "E"}{last_img}'
# ---------- Objects
ob = wb.create_sheet('Objects')
header(ob, ['object_id', 'class_folder', 'n_images', 'first image', 'indoor_outdoor', 'lighting', 'background', 'notes'], [26, 24, 10, 32, 15, 14, 18, 40])
r = 2
for c in classes:
    for g in sorted(set(FINAL[c].values())):
        oid = f'{c}_obj{g:02d}'
        ob.cell(row=r, column=1, value=oid); ob.cell(row=r, column=2, value=c)
        ob.cell(row=r, column=3, value=f'=COUNTIF(Images!$D$2:$D${last_img},A{r})')
        ob.cell(row=r, column=4, value=f'=IFERROR(INDEX(Images!$B$2:$B${last_img},MATCH(A{r},Images!$D$2:$D${last_img},0)),"(no images)")')
        e = dict(ENV.get('_default', {})); e.update(ENV.get(oid, {}))
        for j, key in ((5, 'indoor_outdoor'), (6, 'lighting'), (7, 'background'), (8, 'notes')):
            if key in e: ob.cell(row=r, column=j, value=e[key])
        for j in range(1, 9):
            cell = ob.cell(row=r, column=j); cell.font = body
            if j >= 5: cell.fill = yellow
        r += 1
last_obj = r - 1
lists = {'E': '"Indoor,Outdoor"', 'F': '"Natural,Flash,Mixed"', 'G': '"Tile/concrete floor,Wooden surface,Plain coloured sheet,Patterned cloth,White/light sheet,Dark surface,Soil/ground,Plate/bowl/tray,Mixed,Other"'}
for col, formula in lists.items():
    dv = DataValidation(type='list', formula1=formula, allow_blank=True); ob.add_data_validation(dv); dv.add(f'{col}2:{col}{last_obj + 300}')
ob.conditional_formatting.add(f'C2:C{last_obj}', FormulaRule(formula=[f'C2=0'], fill=PatternFill('solid', fgColor='F8CBAD')))
ob.auto_filter.ref = f'A1:H{last_obj}'
# ---------- Summary
sm = wb.create_sheet('Summary')
header(sm, ['class_folder', 'images', 'proposed objects', 'objects listed', 'objects with 0 images', 'images without object row', 'objects with environment filled'], [24, 10, 12, 12, 14, 16, 18])
OR = f'Objects!$B$2:$B${last_obj + 300}'
for i, c in enumerate(classes, 2):
    sm.cell(row=i, column=1, value=c)
    sm.cell(row=i, column=2, value=f'=COUNTIF(Images!$A$2:$A${last_img},A{i})')
    sm.cell(row=i, column=3, value=len(set(G[c].values())))
    sm.cell(row=i, column=4, value=f'=COUNTIF({OR},A{i})')
    sm.cell(row=i, column=5, value=f'=COUNTIFS({OR},A{i},Objects!$C$2:$C${last_obj + 300},0)')
    sm.cell(row=i, column=6, value=f'=SUMPRODUCT((Images!$A$2:$A${last_img}=A{i})*(COUNTIF(Objects!$A$2:$A${last_obj + 300},Images!$D$2:$D${last_img})=0))')
    sm.cell(row=i, column=7, value=f'=COUNTIFS({OR},A{i},Objects!$E$2:$E${last_obj + 300},"?*",Objects!$F$2:$F${last_obj + 300},"?*",Objects!$G$2:$G${last_obj + 300},"?*")')
    for j in range(1, 8): sm.cell(row=i, column=j).font = body
t = len(classes) + 2
sm.cell(row=t, column=1, value='Total').font = Font(name=F, bold=True)
for j, col in enumerate('BCDEFG', 2):
    c = sm.cell(row=t, column=j, value=f'=SUM({col}2:{col}{t-1})'); c.font = Font(name=F, bold=True)
sm.cell(row=t + 2, column=1, value='"proposed objects" is the automatic proposal (fixed number); the other columns update as you edit.').font = Font(name=F, italic=True, size=9)
wb.save(out); print('rows', last_img - 1, 'objects', last_obj - 1)
