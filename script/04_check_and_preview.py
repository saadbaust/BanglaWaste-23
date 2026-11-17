"""Step 2.7 - Final checks on the resized copy and the split files, plus a preview image.

Checks: every file in folds.csv exists in work/BanglaWaste23_256, 23 classes in every fold,
no object in two folds, every fold-0 validation object outside fold 0.
Writes work/logs/04_preview_grid.jpg (2 random images per class) - OPEN IT and look:
images upright, not squashed, correct class.

Run:  python script/04_check_and_preview.py     (from the repository root)
"""
import random
import pandas as pd
from PIL import Image, ImageDraw
from _paths import RESIZED, SPLITS, LOGS

def main():
    folds = pd.read_csv(SPLITS / "folds.csv")
    cm = pd.read_csv(SPLITS / "class_map.csv")
    f0 = pd.read_csv(SPLITS / "fold0_val.csv")
    problems = []
    missing = [f"{r.class_folder}/{r.file_name}" for r in folds.itertuples()
               if not (RESIZED / r.class_folder / r.file_name).exists()]
    if missing: problems.append(f"{len(missing)} resized files missing, e.g. {missing[:3]}")
    if len(folds) != 4160: problems.append(f"folds.csv has {len(folds)} rows")
    if folds.groupby("fold").class_id.nunique().min() != 23: problems.append("a fold is missing a class")
    if folds.groupby("object_id").fold.nunique().max() != 1: problems.append("an object is in two folds")
    if set(f0.file_name) & set(folds[folds.fold == 0].file_name): problems.append("fold0_val overlaps fold-0 test")
    if f0[f0.is_val].class_id.nunique() != 23: problems.append("fold0_val lacks a class")

    random.seed(0)
    T, cols = 160, 8
    names = list(cm.class_folder)
    grid = Image.new("RGB", (cols * T, ((len(names) * 2 + cols - 1) // cols) * (T + 16)), "white")
    d = ImageDraw.Draw(grid)
    k = 0
    for c in names:
        for fn in random.sample(list(folds[folds.class_folder == c].file_name), 2):
            im = Image.open(RESIZED / c / fn); im.thumbnail((T, T))
            x, y = (k % cols) * T, (k // cols) * (T + 16)
            grid.paste(im, (x + (T - im.width) // 2, y)); d.text((x + 2, y + T + 2), c[:24], fill="black")
            k += 1
    grid.save(LOGS / "04_preview_grid.jpg", quality=90)

    print("\n".join("  PROBLEM: " + p for p in problems) or "  all checks passed")
    print("Preview saved: work/logs/04_preview_grid.jpg  <- open and look at it")
    print("RESULT: PASS" if not problems else "RESULT: FAIL")

if __name__ == "__main__":
    main()
