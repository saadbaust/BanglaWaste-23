"""Resize the BDWaste shared-class images exactly like BanglaWaste-23 (script 02), then zip them for Kaggle.

Input : BDWaste_shared\Digestible\<class>\*.jpg  and  BDWaste_shared\Indigestible\<class>\*.jpg   (dataset root)
Output: work\BDWaste_shared_256\  (same folder structure, same file names)
        work\logs\06_bdwaste_sizes.csv
        work\kaggle_upload\bdwaste-shared-256.zip   -> upload this to Kaggle as the dataset "bdwaste"
Method: EXIF orientation applied, 8-bit RGB, shorter side 256 px (aspect ratio kept, Lanczos), JPEG quality 95.
Already-finished images are skipped, so the script can be stopped and restarted safely.

Run:  python script/06_resize_bdwaste.py        (from the repository root)
Time: about 2-5 minutes.
"""
import os, sys, zipfile
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
from _paths import ROOT, WORK, LOGS

SIZE = 256
SRC = ROOT / "BDWaste_shared"
DST = Path(sys.argv[1]) if len(sys.argv) > 1 else WORK / "BDWaste_shared_256"
EXT = {".jpg", ".jpeg", ".png"}

def resize_one(src):
    rel = src.relative_to(SRC)
    dst = (DST / rel).with_suffix(".jpg")
    if not dst.exists():
        with Image.open(src) as im:
            im.draft("RGB", (2 * SIZE, 2 * SIZE))
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            s = SIZE / min(w, h)
            im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.Resampling.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.stem + ".tmp.jpg")
            im.save(tmp, quality=95)
            os.replace(tmp, dst)
    with Image.open(dst) as im:
        return rel.parts[0], rel.parent.name, rel.name, im.size[0], im.size[1]

def main():
    if not SRC.is_dir():
        sys.exit(f"Not found: {SRC}")
    files = sorted(p for p in SRC.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
    print(f"{len(files)} images in {SRC}")
    out = []
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for i, rec in enumerate(pool.imap_unordered(resize_one, files, chunksize=8), 1):
            out.append(rec)
            if i % 200 == 0:
                print(f"  resized {i}/{len(files)}")
    df = pd.DataFrame(out, columns=["top", "subfolder", "file_name", "width", "height"]).sort_values(["top", "subfolder", "file_name"])
    df.to_csv(LOGS / "06_bdwaste_sizes.csv", index=False)
    print(df.groupby(["top", "subfolder"]).size().rename("images").to_string())

    up = WORK / "kaggle_upload"; up.mkdir(exist_ok=True)
    z = up / "bdwaste-shared-256.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(DST.rglob("*.jpg")):
            zf.write(p, Path("BDWaste_shared_256") / p.relative_to(DST))
    ok = len(df) == len(files) and (df[["width", "height"]].min(axis=1) == SIZE).all()
    print(f"\n{len(df)} images, shorter side {SIZE} for all: {ok}")
    print(f"Upload this file to Kaggle: {z} ({z.stat().st_size / 1e6:.0f} MB)")
    print("RESULT: PASS" if ok else "RESULT: FAIL - check work/logs/06_bdwaste_sizes.csv")

if __name__ == "__main__":
    main()
