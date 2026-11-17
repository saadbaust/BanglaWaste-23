"""Step 2.7 - Pack the resized images + split files into one zip for Kaggle.

Writes work/kaggle_upload/banglawaste23-256.zip containing
  BanglaWaste23_256/<23 class folders>/*.jpg   and   splits/*.csv, split_info.json
Upload it on kaggle.com -> Datasets -> New Dataset (Private), name: banglawaste23-256.

Run:  python script/05_make_kaggle_zip.py   (from the repository root)
"""
import zipfile
from _paths import WORK, RESIZED, SPLITS

def main():
    out = WORK / "kaggle_upload"; out.mkdir(exist_ok=True)
    z = out / "banglawaste23-256.zip"
    n = 0
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:      # JPEGs are already compressed
        for p in sorted(RESIZED.rglob("*.jpg")):
            zf.write(p, p.relative_to(WORK)); n += 1
        for p in sorted(SPLITS.iterdir()):
            zf.write(p, p.relative_to(WORK))
    print(f"{n} images + split files -> {z} ({z.stat().st_size / 1e6:.0f} MB)")
    print("RESULT: PASS" if n == 4160 else "RESULT: FAIL - expected 4160 images")

if __name__ == "__main__":
    main()
