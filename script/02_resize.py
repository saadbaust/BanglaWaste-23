"""Step 2.2 - Make the 256-px training copy in work/BanglaWaste23_256/.

For every image: apply EXIF orientation (2,228 images carry a rotation tag), convert to 8-bit RGB,
resize so the SHORTER side is 256 px (aspect ratio kept, Lanczos), save as JPEG quality 95.
Cropping to 224 x 224 happens later, at training time.
Already-finished images are skipped, so the script can be stopped and restarted safely.

Run:  python script/02_resize.py                (from the repository root)
Time: about 5-15 minutes (uses all CPU cores).
"""
import os
from multiprocessing import Pool
import pandas as pd
from PIL import Image, ImageOps
from _paths import RAW, META, RESIZED, LOGS

SIZE = 256

def resize_one(args):
    cls, name = args
    dst = RESIZED / cls / name
    if not dst.exists():
        with Image.open(RAW / cls / name) as im:
            im.draft("RGB", (2 * SIZE, 2 * SIZE))       # fast JPEG decode at >= 2x target size
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            s = SIZE / min(w, h)
            im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.Resampling.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp.jpg")
            im.save(tmp, quality=95)
            os.replace(tmp, dst)                         # never leaves a half-written file
    with Image.open(dst) as im:
        return cls, name, im.size[0], im.size[1]

def main():
    fl = pd.read_csv(META / "file_list.csv")
    jobs = list(zip(fl.class_folder, fl.file_name))
    out = []
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for i, rec in enumerate(pool.imap_unordered(resize_one, jobs, chunksize=8), 1):
            out.append(rec)
            if i % 250 == 0:
                print(f"  resized {i}/{len(jobs)}")
    df = pd.DataFrame(out, columns=["class_folder", "file_name", "width", "height"]).sort_values(["class_folder", "file_name"])
    df.to_csv(LOGS / "02_resize_sizes.csv", index=False)
    short = df[["width", "height"]].min(axis=1)
    print(f"\nImages: {len(df)} | shorter side = {SIZE} for {(short == SIZE).sum()} | classes: {df.class_folder.nunique()}")
    print("RESULT: PASS" if len(df) == 4160 and (short == SIZE).all() else "RESULT: FAIL - check work/logs/02_resize_sizes.csv")

if __name__ == "__main__":
    main()
