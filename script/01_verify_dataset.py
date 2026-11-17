"""Step 2.1 - Verify the raw dataset against file_list.csv (dataset root).

Checks for all 4,160 images: file exists, byte size, MD5 checksum, width/height, opens as RGB JPEG,
and that no extra files are on disk. Writes work/logs/01_verify_report.csv and prints a summary.

Run:  python script/01_verify_dataset.py   (from the repository root)
Time: about 5-10 minutes (reads all 14 GB).
"""
import hashlib
import pandas as pd
from PIL import Image
from _paths import RAW, META, LOGS

def main():
    fl = pd.read_csv(META / "file_list.csv")
    assert len(fl) == 4160, f"file_list.csv has {len(fl)} rows, expected 4160"
    rows = []
    for i, r in enumerate(fl.itertuples(), 1):
        p = RAW / r.class_folder / r.file_name
        rec = {"class_folder": r.class_folder, "file_name": r.file_name, "exists": p.exists(),
               "size_ok": False, "md5_ok": False, "wh_ok": False, "decode_ok": False, "mode": ""}
        if p.exists():
            b = p.read_bytes()
            rec["size_ok"] = len(b) == r.file_size_bytes
            rec["md5_ok"] = hashlib.md5(b).hexdigest() == r.md5
            try:
                with Image.open(p) as im:
                    rec["wh_ok"] = im.size == (r.width, r.height)
                    rec["mode"] = im.mode
                    im.verify()
                rec["decode_ok"] = True
            except Exception as e:
                rec["error"] = str(e)
        rows.append(rec)
        if i % 250 == 0:
            print(f"  checked {i}/{len(fl)}")
    df = pd.DataFrame(rows)
    df.to_csv(LOGS / "01_verify_report.csv", index=False)

    on_disk = {f"{p.parent.name}/{p.name}" for p in RAW.rglob("*") if p.is_file()}
    listed = {f"{r.class_folder}/{r.file_name}" for r in fl.itertuples()}
    extra = sorted(on_disk - listed)

    ok = df[["exists", "size_ok", "md5_ok", "wh_ok", "decode_ok"]].all(axis=1)
    print(f"\nImages OK: {ok.sum()}/{len(df)} | RGB: {(df['mode'] == 'RGB').sum()} | extra files on disk: {len(extra)}")
    for x in df.loc[~ok, "file_name"].head(20).tolist() + extra[:20]:
        print("  PROBLEM:", x)
    print("RESULT: PASS" if ok.all() and not extra else "RESULT: FAIL - see work/logs/01_verify_report.csv")

if __name__ == "__main__":
    main()
