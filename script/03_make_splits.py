"""Step 2.3-2.6 - Class map, object-disjoint 5 folds, fold-0 inner validation, random split.

Reads  image_metadata.csv (dataset root)  (4,160 rows, one object_id per image)
Writes work/splits/class_map.csv, folds.csv, fold0_val.csv, random_split.csv,
       fold_counts.csv, split_info.json
Deterministic: running it again gives identical files. Never edit the outputs by hand.

Run:  python script/03_make_splits.py           (from the repository root)
Time: a few seconds.
"""
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from _paths import META, SPLITS
DATA = SPLITS
meta = pd.read_csv(META / "image_metadata.csv")
assert len(meta) == 4160 and meta.class_folder.nunique() == 23 and meta.object_id.nunique() == 172

# ---- 2.3 class map (order = class_folder lower-cased; item names and groups from DIB Table 1)
INFO = {
    "Banana_peeling": ("Banana peel", "Food and plant"),
    "Bottle": ("Plastic bottle", "Household non-food"),
    "cloth": ("Cloth scraps", "Household non-food"),
    "cotton": ("Used cotton wool", "Medical"),
    "cucumber_peeling": ("Cucumber peel", "Food and plant"),
    "Egg": ("Eggshell", "Food and plant"),
    "Injection": ("Syringe", "Medical"),
    "Lemon": ("Lemon waste", "Food and plant"),
    "malta_peeling": ("Malta peel", "Food and plant"),
    "mango_leaf": ("Mango leaf", "Food and plant"),
    "mango_peeling": ("Mango peel", "Food and plant"),
    "medicine": ("Medicine blister pack", "Medical"),
    "nut": ("Peanut shell", "Food and plant"),
    "orange_peeling": ("Orange peel", "Food and plant"),
    "packet": ("Snack packet", "Household non-food"),
    "pen": ("Pen", "Household non-food"),
    "pointed_gourd_peeling": ("Pointed gourd peel", "Food and plant"),
    "polythene": ("Polythene bag", "Household non-food"),
    "potato_peeling": ("Potato peel", "Food and plant"),
    "pumpkin": ("Pumpkin peel", "Food and plant"),
    "saline": ("Saline (IV infusion) set", "Medical"),
    "sugarcane": ("Sugarcane bagasse", "Food and plant"),
    "watermelon": ("Watermelon rind", "Food and plant"),
}
assert set(INFO) == set(meta.class_folder)
folders = sorted(INFO, key=str.lower)
cm = pd.DataFrame({"class_id": range(23), "class_folder": folders,
                   "item_name": [INFO[f][0] for f in folders],
                   "group": [INFO[f][1] for f in folders]})
cm.to_csv(DATA / "class_map.csv", index=False)
meta = meta.merge(cm[["class_folder", "class_id", "group"]], on="class_folder")

# ---- 2.4 object-disjoint stratified 5-fold split
# StratifiedGroupKFold cannot guarantee every class in every fold here (several classes have
# only 5 objects of very unequal size), so objects are assigned per class directly:
# within each class, objects are taken largest first; the first 5 go one to each fold, every
# later object goes to the fold that holds the fewest images of that class. Ties are broken
# randomly with a fixed seed. Result: every fold has >= 1 object of every class.
import numpy as np
SEED = 42
rng = np.random.default_rng(SEED)
obj = meta.groupby(["class_id", "object_id"]).size().rename("n").reset_index()
obj_fold = {}
for cid, g in obj.groupby("class_id"):
    g = g.sample(frac=1, random_state=int(rng.integers(1_000_000))).sort_values("n", ascending=False, kind="stable")
    load = np.zeros(5, dtype=int)
    order = rng.permutation(5)
    for i, r in enumerate(g.itertuples()):
        if i < 5:
            k = int(order[i])
        else:
            cand = np.flatnonzero(load == load.min())
            k = int(rng.choice(cand))
        obj_fold[r.object_id] = k
        load[k] += r.n
meta["fold"] = meta.object_id.map(obj_fold)
seed = SEED
assert meta.groupby("fold").class_id.nunique().min() == 23     # every class in every fold
assert meta.groupby("object_id").fold.nunique().max() == 1      # no object in two folds
cols = ["file_name", "class_folder", "class_id", "group", "object_id", "device", "fold"]
meta[cols].to_csv(DATA / "folds.csv", index=False)
counts = pd.crosstab(meta.class_folder, meta.fold, margins=True, margins_name="total")
counts.columns = [f"fold{c}" if c != "total" else c for c in counts.columns]
counts.to_csv(DATA / "fold_counts.csv")

# ---- 2.5 fold-0 inner validation: one held-out object per class from folds 1-4
tr0 = meta[meta.fold != 0]
val_objs = (tr0.drop_duplicates("object_id").groupby("class_id")
               .sample(1, random_state=42).object_id)
fold0 = tr0.assign(is_val=tr0.object_id.isin(val_objs))
fold0[["file_name", "class_folder", "class_id", "object_id", "is_val"]].to_csv(DATA / "fold0_val.csv", index=False)

# ---- 2.6 image-level random split (ONLY for the leakage comparison, Step 8.1)
trv, te = train_test_split(meta, test_size=0.15, stratify=meta.class_id, random_state=42)
tr, va = train_test_split(trv, test_size=0.15 / 0.85, stratify=trv.class_id, random_state=42)
rs = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])
rs[["file_name", "class_folder", "class_id", "object_id", "split"]].sort_values("file_name").to_csv(
    DATA / "random_split.csv", index=False)
shared = len(set(rs[rs.split == "train"].object_id) & set(rs[rs.split == "test"].object_id))

info = {
    "n_images": int(len(meta)), "n_classes": 23, "n_objects": int(meta.object_id.nunique()),
    "folds_seed": seed, "fold_sizes_images": meta.fold.value_counts().sort_index().tolist(),
    "fold_sizes_objects": meta.groupby("fold").object_id.nunique().tolist(),
    "fold0_inner": {"train_images": int((~fold0.is_val).sum()), "val_images": int(fold0.is_val.sum()),
                    "val_objects": sorted(val_objs.tolist())},
    "random_split_sizes": rs.split.value_counts().to_dict(),
    "random_split_objects_shared_train_test": shared,
}
(DATA / "split_info.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
print(counts)
