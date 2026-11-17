# Split files

All files have one row per image. Join them to the images on `class_folder/file_name`.

| File | Columns | Use |
|---|---|---|
| `folds.csv` | file_name, class_folder, class_id, group, object_id, device, fold (0–4) | **Main evaluation**: object-disjoint 5-fold cross-validation |
| `fold0_val.csv` | file_name, class_folder, class_id, object_id, is_val | Tuning only: folds 1–4 with one object per class held out (2,689 train / 623 validation images, 23 validation objects) |
| `random_split.csv` | file_name, class_folder, class_id, object_id, split (train/val/test) | Leakage comparison only: stratified 70/15/15 random image split (155 of 172 objects in both train and test) |
| `class_map.csv` | class_id, class_folder, item_name, group | Class index, readable name and group (Food and plant / Household non-food / Medical) |
| `fold_counts.csv` | images per class per fold | Check |
| `split_info.json` | sizes, seed, validation objects | Check |

How the folds were made:
- Within each class, objects are sorted by number of images, largest first.
- The first five objects go one to each fold.
- Each remaining object goes to the fold that holds the fewest images of that class.
- Seed 42.

Result: every class is in every fold, and no object is in two folds. Fold sizes are 848 / 936 / 811 / 740 / 825 images
(32 / 33 / 32 / 38 / 37 objects).

Never tune on the test fold. Report pooled out-of-fold metrics over all 4,160 images.
