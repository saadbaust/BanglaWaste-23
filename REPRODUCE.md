# Reproducing the paper

The full pipeline runs in four stages. Stage 4 alone (figures and tables from the saved results) needs no GPU and no images.

| Stage | What | Where | Hardware |
|---|---|---|---|
| 1 | Download and prepare the images, rebuild the splits | `script/` | any PC |
| 2 | Train and evaluate all models | `code/*/` notebooks | NVIDIA T4 (Kaggle), ~1–2 h per notebook |
| 3 | Collect the run outputs into `results/` | `code/*/analysis/` | CPU |
| 4 | Figures and tables | `code/paper/` | CPU |

## Stage 1 — data

1. Download the dataset from Zenodo (DOI `TODO`) and extract the images so that the folders are
   `BanglaWaste-23-raw/<class_folder>/<file_name>.jpg` in the repository root (23 class folders, 4,160 images).
   The release metadata is already in `metadata/`.
2. Run, from the repository root:

```bash
python script/01_verify_dataset.py       # checks all 4,160 files against metadata/file_list.csv (MD5) -> RESULT: PASS
python script/02_resize.py               # EXIF rotation, RGB, shorter side 256 px (Lanczos), JPEG q95 -> work/BanglaWaste23_256/
python script/03_make_splits.py          # rebuilds the splits -> work/splits/ (identical to splits/; deterministic, seed 42)
python script/04_check_and_preview.py    # sanity checks + preview images
python script/05_make_kaggle_zip.py      # zip of the resized images + splits for a Kaggle dataset
```

Script 03 reproduces the files in `splits/` exactly. Line endings may differ between Windows and Linux.

For the cross-dataset test, download BDWaste (Rahman et al., 2024, *Data in Brief*). Copy the class folders it shares
with BanglaWaste-23 to `BDWaste_shared/Digestible/<class>/` and `BDWaste_shared/Indigestible/<class>/`, then run
`python script/06_resize_bdwaste.py`.

## Stage 2 — training (Kaggle notebooks)

Every notebook is self-contained: the training code is inside it. The same code is in the `*_core.py` files, which are
easier to read. Upload the zip from script 05 as a Kaggle dataset, add it as input, and run the notebook with a GPU (T4).
Each notebook writes its predictions, metrics and history as CSV/JSON. The outputs of our runs are in `runs/`.

| Notebooks | Experiment |
|---|---|
| `code/pretrained/round1_pilot/waste_R1_*` | Learning-rate tuning for the 11 pretrained models on the fold-0 inner split (LR 1e-3, 3e-4, 1e-4) |
| `code/pretrained/round2_main/waste_R2_*` | 5-fold object-disjoint benchmark of the 11 pretrained models (40 epochs, chosen LR); random-split runs for ConvNeXt-T, MobileNetV3-L, MobileViTv2 |
| `code/custom/round1_pilot/waste_C1_*` | Custom-model pilot on the tuning split: V0, redesign V1, knowledge distillation, MobileNetV3-L + custom neck |
| `code/batch3/notebooks/waste_B3_acc2_nb1`, `acc2_nb2`, `acc3_nb1` | Custom CNN V0 from scratch, 5 folds + random split (150 epochs), Grad-CAM |
| `code/batch3/notebooks/waste_B3_acc4_nb2` | MobileNetV3-L from scratch, 5 folds + random split, Grad-CAM |
| `code/batch3/notebooks/waste_B3_acc1_nb1` | Shortcut test (JPEG q30, 96 px) and Grad-CAM for ConvNeXt-T and EfficientNet-B0 (uses the Round 2 weights) |
| `code/batch3/notebooks/waste_B3_acc4_nb1` | BDWaste cross-dataset test (trains on all 4,160 images) |
| `code/pretrained/waste_pretrained_TEMPLATE.ipynb` | Same code with an editable job list, for new runs |

Training settings, shared by all models:
- AdamW (weight decay 1e-4), batch size 32, 5 warm-up epochs then cosine decay.
- MixUp 0.2 / CutMix 1.0 (p = 0.5); random resized crop 224 (scale 0.6–1.0); flips; rotation ±20°; brightness/contrast jitter 0.15.
- Mixed precision; EMA weights (0.999) evaluated after the final epoch.
- No early stopping, test-time augmentation or class weights.
- Seed 42 + fold number.

## Stage 3 — collect results

```bash
python code/pretrained/analysis/waste_analyse_round1.py   # LR choice   -> runs/round1_pilot/round1_lr_table.csv, round1_choice.csv
python code/pretrained/analysis/waste_analyse_round2.py   # benchmark   -> results/round2/
python code/batch3/analysis/waste_analyse_batch3.py       # V0, shortcut, BDWaste, Grad-CAM -> results/batch3/
```

They read the notebook outputs extracted under `runs/<round>/extracted/`. That folder layout is kept in this repository.

## Stage 4 — figures and tables

```bash
python code/paper/waste_make_figures_tables.py     # all figures (PDF+PNG) and tables (CSV+LaTeX) -> results/paper/
python code/paper/waste_make_manuscript_items.py   # protocol diagram, combined figure, journal-style tables
```

The scripts compute:
- pooled out-of-fold metrics
- 95% confidence intervals by bootstrap over the 172 objects (2,000 resamples)
- McNemar tests against the best model with Holm correction

Grad-CAM panels need the `gradcam_data.npz` files from the notebooks. They are not stored here because of their size,
so those panels are skipped when the files are missing.

## Object IDs (how they were made)

`code/object_grouping/` documents how images were grouped into physical objects:
1. Sequential photos were grouped by image features and border colour (`01_features.py`, `02_group.py`, `06_coarse.py`).
2. The groups were printed as contact sheets and an Excel review workbook (`03_sheets.py`, `04_workbook.py`).
3. The team's corrections were applied (`05_apply_edits.py`, `review_edits.json`).
4. Lighting scores were computed (`07_light_scores.py`), and per-object conditions were recorded in `object_environment.json`.

The final IDs are the `object_id` column in `metadata/image_metadata.csv` and `splits/folds.csv`.
