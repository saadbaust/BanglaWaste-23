# BanglaWaste-23: household and hospital waste images with an object-disjoint benchmark

Code, evaluation splits and full results for the paper

> **BanglaWaste-23: A household and hospital waste image dataset with an object-disjoint benchmark showing that random
> splits overstate generalisation.** Jahan N., Esha M.E.J., Sadia M., Ahmed S., Alam S.M.J. *Waste Management* (under review).

The **images** are on Zenodo (link below). This repository holds everything else needed to check or reproduce the
paper: data-preparation scripts, the exact split files, training notebooks, analysis scripts, all saved predictions and
metrics, and the scripts that make every figure and table.

## The dataset in brief

- 4,160 smartphone images at original resolution, **23 classes** in three groups: food and plant (14 classes, 2,412 images),
  household non-food (5, 967) and **medical waste collected at a hospital** (4, 781).
- **172 physical objects**, each with an ID: every image is linked to the item it shows, with per-object capture
  conditions (indoor/outdoor, lighting, background).
- Collected in Saidpur and Thakurgaon, Bangladesh; labels cross-checked by two people, medical classes verified by a nurse.
- Images: Zenodo, DOI **`TODO`** (restricted during review; public on publication), licence CC BY-NC 4.0.

## Main findings

Object-disjoint 5-fold cross-validation: no physical object appears in both training and test data.

| Model | Training | Params (M) | MACs (M) | Pooled macro-F1 % [95% CI] |
|---|---|---:|---:|---|
| ConvNeXt-T | ImageNet-pretrained | 27.8 | 4,455 | **98.82** [94.5, 99.4] |
| ResNet-50 | ImageNet-pretrained | 23.6 | 4,087 | 98.62 [94.2, 99.1] |
| MobileViTv2 | ImageNet-pretrained | 4.4 | 1,412 | 98.52 [94.2, 99.0] |
| DenseNet-121 | ImageNet-pretrained | 7.0 | 2,833 | 98.47 [94.1, 99.1] |
| DeiT-S | ImageNet-pretrained | 21.7 | 4,599 | 98.37 [93.9, 99.0] |
| EfficientNet-B0 | ImageNet-pretrained | 4.0 | 385 | 98.29 [93.9, 98.9] |
| MobileNetV3-L | ImageNet-pretrained | 4.2 | 215 | 97.91 [93.3, 98.7] |
| MobileNetV2 | ImageNet-pretrained | 2.3 | 300 | 97.80 [93.3, 98.4] |
| GhostNetV2 | ImageNet-pretrained | 4.9 | 166 | 97.76 [93.3, 98.3] |
| EfficientFormerV2-S0 | ImageNet-pretrained | 3.3 | 396 | 97.43 [93.1, 98.2] |
| MobileNetV4-S | ImageNet-pretrained | 2.5 | 185 | 96.03 [91.7, 96.8] |
| Custom CNN V0 | from scratch | 0.38 | 165 | 81.42 [75.6, 83.0] |
| MobileNetV3-L | from scratch | 4.2 | 215 | 79.54 [73.4, 81.6] |

**A random image-level split overstates accuracy on new objects.** It puts 155 of the 172 objects in both training and
test data:

| Model | Random split | Unseen objects | Gap (points) |
|---|---:|---:|---:|
| ConvNeXt-T (pretrained) | 100 | 98.82 | 1.2 |
| MobileNetV3-L (pretrained) | 100 | 97.91 | 2.1 |
| Custom CNN V0 (from scratch) | 96.87 | 81.42 | 15.4 |
| MobileNetV3-L (from scratch) | 98.21 | 79.54 | 18.7 |

Other analyses in the paper:
- **Medical waste:** pretrained models recall 99–100% of medical items; from-scratch models recall only 87–89%.
- **Hardest objects:** errors concentrate on visually similar peels.
- **Shortcut test:** JPEG compression and low resolution cost only 1–3 points, and the per-device check shows no phone effect.
- **Cross-region test:** on BDWaste images from Dhaka, accuracy drops by 15–20 points.
- **Grad-CAM:** visual explanations for four models.

## Repository layout

| Folder | Contents |
|---|---|
| `splits/` | **The evaluation splits used in the paper**: `folds.csv` (object-disjoint 5 folds), `fold0_val.csv` (tuning split), `random_split.csv` (leakage comparison only), `class_map.csv`, `fold_counts.csv`, `split_info.json`. See `splits/README.md`. |
| `metadata/` | Release metadata (same as on Zenodo): per-image and per-object information, file list with MD5 checksums, class summary |
| `script/` | Data preparation: verify download, resize, make splits, checks, Kaggle zip, BDWaste resize |
| `code/pretrained/` | Training code and notebooks for the 11 pretrained models: learning-rate tuning (`round1_pilot`) and the 5-fold benchmark plus random split (`round2_main`) |
| `code/custom/` | Pilot of custom models: the authors' earlier CNN (V0), a redesign, knowledge distillation, pretrained backbone + custom neck |
| `code/batch3/` | V0 and MobileNetV3-L from scratch on 5 folds and random split, shortcut test, Grad-CAM, BDWaste cross-dataset test |
| `code/object_grouping/` | How object IDs were assigned (feature grouping, contact sheets, review edits) |
| `code/paper/` | Make every figure and table from the saved results (no GPU) |
| `results/` | Aggregated results: summaries, out-of-fold predictions for every model, per-class metrics, figures and tables |
| `runs/` | Raw logs of every training run: per-epoch history, per-fold metrics and predictions (CSV/JSON) |

## Quick start

```bash
pip install -r requirements.txt

# Re-create every figure and table from the saved results (CPU only)
python code/paper/waste_make_figures_tables.py
python code/paper/waste_make_manuscript_items.py
```

To reproduce everything from the images (data preparation, training on Kaggle GPUs, analysis), follow **[REPRODUCE.md](REPRODUCE.md)**.

### Using the splits with your own model

```python
import pandas as pd
folds = pd.read_csv("splits/folds.csv")            # file_name, class_folder, class_id, group, object_id, device, fold
for k in range(5):
    train = folds[folds.fold != k]; test = folds[folds.fold == k]
    assert not set(train.object_id) & set(test.object_id)   # no object in both
```

Report pooled out-of-fold macro-F1 over all 4,160 images, so your results are comparable with the table above.

## Citation

Please cite the paper, and the dataset when you use the images. `CITATION.cff` has the details; the BibTeX will be
added on publication.

## Licence

- **Code** (`script/`, `code/`): MIT License (see `LICENSE`).
- **Dataset, metadata and split files** (`metadata/`, `splits/`, images on Zenodo): CC BY-NC 4.0.
- **Results** (`results/`, `runs/`): CC BY-NC 4.0.

## Contact

Saad Ahmed and S M Jahangir Alam, Department of Computer Science and Engineering, Bangladesh Army University of Science
and Technology (BAUST), Saidpur, Bangladesh.
