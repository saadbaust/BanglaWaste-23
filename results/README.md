# Results

| Path | Contents |
|---|---|
| `round2/summary.csv` | 11 pretrained models: params, MACs, LR, fold mean ± SD, pooled out-of-fold metrics with CI, GPU throughput |
| `round2/fold_results.csv` | Every model × fold |
| `round2/oof_predictions_<model>.csv` | Out-of-fold prediction for all 4,160 images (true, predicted, class probabilities, object, fold) |
| `round2/per_class_pooled.csv`, `confusion_pooled/` | Per-class metrics and confusion matrices |
| `round2/groups.csv`, `leakage.csv` | Food/household/medical group accuracy; random vs object-disjoint |
| `batch3/` | From-scratch models (V0, MobileNetV3-L) as above, `leakage_all.csv`, shortcut test, BDWaste test (`bdwaste_*`), Grad-CAM item lists |
| `custom_pilot/pilot_summary.csv` | Custom-model pilot on the tuning split |
| `reference/` | Copies of the split and metadata files used by the analysis scripts |
| `paper/tables/` | Every table of the paper and supplementary material (CSV + LaTeX) |
| `paper/figures/` | Every figure (PDF + PNG), except the Grad-CAM panels |
| `master_results.xlsx` | Main tables in one workbook |

Model names in the files are timm names: for example, `convnext_tiny` is ConvNeXt-T and `custom_v0` is the custom CNN V0.

## Citation

Please cite the paper and the dataset if you use these files (full details in the [main README](../README.md#citation)).

**Paper (arXiv preprint):**

```bibtex
@misc{jahan2026banglawaste23,
  title         = Do waste classifiers recognise items they have never seen? Object-disjoint evaluation on household and medical waste},
  author        = {Jahan, Nusrat and Esha, Mst. Eshrat Jahan and Sadia, Masuma and Ahmed, Saad and Hassan, Md Zahim and Alam, S M Jahangir},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.XXXX.XXXXX},
  url           = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

**Dataset (Zenodo):**

```bibtex
@misc{banglawaste23_dataset,
  title     = {{BanglaWaste-23}: A Fine-Grained Image Dataset of Household and Medical Waste from Bangladesh},
  author    = {Jahan, Nusrat and Esha, Mst. Eshrat Jahan and Sadia, Masuma and Ahmed, Saad and Hassan, Md Zahim and Alam, S M Jahangir},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23694940},
  url       = {https://doi.org/10.5281/zenodo.23694940}
}
```
