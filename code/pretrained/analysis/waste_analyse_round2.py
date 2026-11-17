"""Round 2 (main results) + Round 3 (leakage) analysis for the pretrained models.

Reads  runs/round2_main/extracted/runs/{cv,random}/<model>_lr<lr>/<fold|random>/{metrics.csv,predictions.csv}
       results/class_map.csv
Writes results/round2/
  fold_results.csv        one row per model x fold
  summary.csv             per model: params, MACs, fold mean/SD, pooled out-of-fold (OOF) metrics, 95% CI, McNemar
  per_class_pooled.csv    per model x class: precision, recall, F1, support (pooled OOF)
  groups.csv              per model: 3-group accuracy (food/household/medical), medical recall/precision, cross-group errors
  leakage.csv             random image-level split vs object-disjoint (pooled OOF)
  oof_predictions_<model>.csv  every image predicted once (all 4,160)
  confusion_pooled/<model>.csv
and updates results/master_results.xlsx (FoldResults, Summary, PerClass, Groups, Leakage).
Statistics: 95% CI = cluster bootstrap over the 172 objects (2,000 resamples); McNemar (chi-square, continuity
correction) of each model vs the best model on pooled OOF correctness, Holm-corrected.
Run:   python code/pretrained/analysis/waste_analyse_round2.py   (from the Claude folder)
"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs" / "round2_main" / "extracted" / "runs"
OUT = ROOT / "results" / "round2"; (OUT / "confusion_pooled").mkdir(parents=True, exist_ok=True)
CM = pd.read_csv(ROOT / "results" / "class_map.csv")
NAMES = CM.class_folder.tolist(); K = len(NAMES)
GROUP = dict(zip(CM.class_id, CM.group))
NICE = {"mobilenetv2_100": "MobileNetV2", "mobilenetv3_large_100": "MobileNetV3-Large", "mobilenetv4_conv_small": "MobileNetV4-Conv-S",
        "efficientnet_b0": "EfficientNet-B0", "ghostnetv2_100": "GhostNetV2", "mobilevitv2_100": "MobileViTv2-1.0",
        "efficientformerv2_s0": "EfficientFormerV2-S0", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
        "convnext_tiny": "ConvNeXt-Tiny", "deit_small_patch16_224": "DeiT-Small"}

def macro_f1_fast(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    return np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0))

# ---------- load
fold_rows, oof = [], {}
for d in sorted(glob.glob(str(RUNS / "cv" / "*" / "fold*"))):
    m = pd.read_csv(Path(d) / "metrics.csv").iloc[0]
    fold_rows.append({k: m[k] for k in ["model", "lr", "epochs", "fold", "accuracy", "macro_f1", "balanced_accuracy",
                                        "weighted_f1", "params", "macs_M", "minutes", "sec_per_epoch", "gpu_infer_img_per_s", "gpu"]})
    oof.setdefault(m.model, []).append(pd.read_csv(Path(d) / "predictions.csv").assign(fold=int(m.fold)))
F = pd.DataFrame(fold_rows).sort_values(["model", "fold"])
F.insert(1, "name", F.model.map(NICE))
F.to_csv(OUT / "fold_results.csv", index=False)
assert F.groupby("model").fold.nunique().eq(5).all(), "a model is missing folds"

rng = np.random.default_rng(0); B = 2000
summ, perclass, groups = [], [], []
P = {}
for model, parts in oof.items():
    o = pd.concat(parts).sort_values("file_name").reset_index(drop=True)
    assert len(o) == 4160 and o.file_name.is_unique
    P[model] = o
    o.to_csv(OUT / f"oof_predictions_{model}.csv", index=False)
    y, p = o.y_true.values, o.y_pred.values
    cm = pd.crosstab(pd.Categorical(y, range(K)), pd.Categorical(p, range(K)), dropna=False); cm.index = NAMES; cm.columns = NAMES
    cm.to_csv(OUT / "confusion_pooled" / f"{model}.csv")
    # cluster bootstrap over objects
    objs = o.object_id.values; uo, inv = np.unique(objs, return_inverse=True)
    idx_by_obj = [np.flatnonzero(inv == i) for i in range(len(uo))]
    bs_f1, bs_acc = [], []
    for _ in range(B):
        take = np.concatenate([idx_by_obj[i] for i in rng.integers(0, len(uo), len(uo))])
        bs_f1.append(macro_f1_fast(y[take], p[take])); bs_acc.append((y[take] == p[take]).mean())
    f = F[F.model == model]
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(K), zero_division=0)
    for i in range(K):
        perclass.append({"model": model, "name": NICE[model], "class": NAMES[i], "group": GROUP[i],
                         "precision": pr[i], "recall": rc[i], "f1": f1[i], "support": int(sup[i])})
    gy = np.vectorize(GROUP.get)(y); gp = np.vectorize(GROUP.get)(p)
    med_t, med_p = gy == "Medical", gp == "Medical"
    groups.append({"model": model, "name": NICE[model], "group_accuracy": (gy == gp).mean(),
                   "medical_recall": (med_t & med_p).sum() / med_t.sum(), "medical_precision": (med_t & med_p).sum() / max(med_p.sum(), 1),
                   "medical_to_nonmedical_errors": int((med_t & ~med_p).sum()), "nonmedical_to_medical_errors": int((~med_t & med_p).sum()),
                   "class_accuracy_medical_items": (y[med_t] == p[med_t]).mean()})
    summ.append({"model": model, "name": NICE[model], "lr": f.lr.iloc[0], "epochs": int(f.epochs.iloc[0]),
                 "params_M": f.params.iloc[0] / 1e6, "macs_M": f.macs_M.iloc[0],
                 "acc_mean": f.accuracy.mean(), "acc_sd": f.accuracy.std(), "macro_f1_mean": f.macro_f1.mean(), "macro_f1_sd": f.macro_f1.std(),
                 "bal_acc_mean": f.balanced_accuracy.mean(), "weighted_f1_mean": f.weighted_f1.mean(),
                 "oof_acc": (y == p).mean(), "oof_macro_f1": f1_score(y, p, average="macro"), "oof_bal_acc": balanced_accuracy_score(y, p),
                 "oof_macro_f1_ci_low": np.percentile(bs_f1, 2.5), "oof_macro_f1_ci_high": np.percentile(bs_f1, 97.5),
                 "oof_acc_ci_low": np.percentile(bs_acc, 2.5), "oof_acc_ci_high": np.percentile(bs_acc, 97.5),
                 "train_min_per_fold": f.minutes.mean(), "gpu_infer_img_per_s": f.gpu_infer_img_per_s.mean()})
S = pd.DataFrame(summ).sort_values("oof_macro_f1", ascending=False).reset_index(drop=True)
best = S.model.iloc[0]; cb = P[best].y_true.values == P[best].y_pred.values
pv = []
for m in S.model:
    c = P[m].y_true.values == P[m].y_pred.values
    t = [[np.sum(cb & c), np.sum(cb & ~c)], [np.sum(~cb & c), np.sum(~cb & ~c)]]
    pv.append(np.nan if m == best else mcnemar(t, exact=False, correction=True).pvalue)
mask = ~np.isnan(pv); adj = np.full(len(pv), np.nan)
adj[mask] = multipletests(np.array(pv)[mask], method="holm")[1]
S["mcnemar_p_vs_best"] = pv; S["mcnemar_p_holm"] = adj; S["best_model"] = NICE[best]
S.to_csv(OUT / "summary.csv", index=False)
PC = pd.DataFrame(perclass); PC.to_csv(OUT / "per_class_pooled.csv", index=False)
G = pd.DataFrame(groups).set_index("model").loc[S.model].reset_index(); G.to_csv(OUT / "groups.csv", index=False)

# ---------- leakage (Round 3)
lk = []
for d in sorted(glob.glob(str(RUNS / "random" / "*" / "random"))):
    m = pd.read_csv(Path(d) / "metrics.csv").iloc[0]; s = S[S.model == m.model].iloc[0]
    lk.append({"model": m.model, "name": NICE[m.model], "random_acc": m.accuracy, "random_macro_f1": m.macro_f1,
               "object_disjoint_oof_acc": s.oof_acc, "object_disjoint_oof_macro_f1": s.oof_macro_f1,
               "gap_acc_pp": 100 * (m.accuracy - s.oof_acc), "gap_macro_f1_pp": 100 * (m.macro_f1 - s.oof_macro_f1)})
L = pd.DataFrame(lk); L.to_csv(OUT / "leakage.csv", index=False)

# ---------- master workbook
xl = ROOT / "results" / "master_results.xlsx"
try:
    with pd.ExcelWriter(xl, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        F.to_excel(w, "FoldResults", index=False); S.to_excel(w, "Summary", index=False)
        PC.to_excel(w, "PerClass", index=False); G.to_excel(w, "Groups", index=False); L.to_excel(w, "Leakage", index=False)
except PermissionError:
    print("master_results.xlsx is open in Excel - close it and run again to update the workbook (CSVs are written).")

pd.set_option("display.width", 250)
show = S[["name", "lr", "params_M", "macs_M", "acc_mean", "acc_sd", "macro_f1_mean", "macro_f1_sd", "oof_macro_f1",
          "oof_macro_f1_ci_low", "oof_macro_f1_ci_high", "mcnemar_p_holm"]].copy()
for c in ["acc_mean", "acc_sd", "macro_f1_mean", "macro_f1_sd", "oof_macro_f1", "oof_macro_f1_ci_low", "oof_macro_f1_ci_high"]:
    show[c] = (100 * show[c]).round(2)
print(show.round({"params_M": 2, "macs_M": 0, "mcnemar_p_holm": 4}).to_string(index=False))
print("\nLEAKAGE\n", L.round(4).to_string(index=False))
print("\nGROUPS\n", G.round(4).to_string(index=False))
