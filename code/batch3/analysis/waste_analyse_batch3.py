"""Batch 3 analysis: V0 from scratch (5 folds + random split), shortcut test, BDWaste cross-dataset.

Reads  runs/batch3/extracted/<notebook>/runs/...      and results/round2/ (pretrained OOF predictions, summary)
Writes results/batch3/
  v0_fold_results.csv, v0_summary.csv, oof_predictions_custom_v0.csv
  leakage_all.csv        random split vs object-disjoint for MobileNetV3-L, MobileViTv2, ConvNeXt-T and V0
  shortcut_summary.csv   mean over 5 folds per model x condition (+ per fold in shortcut_per_fold.csv)
  bdwaste_summary.csv, bdwaste_mapping.csv, bdwaste_per_class.csv
and updates results/master_results.xlsx (sheets Leakage, Shortcut, CrossDataset, V0).
Run: python code/batch3/analysis/waste_analyse_batch3.py   (from the Claude folder)
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parents[3]
E = ROOT / "runs" / "batch3" / "extracted"; OUT = ROOT / "results" / "batch3"; OUT.mkdir(parents=True, exist_ok=True)
R2 = ROOT / "results" / "round2"
CM = pd.read_csv(ROOT / "results" / "class_map.csv"); NAMES = CM.class_folder.tolist(); K = len(NAMES)
V0_FOLDS = [("waste_B3_acc2_nb1", 0), ("waste_B3_acc2_nb1", 1), ("waste_B3_acc2_nb2", 2), ("waste_B3_acc2_nb2", 3), ("waste_B3_acc3_nb1", 4)]

def macro_f1_fast(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K); tp = np.diag(cm)
    d = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp); return np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0))

# ---------- V0 5-fold
F = pd.concat([pd.read_csv(E / nb / "runs" / "cv" / "custom_v0_lr0.001" / f"fold{k}" / "metrics.csv") for nb, k in V0_FOLDS])
F.to_csv(OUT / "v0_fold_results.csv", index=False)
o = pd.concat([pd.read_csv(E / nb / "runs" / "cv" / "custom_v0_lr0.001" / f"fold{k}" / "predictions.csv").assign(fold=k)
               for nb, k in V0_FOLDS]).sort_values("file_name").reset_index(drop=True)
assert len(o) == 4160 and o.file_name.is_unique
o.to_csv(OUT / "oof_predictions_custom_v0.csv", index=False)
y, p = o.y_true.values, o.y_pred.values
uo, inv = np.unique(o.object_id.values, return_inverse=True); idx = [np.flatnonzero(inv == i) for i in range(len(uo))]
rng = np.random.default_rng(0); bs = []
for _ in range(2000):
    t = np.concatenate([idx[i] for i in rng.integers(0, len(uo), len(uo))]); bs.append(macro_f1_fast(y[t], p[t]))
best = pd.read_csv(R2 / "oof_predictions_convnext_tiny.csv").sort_values("file_name")
cb = best.y_true.values == best.y_pred.values; c = y == p
tab = [[np.sum(cb & c), np.sum(cb & ~c)], [np.sum(~cb & c), np.sum(~cb & ~c)]]
V = pd.DataFrame([{"model": "custom_v0", "name": "V0 (custom CNN, from scratch)", "lr": 1e-3, "epochs": 150,
                   "params_M": F.params.iloc[0] / 1e6, "macs_M": F.macs_M.iloc[0],
                   "acc_mean": F.accuracy.mean(), "acc_sd": F.accuracy.std(), "macro_f1_mean": F.macro_f1.mean(), "macro_f1_sd": F.macro_f1.std(),
                   "oof_acc": c.mean(), "oof_macro_f1": f1_score(y, p, average="macro"),
                   "oof_macro_f1_ci_low": np.percentile(bs, 2.5), "oof_macro_f1_ci_high": np.percentile(bs, 97.5),
                   "mcnemar_p_vs_convnext": mcnemar(tab, exact=False, correction=True).pvalue, "train_min_per_fold": F.minutes.mean()}])
V.to_csv(OUT / "v0_summary.csv", index=False)
pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(K), zero_division=0)
pd.DataFrame({"class": NAMES, "precision": pr, "recall": rc, "f1": f1, "support": sup}).to_csv(OUT / "v0_per_class_pooled.csv", index=False)
cmx = pd.crosstab(pd.Categorical(y, range(K)), pd.Categorical(p, range(K)), dropna=False); cmx.index = NAMES; cmx.columns = NAMES
cmx.to_csv(OUT / "v0_confusion_pooled.csv")

# ---------- leakage (random vs object-disjoint), pretrained from Round 2/3 + V0
L = pd.read_csv(R2 / "leakage.csv")
r = pd.read_csv(E / "waste_B3_acc3_nb1" / "runs" / "random" / "custom_v0_lr0.001" / "random" / "metrics.csv").iloc[0]
L = pd.concat([L, pd.DataFrame([{"model": "custom_v0", "name": "V0 (custom CNN, from scratch)", "random_acc": r.accuracy,
               "random_macro_f1": r.macro_f1, "object_disjoint_oof_acc": V.oof_acc[0], "object_disjoint_oof_macro_f1": V.oof_macro_f1[0],
               "gap_acc_pp": 100 * (r.accuracy - V.oof_acc[0]), "gap_macro_f1_pp": 100 * (r.macro_f1 - V.oof_macro_f1[0])}])], ignore_index=True)
L.to_csv(OUT / "leakage_all.csv", index=False)

# ---------- shortcut
rows = []
for f in glob.glob(str(E / "waste_B3_acc1_nb1" / "runs" / "shortcut" / "*" / "fold*" / "shortcut_summary.csv")):
    pth = Path(f); rows.append(pd.read_csv(f).assign(model=pth.parts[-3].split("_lr")[0], fold=int(pth.parts[-2][4:])))
S = pd.concat(rows); S.to_csv(OUT / "shortcut_per_fold.csv", index=False)
SS = S.groupby(["model", "condition"]).agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                                           pred_as_malta_not_malta=("pred_as_malta_not_malta", "sum"), malta_recall=("malta_recall", "mean")).reset_index()
SS.to_csv(OUT / "shortcut_summary.csv", index=False)

# ---------- BDWaste
bd = E / "waste_B3_acc4_nb1" / "runs"
B = pd.read_csv(bd / "bdwaste_summary.csv"); B.to_csv(OUT / "bdwaste_summary.csv", index=False)
pd.read_csv(bd / "bdwaste_mapping.csv").to_csv(OUT / "bdwaste_mapping.csv", index=False)
pc = []
for f in glob.glob(str(bd / "all_bdwaste" / "*" / "all" / "predictions.csv")):
    d = pd.read_csv(f); model = Path(f).parts[-3].split("_lr")[0]
    prob = d[[f"p_{x}" for x in NAMES]].values; yt = d.y_true.values; shared = sorted(set(yt))
    rp = np.array(shared)[prob[:, shared].argmax(1)]
    for c_ in shared:
        mk = yt == c_; pc.append({"model": model, "class": NAMES[c_], "images": int(mk.sum()),
                                  "acc_23way": float((d.y_pred.values[mk] == c_).mean()), "acc_restricted": float((rp[mk] == c_).mean())})
PC = pd.DataFrame(pc); PC.to_csv(OUT / "bdwaste_per_class.csv", index=False)

xl = ROOT / "results" / "master_results.xlsx"
try:
    with pd.ExcelWriter(xl, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        L.to_excel(w, sheet_name="Leakage", index=False); SS.to_excel(w, sheet_name="Shortcut", index=False)
        B.to_excel(w, sheet_name="CrossDataset", index=False); PC.to_excel(w, sheet_name="CrossDataset_perclass", index=False)
        V.to_excel(w, sheet_name="V0", index=False)
except PermissionError:
    print("Close master_results.xlsx in Excel and run again to update it (CSVs are written).")
pd.set_option("display.width", 220)
print(V.round(4).to_string(index=False)); print(L.round(4).to_string(index=False)); print(SS.round(4).to_string(index=False))
print(B.round(4).to_string(index=False)); print(PC.pivot(index="class", columns="model", values="acc_restricted").round(3).to_string())
