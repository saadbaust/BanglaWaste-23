"""Make every paper figure (PDF + PNG) and table (CSV + LaTeX) from the saved results. No GPU needed.

Reads  results/round2/*, results/batch3/*, results/reference/*, runs/round1_pilot/*, runs/batch3/extracted/*
Writes results/paper/figures/*.pdf|png   results/paper/tables/*.csv|tex  (then run waste_make_manuscript_items.py)
Includes MobileNetV3-L from scratch automatically once runs/batch3/extracted/waste_B3_acc4_nb2 exists.
Run:   python code/paper/waste_make_figures_tables.py   (from the Claude folder)
"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import f1_score, precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parents[2]
R2, B3, REF = ROOT / "results" / "round2", ROOT / "results" / "batch3", ROOT / "results" / "reference"
E3 = ROOT / "runs" / "batch3" / "extracted"
FIG, TAB = ROOT / "results" / "paper" / "figures", ROOT / "results" / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
CM = pd.read_csv(ROOT / "results" / "class_map.csv"); NAMES = CM.class_folder.tolist(); K = len(NAMES)
ITEM = dict(zip(CM.class_folder, CM.item_name)); GROUP = dict(zip(CM.class_folder, CM.group))

# ---------- style: see waste_figstyle.py (font, size); colours below
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_PRE, C_SCR, C_RAND, C_GREEN = "#2a78d6", "#eb6834", "#898781", "#1baf7a"          # pretrained, from scratch, random-split bars
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import waste_figstyle as FS                     # paper font, text width
FS.apply()
def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
def tex(df, name, caption, fmt=None):
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.tex").write_text(df.to_latex(index=False, escape=True, caption=caption, label=f"tab:{name}",
                                                 float_format=fmt or (lambda x: f"{x:.2f}")))

NICE = {"mobilenetv2_100": "MobileNetV2", "mobilenetv3_large_100": "MobileNetV3-L", "mobilenetv4_conv_small": "MobileNetV4-S",
        "efficientnet_b0": "EfficientNet-B0", "ghostnetv2_100": "GhostNetV2", "mobilevitv2_100": "MobileViTv2",
        "efficientformerv2_s0": "EfficientFormerV2-S0", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
        "convnext_tiny": "ConvNeXt-T", "deit_small_patch16_224": "DeiT-S", "custom_v0": "Custom CNN (V0)",
        "mobilenetv3_large_100_scratch": "MobileNetV3-L (scratch)"}

def macro_f1_fast(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K); tp = np.diag(cm)
    d = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp); return np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0))

# ======================================================================= data
S = pd.read_csv(R2 / "summary.csv"); S["training"] = "ImageNet-pretrained"
OOF = {m: pd.read_csv(R2 / f"oof_predictions_{m}.csv").sort_values("file_name").reset_index(drop=True) for m in S.model}
V = pd.read_csv(B3 / "v0_summary.csv"); V["training"] = "from scratch"
OOF["custom_v0"] = pd.read_csv(B3 / "oof_predictions_custom_v0.csv").sort_values("file_name").reset_index(drop=True)
S = pd.concat([S, V], ignore_index=True)
LEAK = pd.read_csv(B3 / "leakage_all.csv")

# MobileNetV3-L from scratch (optional, when its zip has been unpacked)
SCR = E3 / "waste_B3_acc4_nb2" / "runs"
if SCR.exists() and len(glob.glob(str(SCR / "cv" / "*" / "fold*" / "predictions.csv"))) == 5:
    d = SCR / "cv" / "mobilenetv3_large_100_scratch_lr0.001"
    F = pd.concat([pd.read_csv(d / f"fold{k}" / "metrics.csv") for k in range(5)])
    o = pd.concat([pd.read_csv(d / f"fold{k}" / "predictions.csv") for k in range(5)]).sort_values("file_name").reset_index(drop=True)
    o.to_csv(B3 / "oof_predictions_mobilenetv3_large_100_scratch.csv", index=False)
    OOF["mobilenetv3_large_100_scratch"] = o
    y, p = o.y_true.values, o.y_pred.values
    uo, inv = np.unique(o.object_id.values, return_inverse=True); idx = [np.flatnonzero(inv == i) for i in range(len(uo))]
    rng = np.random.default_rng(0); bs = [macro_f1_fast(y[t], p[t]) for t in
          (np.concatenate([idx[i] for i in rng.integers(0, len(uo), len(uo))]) for _ in range(2000))]
    S = pd.concat([S, pd.DataFrame([{"model": "mobilenetv3_large_100_scratch", "name": NICE["mobilenetv3_large_100_scratch"],
        "lr": 1e-3, "epochs": 150, "params_M": F.params.iloc[0] / 1e6, "macs_M": F.macs_M.iloc[0], "acc_mean": F.accuracy.mean(),
        "acc_sd": F.accuracy.std(), "macro_f1_mean": F.macro_f1.mean(), "macro_f1_sd": F.macro_f1.std(), "oof_acc": (y == p).mean(),
        "oof_macro_f1": f1_score(y, p, average="macro"), "oof_macro_f1_ci_low": np.percentile(bs, 2.5),
        "oof_macro_f1_ci_high": np.percentile(bs, 97.5), "train_min_per_fold": F.minutes.mean(),
        "gpu_infer_img_per_s": F.gpu_infer_img_per_s.mean(), "training": "from scratch"}])], ignore_index=True)
    r = pd.read_csv(SCR / "random" / "mobilenetv3_large_100_scratch_lr0.001" / "random" / "metrics.csv").iloc[0]
    LEAK = pd.concat([LEAK, pd.DataFrame([{"model": "mobilenetv3_large_100_scratch", "name": NICE["mobilenetv3_large_100_scratch"],
        "random_acc": r.accuracy, "random_macro_f1": r.macro_f1, "object_disjoint_oof_acc": (y == p).mean(),
        "object_disjoint_oof_macro_f1": f1_score(y, p, average="macro"), "gap_acc_pp": 100 * (r.accuracy - (y == p).mean()),
        "gap_macro_f1_pp": 100 * (r.macro_f1 - f1_score(y, p, average="macro"))}])], ignore_index=True)
    print("included MobileNetV3-L from scratch")
S["name"] = S.model.map(NICE)
# GPU throughput for V0 from its fold metrics
if S.gpu_infer_img_per_s.isna().any():
    v0f = pd.read_csv(B3 / "v0_fold_results.csv"); S.loc[S.model == "custom_v0", "gpu_infer_img_per_s"] = v0f.gpu_infer_img_per_s.mean()
# McNemar vs best (ConvNeXt-T) for every model, Holm over all comparisons
from statsmodels.stats.multitest import multipletests
best = S.sort_values("oof_macro_f1", ascending=False).model.iloc[0]; cb = OOF[best].y_true.values == OOF[best].y_pred.values
pv = {}
for m in S.model:
    if m == best: continue
    c = OOF[m].y_true.values == OOF[m].y_pred.values
    pv[m] = mcnemar([[np.sum(cb & c), np.sum(cb & ~c)], [np.sum(~cb & c), np.sum(~cb & ~c)]], exact=False, correction=True).pvalue
adj = dict(zip(pv, multipletests(list(pv.values()), method="holm")[1]))
S["p_holm_vs_best"] = S.model.map(adj)
S["_scr"] = S.training.eq("from scratch"); S = S.sort_values(["_scr", "oof_macro_f1"], ascending=[True, False]).drop(columns="_scr").reset_index(drop=True)

# ======================================================================= TABLES
T = pd.DataFrame({
    "Model": S.name, "Training": S.training, "Params (M)": S.params_M.round(2), "MACs (M)": S.macs_M.round(0).astype(int),
    "LR": S.lr.map(lambda x: f"{x:g}"), "Epochs": S.epochs.astype(int),
    "Accuracy % (mean ± SD)": [f"{a*100:.2f} ± {b*100:.2f}" for a, b in zip(S.acc_mean, S.acc_sd)],
    "Macro-F1 % (mean ± SD)": [f"{a*100:.2f} ± {b*100:.2f}" for a, b in zip(S.macro_f1_mean, S.macro_f1_sd)],
    "Pooled macro-F1 % [95% CI]": [f"{a*100:.2f} [{l*100:.1f}, {h*100:.1f}]" for a, l, h in zip(S.oof_macro_f1, S.oof_macro_f1_ci_low, S.oof_macro_f1_ci_high)],
    "p (Holm) vs best": S.p_holm_vs_best.map(lambda x: "—" if pd.isna(x) else ("<0.001" if x < 0.001 else f"{x:.3f}"))})
tex(T, "table_main_results", "Object-disjoint 5-fold results on BanglaWaste-23. Pooled macro-F1 over all 4,160 out-of-fold predictions; 95% CI by object-level bootstrap; McNemar test vs the best model with Holm correction.")
S.to_csv(TAB / "main_results_full.csv", index=False)

L = LEAK.copy(); L["name"] = L.model.map(NICE)
L["training"] = np.where(L.model.isin(["custom_v0", "mobilenetv3_large_100_scratch"]), "from scratch", "ImageNet-pretrained")
tex(pd.DataFrame({"Model": L.name, "Training": L.training, "Random split macro-F1 %": (L.random_macro_f1 * 100).round(2),
                  "Object-disjoint macro-F1 %": (L.object_disjoint_oof_macro_f1 * 100).round(2), "Gap (pp)": L.gap_macro_f1_pp.round(2)}),
    "table_leakage", "Random image-level split vs object-disjoint evaluation (pooled out-of-fold).")

G = pd.read_csv(R2 / "groups.csv")
gv = []
for m in [x for x in ["custom_v0", "mobilenetv3_large_100_scratch"] if x in OOF]:
    o = OOF[m]; gy = o.y_true.map(lambda i: GROUP[NAMES[i]]).values; gp = o.y_pred.map(lambda i: GROUP[NAMES[i]]).values
    mt, mp = gy == "Medical", gp == "Medical"
    gv.append({"model": m, "group_accuracy": (gy == gp).mean(), "medical_recall": (mt & mp).sum() / mt.sum(),
               "medical_precision": (mt & mp).sum() / max(mp.sum(), 1), "medical_to_nonmedical_errors": int((mt & ~mp).sum()),
               "nonmedical_to_medical_errors": int((~mt & mp).sum())})
G = pd.concat([G, pd.DataFrame(gv)], ignore_index=True); G["name"] = G.model.map(NICE)
tex(pd.DataFrame({"Model": G.name, "3-group accuracy %": (G.group_accuracy * 100).round(2), "Medical recall %": (G.medical_recall * 100).round(2),
                  "Medical precision %": (G.medical_precision * 100).round(2), "Medical → non-medical": G.medical_to_nonmedical_errors,
                  "Non-medical → medical": G.nonmedical_to_medical_errors}), "table_groups_medical",
    "Accuracy at the level of the three waste groups and errors involving the medical group (pooled out-of-fold, 4,160 images, 781 medical).")

SH = pd.read_csv(B3 / "shortcut_summary.csv")
SH["name"] = SH.model.map(NICE)
SHp = SH.pivot(index="name", columns="condition", values="macro_f1")[["original", "jpeg_q30", "lowres_96"]] * 100
SHp = SHp.reset_index().rename(columns={"name": "Model", "original": "Original %", "jpeg_q30": "JPEG q30 %", "lowres_96": "96-px low-res %"})
SHp["Images re-routed to malta"] = SH.groupby("name").pred_as_malta_not_malta.sum().reindex(SHp.Model).values
tex(SHp.round(2), "table_shortcut", "Shortcut test: macro-F1 (mean of 5 folds) on the same test images after strong JPEG compression and after resolution loss.")

BD = pd.read_csv(B3 / "bdwaste_summary.csv"); BD["name"] = BD.model.map(NICE)
tex(pd.DataFrame({"Model": BD.name, "Images": BD.images, "Accuracy, 23-way %": (BD.acc_23way * 100).round(2),
                  "Accuracy, 11 shared classes %": (BD.acc_restricted_to_shared * 100).round(2),
                  "Macro-F1, 11 shared classes %": (BD.macro_f1_restricted * 100).round(2)}).sort_values("Accuracy, 11 shared classes %", ascending=False),
    "table_bdwaste", "Cross-dataset test: models trained on all BanglaWaste-23 images, tested on the BDWaste images of the 11 shared classes.")

R1 = pd.read_csv(ROOT / "runs" / "round1_pilot" / "round1_lr_table.csv"); ch = pd.read_csv(ROOT / "runs" / "round1_pilot" / "round1_choice.csv")
R1p = (R1.pivot(index="model", columns="lr", values="score_ep16_40") * 100).round(2)
R1p.columns = [f"LR {c:g}" for c in R1p.columns]; R1p = R1p.reset_index(); R1p["Chosen LR"] = R1p.model.map(dict(zip(ch.model, ch.lr))).map(lambda x: f"{x:g}")
R1p.insert(0, "Model", R1p.model.map(NICE)); R1p = R1p.drop(columns="model")
tex(R1p, "table_lr_tuning", "Learning-rate selection on the fold-1 validation objects: mean validation macro-F1 (%) over epochs 16–40.")

PC = []
for m, o in OOF.items():
    pr, rc, f1, sup = precision_recall_fscore_support(o.y_true, o.y_pred, labels=range(K), zero_division=0)
    PC.append(pd.Series(f1 * 100, index=NAMES, name=NICE[m]))
PCm = pd.DataFrame(PC).T.round(1); PCm.insert(0, "Item", [ITEM[c] for c in PCm.index]); PCm.insert(1, "Group", [GROUP[c] for c in PCm.index])
order_models = [NICE[m] for m in S.model]
PCm = PCm[["Item", "Group"] + order_models]
tex(PCm.reset_index().rename(columns={"index": "Class"}), "table_per_class_f1", "Per-class F1 (%) of every model (pooled out-of-fold).", fmt=lambda x: f"{x:.1f}")

# hardest objects (pretrained models only)
pre = [m for m in S.model if S.set_index("model").training[m] == "ImageNet-pretrained"]
acc = pd.concat([OOF[m].assign(ok=OOF[m].y_true == OOF[m].y_pred).groupby("object_id").ok.mean().rename(NICE[m]) for m in pre], axis=1)
n = OOF[pre[0]].groupby("object_id").size().rename("Images")
fold = (OOF[pre[0]].groupby("object_id").fold.first() + 1).rename("Fold")      # paper numbers folds 1-5
H = pd.concat([n, fold, (acc.mean(1) * 100).rename("Mean accuracy % (11 pretrained)"), (acc.min(1) * 100).rename("Worst model %"),
               (acc[NICE[best]] * 100).rename(f"{NICE[best]} %")], axis=1).sort_values("Mean accuracy % (11 pretrained)").head(15)
tex(H.reset_index().rename(columns={"object_id": "Object"}).round(1), "table_hardest_objects", "The 15 hardest objects (lowest accuracy averaged over the 11 pretrained models).", fmt=lambda x: f"{x:.1f}")

# per-device check (classes photographed with two phones)
meta = pd.read_csv(REF / "folds.csv")[["file_name", "device"]]
two = meta.merge(pd.read_csv(REF / "folds.csv")[["file_name", "class_folder"]]).groupby("class_folder").device.nunique()
two = two[two > 1].index.tolist(); DV = []
for m in [best, "efficientnet_b0", "mobilenetv3_large_100", "custom_v0"]:
    o = OOF[m].merge(meta, on="file_name")
    for c in two:
        for dev, g in o[o.class_folder == c].groupby("device"):
            DV.append({"model": NICE[m], "class": c, "device": dev, "images": len(g), "accuracy": (g.y_true == g.y_pred).mean()})
DV = pd.DataFrame(DV); DV.to_csv(TAB / "device_check_long.csv", index=False)
DVs = DV.groupby(["model", "class"]).accuracy.agg(lambda s: 100 * (s.max() - s.min())).rename("Accuracy difference between phones (pp)").reset_index()
tex(DVs.pivot(index="class", columns="model", values="Accuracy difference between phones (pp)").round(1).reset_index().rename(columns={"class": "Class"}),
    "table_device_check", "Per-device check: absolute accuracy difference (percentage points) between the two phones used for a class.", fmt=lambda x: f"{x:.1f}")

pd.DataFrame([
    ("Input", "224 × 224 RGB (shorter side resized to 256 px with EXIF orientation applied; random-resized crop for training, centre crop for testing)"),
    ("Augmentation", "random resized crop (scale 0.6–1.0), horizontal and vertical flip, rotation ±20°, brightness/contrast jitter 0.15"),
    ("Mixing", "MixUp α = 0.2 / CutMix α = 1.0, applied with probability 0.5"),
    ("Optimiser", "AdamW, weight decay 1e-4, batch size 32, mixed precision"),
    ("Schedule", "5 warm-up epochs, then cosine decay to 0"),
    ("Learning rate", "per pretrained model from {1e-3, 3e-4, 1e-4} on fold-1 validation objects; 1e-3 for models trained from scratch"),
    ("Epochs", "40 (ImageNet-pretrained), 150 (from scratch); no early stopping"),
    ("Weights evaluated", "exponential moving average (decay 0.999) of the final epoch; no test-time augmentation"),
    ("Pretraining", "ImageNet-1k weights from timm for all pretrained models"),
    ("Evaluation", "object-disjoint 5-fold cross-validation (172 objects; every class in every fold); seed 42 + fold"),
    ("Hardware", "NVIDIA Tesla T4 (Kaggle), PyTorch + timm"),
], columns=["Setting", "Value"]).pipe(lambda d: tex(d, "table_training_settings", "Training and evaluation settings."))

# ======================================================================= FIGURES
# Style: code/paper/waste_figstyle.py (paper font, text width, no titles inside figures). Colours as before;
# the leakage bar chart also uses a hatch pattern so its two series stay apart in black-and-white print.
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
W = FS.TEXTWIDTH

# F1: accuracy vs compute (numbered points + key, so labels never collide). Pretrained = blue, scratch = orange.
fig, ax = plt.subplots(figsize=(W * 0.66, 3.1)); SS_ = S.sort_values("macs_M").reset_index(drop=True); key = []
for i, r in SS_.iterrows():
    pre = r.training == "ImageNet-pretrained"
    ax.errorbar(r.macs_M, r.oof_macro_f1 * 100, yerr=[[100 * (r.oof_macro_f1 - r.oof_macro_f1_ci_low)], [100 * (r.oof_macro_f1_ci_high - r.oof_macro_f1)]],
                fmt="o", ms=5.5, color=C_PRE if pre else C_SCR, ecolor=C_PRE if pre else C_SCR, elinewidth=0.9, capsize=1.8, mec="white", mew=0.8, zorder=3)
    off = {"ghostnetv2_100": (-3, 6), "mobilenetv4_conv_small": (2, -11), "mobilenetv3_large_100": (4, 3), "efficientnet_b0": (-4, 6),
           "efficientformerv2_s0": (4, -9), "densenet121": (4, -10), "resnet50": (-11, 4), "convnext_tiny": (-2, 6),
           "deit_small_patch16_224": (4, -10)}.get(r.model, (4, 3))
    ax.annotate(str(i + 1), (r.macs_M, r.oof_macro_f1 * 100), xytext=off, textcoords="offset points", fontsize=7.5, color=FS.INK)
    key.append(f"{i + 1}  {r['name'].replace('Custom CNN (V0)', 'Custom CNN V0')}")
ax.set_xscale("log"); ax.set_xlabel("MACs per image (millions, log scale)"); ax.set_ylabel("Pooled macro-F1 (%)")
ax.grid(axis="y", color=FS.GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_PRE, ms=5.5, label="ImageNet-pretrained"),
                   Line2D([], [], marker="o", ls="", color=C_SCR, ms=5.5, label="Trained from scratch")], loc="center right")
fig.text(0.985, 0.5, "\n".join(key), fontsize=7.5, va="center", ha="left", linespacing=1.35)
save(fig, "fig_accuracy_vs_macs")

# F2: leakage. Two series only: random split = grey with hatch pattern, unseen objects = solid blue; groups marked on the axis.
L["_scr"] = L.training.eq("from scratch"); L = L.sort_values(["_scr", "object_disjoint_oof_macro_f1"], ascending=[True, False]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(W, 2.9)); x = np.arange(len(L)); w = 0.36
ax.bar(x - w / 2 - 0.01, L.random_macro_f1 * 100, w, facecolor="#d5d4ce", edgecolor=FS.INK2, hatch="////", lw=0.6, zorder=2)
ax.bar(x + w / 2 + 0.01, L.object_disjoint_oof_macro_f1 * 100, w, facecolor=C_PRE, edgecolor=FS.INK2, lw=0.6, zorder=2)
for i, r in enumerate(L.itertuples()):
    ax.text(i + w / 2 + 0.01, r.object_disjoint_oof_macro_f1 * 100 + 1.2, f"−{r.gap_macro_f1_pp:.1f}", ha="center", fontsize=7.5)
ax.set_xticks(x, [n.replace(" (scratch)", "").replace("Custom CNN (V0)", "Custom\nCNN V0").replace("MobileNetV3-L", "MobileNet-\nV3-L")
                 .replace("ConvNeXt-T", "ConvNeXt-T\n").replace("MobileViTv2", "MobileViTv2\n") for n in L.name], fontsize=8)
ns = int((~L._scr).sum()); ax.axvline(ns - 0.5, color=FS.MID, lw=0.7, ls=(0, (3, 2)))
for x0, x1, t in ((0, ns - 1, "ImageNet-pretrained"), (ns, len(L) - 1, "Trained from scratch")):
    ax.text((x0 + x1) / 2, 109, t, ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, 116); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("Macro-F1 (%)")
ax.grid(axis="y", color=FS.GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(handles=[Patch(facecolor="#d5d4ce", edgecolor=FS.INK2, hatch="////", label="Random image split"),
                   Patch(facecolor=C_PRE, edgecolor=FS.INK2, label="Unseen objects")],
          loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
save(fig, "fig_leakage")

# F3: confusion matrices (row-normalised) best model and V0
def cm_plot(m, name):
    o = OOF[m]; cm = pd.crosstab(pd.Categorical(o.y_true, range(K)), pd.Categorical(o.y_pred, range(K)), dropna=False).values.astype(float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(W, W * 0.92)); im = ax.imshow(cmn * 100, cmap=SEQ, vmin=0, vmax=100)
    lab = [ITEM[c] for c in NAMES]; ax.set_xticks(range(K), lab, rotation=90, fontsize=6.5); ax.set_yticks(range(K), lab, fontsize=6.5)
    for i in range(K):
        for j in range(K):
            if cmn[i, j] >= 0.02:
                ax.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center", fontsize=5, color="white" if cmn[i, j] > 0.55 else FS.INK)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.spines[:].set_visible(False); ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Share of the true class (%)"); cb.outline.set_visible(False)
    save(fig, name)
cm_plot(best, "fig_confusion_best"); cm_plot("custom_v0", "fig_confusion_custom_v0")

# F4: per-class F1 heatmap
hm = PCm[order_models]
fig, ax = plt.subplots(figsize=(W, 5.4)); im = ax.imshow(hm.values, cmap=SEQ, vmin=50, vmax=100, aspect="auto")
ax.set_xticks(range(len(order_models)), [n.replace("Custom CNN (V0)", "Custom CNN V0") for n in order_models], rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(K), [f"{ITEM[c]} ({GROUP[c][0]})" for c in hm.index], fontsize=7)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]; ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5, color="white" if v > 85 else FS.INK)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("F1 (%); values below 50 use the lightest shade"); cb.outline.set_visible(False)
ax.spines[:].set_visible(False); ax.tick_params(length=0)
save(fig, "fig_per_class_f1")

# F5: shortcut (stand-alone version; the paper uses the combined Fig. 5)
conds = ["original", "jpeg_q30", "lowres_96"]; lab = ["Original", "JPEG quality 30", "96 px, re-enlarged"]
fig, ax = plt.subplots(figsize=(W * 0.6, 2.6)); models = SH.name.unique().tolist(); x = np.arange(len(conds)); w = 0.36
for k, (mname, fc_) in enumerate(zip(models, [C_PRE, C_GREEN])):
    v = [SH[(SH.name == mname) & (SH.condition == c)].macro_f1.iloc[0] * 100 for c in conds]
    ax.bar(x + (k - 0.5) * (w + 0.02), v, w, color=fc_, label=mname, zorder=2)
    for xi, vi in zip(x, v): ax.text(xi + (k - 0.5) * (w + 0.02), vi + 0.25, f"{vi:.1f}", ha="center", fontsize=7)
ax.set_xticks(x, lab); ax.set_ylim(90, 100.5); ax.set_ylabel("Macro-F1 (%), mean of 5 folds")
ax.grid(axis="y", color=FS.GRID, lw=0.5); ax.set_axisbelow(True); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
save(fig, "fig_shortcut")

# F6: BDWaste per-class (stand-alone version)
BP = pd.read_csv(B3 / "bdwaste_per_class.csv"); BP["name"] = BP.model.map(NICE)
bp = (BP.pivot(index="class", columns="name", values="acc_restricted") * 100)
bp = bp[[c for c in ["ConvNeXt-T", "MobileNetV3-L", "EfficientNet-B0", "Custom CNN (V0)"] if c in bp.columns]]
fig, ax = plt.subplots(figsize=(W * 0.6, 3.6)); im = ax.imshow(bp.values, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(bp.shape[1]), [c.replace("Custom CNN (V0)", "Custom CNN V0") for c in bp.columns], rotation=30, ha="right")
ax.set_yticks(range(bp.shape[0]), [ITEM[c] for c in bp.index])
for i in range(bp.shape[0]):
    for j in range(bp.shape[1]):
        v = bp.values[i, j]; ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 55 else FS.INK)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("Accuracy on BDWaste (%)"); cb.outline.set_visible(False)
ax.spines[:].set_visible(False); ax.tick_params(length=0); save(fig, "fig_bdwaste_per_class")

# F7: fold composition (used at 0.6 x text width)
fc = pd.read_csv(REF / "fold_counts.csv", index_col=0).drop(index="total", errors="ignore").drop(columns="total", errors="ignore")
fc = fc.loc[sorted(fc.index, key=lambda c: ITEM[c].lower())]
fig, ax = plt.subplots(figsize=(W * 0.6, 4.4)); im = ax.imshow(fc.values, cmap=SEQ, aspect="auto")
ax.set_xticks(range(5), [f"Fold {i + 1}" for i in range(5)], fontsize=7.5); ax.set_yticks(range(len(fc)), [ITEM[c] for c in fc.index], fontsize=7.5)
for i in range(fc.shape[0]):
    for j in range(5):
        v = fc.values[i, j]; ax.text(j, i, f"{v}", ha="center", va="center", fontsize=6.5, color="white" if v > 50 else FS.INK)
ax.spines[:].set_visible(False); ax.tick_params(length=0); save(fig, "fig_fold_composition")

# F8: Round 1 validation curves (supplementary). Learning rates in blue / orange / green, as before.
ms = sorted(R1.model.unique(), key=lambda m: NICE[m]); fig, axes = plt.subplots(3, 4, figsize=(W, 4.3), sharex=True, sharey=True)
sty = {1e-3: dict(color=C_PRE), 3e-4: dict(color=C_SCR), 1e-4: dict(color=C_GREEN)}
for ax, m in zip(axes.ravel(), ms):
    for lr, st in sty.items():
        f = glob.glob(str(ROOT / "runs" / "round1_pilot" / "extracted" / "runs" / "fold0_val" / f"{m}_lr{lr:g}" / "fold0val" / "history.csv"))
        if f:
            h = pd.read_csv(f[0]).dropna(subset=["val_macro_f1_ema"]); ax.plot(h.epoch, h.val_macro_f1_ema * 100, lw=1.2, label=f"LR {lr:g}", **st)
    ax.axvline(40, color=FS.MID, lw=0.7, ls="--"); ax.set_title(NICE[m], fontsize=7.5, pad=2)
    ax.grid(axis="y", color=FS.GRID, lw=0.4); ax.set_ylim(60, 100); ax.tick_params(labelsize=7)
for ax in axes.ravel()[len(ms):]: ax.axis("off")
fig.legend(handles=[Line2D([], [], lw=1.2, label=f"LR {lr:g}", **st) for lr, st in sty.items()], loc="lower right",
           bbox_to_anchor=(0.97, 0.08), fontsize=8)
fig.supxlabel("Epoch", fontsize=9); fig.supylabel("Validation macro-F1 (%)", fontsize=9); fig.tight_layout(h_pad=0.4, w_pad=0.4)
save(fig, "fig_supp_tuning_curves")

# F9: Grad-CAM panels (no title; the caption names model, fold and kind). Tile labels: class (correct) or true/predicted (errors).
from PIL import Image as PILImage
GC = {"convnext_tiny": B3 / "gradcam" / "convnext_tiny_fold0_items.csv", "efficientnet_b0": B3 / "gradcam" / "efficientnet_b0_fold0_items.csv",
      "custom_v0": B3 / "gradcam" / "custom_v0_fold0_items.csv",
      "mobilenetv3_large_100_scratch": B3 / "gradcam" / "mobilenetv3_scratch" / "gradcam_items.csv"}
NPZ = {"convnext_tiny": glob.glob(str(E3 / "waste_B3_acc1_nb1" / "runs" / "gradcam" / "convnext_tiny_lr*" / "fold0" / "gradcam_data.npz")),
       "efficientnet_b0": glob.glob(str(E3 / "waste_B3_acc1_nb1" / "runs" / "gradcam" / "efficientnet_b0_lr*" / "fold0" / "gradcam_data.npz")),
       "custom_v0": glob.glob(str(E3 / "waste_B3_acc2_nb1" / "runs" / "cv" / "custom_v0_lr0.001" / "fold0" / "gradcam" / "gradcam_data.npz")),
       "mobilenetv3_large_100_scratch": glob.glob(str(SCR / "cv" / "mobilenetv3_large_100_scratch_lr0.001" / "fold0" / "gradcam" / "gradcam_data.npz"))}
def heat(cam, size):
    c = np.asarray(cam, dtype=np.float32)
    return np.asarray(PILImage.fromarray(c).resize((size[1], size[0]), PILImage.BILINEAR))
for m in GC:
    if not NPZ[m]: continue
    d = np.load(NPZ[m][0], allow_pickle=True); items = pd.read_csv(GC[m]); imgs, cams = d["images"], d["cams"]
    for kind, cols_ in (("correct", 6), ("wrong", 4)):
        sel = np.flatnonzero(items.kind.values == kind)
        if not len(sel): continue
        cols_ = min(cols_, len(sel)); tile = W / 6; nrow = int(np.ceil(len(sel) / cols_))
        fig, axes = plt.subplots(nrow * 2, cols_, figsize=(tile * cols_, (2 * tile + 0.32) * nrow))
        axes = np.array(axes).reshape(nrow * 2, cols_)
        for a in axes.ravel(): a.axis("off")
        for j, i in enumerate(sel):
            rr, cc = 2 * (j // cols_), j % cols_; r = items.iloc[i]
            axes[rr, cc].imshow(imgs[i]); axes[rr + 1, cc].imshow(imgs[i]); axes[rr + 1, cc].imshow(heat(cams[i], imgs[i].shape[:2]), cmap="jet", alpha=0.45)
            t = ITEM[r.true] if kind == "correct" else f"True: {ITEM[r.true]}\nPred.: {ITEM[r.pred]}"
            axes[rr, cc].set_title(t, fontsize=6.5, pad=2)
        fig.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.2); save(fig, f"fig_gradcam_{'mobilenetv3_scratch' if m.endswith('_scratch') else m}_{kind}")

print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
print("tables:", sorted(p.name for p in TAB.glob("*.csv")))
print(T.to_string(index=False))
