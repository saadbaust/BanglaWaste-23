"""Build the manuscript-only items and copy every figure/table into the manuscript folders. No GPU needed.

Run after waste_make_figures_tables.py, from the Claude folder:
    python code/paper/waste_make_manuscript_items.py

Makes   results/paper/figures/fig_split_protocol, fig_crossregion_shortcut (PDF + PNG)
        results/paper/tables/table_dataset_comparison (CSV) and journal-style LaTeX tables
Copies  main / supplementary figures to  manuscript/figures/{main,supp}
        and figures + tables to  manuscript/overleaf/{figures,tables}/{main,supp}
"""
import shutil
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RES, B3 = ROOT / "results", ROOT / "results" / "batch3"
FIG, TAB = RES / "paper" / "figures", RES / "paper" / "tables"
SH_ = ROOT / "manuscript"                      # figures/ lives here
OV = ROOT / "manuscript" / "overleaf"
CM = pd.read_csv(RES / "class_map.csv"); ITEM = dict(zip(CM.class_folder, CM.item_name))

INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_PRE, C_SCR, C_RAND, C_GREEN = "#2a78d6", "#eb6834", "#898781", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
NICE = {"convnext_tiny": "ConvNeXt-T", "mobilenetv3_large_100": "MobileNetV3-L", "efficientnet_b0": "EfficientNet-B0",
        "custom_v0": "Custom CNN (V0)"}
def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)

# ============================================================ Figure: split protocol
OBJ_COL = ["#2a78d6", "#1baf7a", "#eb6834", "#9b59b6", "#d4a017"]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 1.35]})
for ax in (axa, axb):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

# (a) random image-level split: images of each object fall into both train and test
axa.set_title("(a) Random image-level split", loc="left")
axa.text(0.1, 9.0, "Images of the same physical object", fontsize=7.5, color=INK2)
rng = np.random.default_rng(3)
for r, (name, col) in enumerate(zip(["Object 1", "Object 2", "Object 3"], OBJ_COL[:3])):
    y = 7.6 - r * 1.25
    axa.text(0.1, y + 0.25, name, fontsize=7.5, color=INK, va="center")
    test = set(rng.choice(7, 2, replace=False))
    for j in range(7):
        x = 2.0 + j * 0.62
        axa.add_patch(Rectangle((x, y), 0.5, 0.5, facecolor=col if j not in test else "white", edgecolor=col, lw=1.4,
                                hatch="////" if j in test else None))
# legend + result boxes
axa.add_patch(Rectangle((0.2, 2.75), 0.45, 0.45, facecolor=MUTED, edgecolor=MUTED)); axa.text(0.85, 2.97, "train image", fontsize=7, va="center")
axa.add_patch(Rectangle((3.6, 2.75), 0.45, 0.45, facecolor="white", edgecolor=MUTED, hatch="////")); axa.text(4.25, 2.97, "test image", fontsize=7, va="center")
axa.add_patch(FancyBboxPatch((0.15, 0.2), 9.6, 2.0, boxstyle="round,pad=0.05,rounding_size=0.2", facecolor="#fdece4", edgecolor=C_SCR, lw=1))
axa.text(0.4, 1.2, "Every object appears in both train and test\n(BanglaWaste-23: 155 of 172 objects) → the\nmodel is tested on objects it has already seen",
         fontsize=7.2, va="center", color=INK)

# (b) object-disjoint 5-fold CV
axb.set_title("(b) Object-disjoint 5-fold cross-validation (ours)", loc="left")
axb.text(0.1, 9.0, "Whole objects are assigned to folds; every class is in every fold", fontsize=7.5, color=INK2)
x0, w, gap = 1.7, 1.45, 0.17
for k in range(5):
    x = x0 + k * (w + gap); test = k == 0
    axb.add_patch(FancyBboxPatch((x, 3.35), w, 5.0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                 facecolor="#fdece4" if test else "#eaf2fc", edgecolor=C_SCR if test else C_PRE, lw=1.3 if test else 0.8))
    axb.text(x + w / 2, 7.95, f"Fold {k + 1}", ha="center", fontsize=7.5, color=INK, fontweight="bold")
    axb.text(x + w / 2, 3.6, "test" if test else "train", ha="center", fontsize=7.2, color=C_SCR if test else C_PRE)
for r, cls in enumerate(["Class A", "Class B", "Class C"]):
    y = 6.75 - r * 1.05
    axb.text(0.05, y + 0.2, cls, fontsize=7.2, color=INK, va="center")
    for k in range(5):
        x = x0 + k * (w + gap) + 0.25
        n = 1 if (k + r) % 3 else 2                               # some folds hold two objects of a class
        for s in range(n):
            col = OBJ_COL[(r * 2 + k + s) % 5]
            for d in range(3):                                    # a small stack = several images of one object
                axb.add_patch(Rectangle((x + s * 0.55 + d * 0.06, y + d * 0.06), 0.42, 0.42, facecolor=col, edgecolor="white", lw=0.6))
axb.add_patch(FancyBboxPatch((0.15, 0.2), 9.7, 2.6, boxstyle="round,pad=0.05,rounding_size=0.2", facecolor="#eaf2fc", edgecolor=C_PRE, lw=1))
axb.text(0.4, 1.5, "• Each object in exactly one fold; 5 rounds, each fold tested once\n"
                   "• Tuning (learning rate, epochs): one object per class held out\n   from the training folds; test folds never used for decisions\n"
                   "• Pooled out-of-fold predictions for all 4,160 images",
         fontsize=7.0, va="center", color=INK)
fig.tight_layout(w_pad=1.2); save(fig, "fig_split_protocol")

# ============================================================ Figure: cross-region + shortcut
BP = pd.read_csv(B3 / "bdwaste_per_class.csv"); BS = pd.read_csv(B3 / "bdwaste_summary.csv")
order_m = ["mobilenetv3_large_100", "convnext_tiny", "efficientnet_b0", "custom_v0"]
bp = BP.pivot(index="class", columns="model", values="acc_restricted")[order_m] * 100
bp = bp.loc[sorted(bp.index, key=lambda c: -bp.loc[c].mean())]
overall = BS.set_index("model").acc_restricted_to_shared.reindex(order_m) * 100
mat = np.vstack([bp.values, overall.values]); rows = [ITEM[c] for c in bp.index] + ["All 11 classes"]
SH = pd.read_csv(B3 / "shortcut_summary.csv")
fig = plt.figure(figsize=(7.2, 3.9)); gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.55)
ax = fig.add_subplot(gs[0]); im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
ax.axhline(len(bp) - 0.5, color="white", lw=3)
ax.set_xticks(range(4), [NICE[m] for m in order_m], rotation=30, ha="right", fontsize=7.5); ax.set_yticks(range(len(rows)), rows, fontsize=7.5)
ax.get_yticklabels()[-1].set_fontweight("bold")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]; ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 55 else INK,
                                fontweight="bold" if i == len(rows) - 1 else None)
ax.spines[:].set_visible(False); ax.tick_params(length=0)
ax.set_title("(a) Cross-region test on BDWaste (Dhaka)\naccuracy %, 11 shared classes", loc="left", fontsize=8.5)
ax2 = fig.add_subplot(gs[1]); conds = ["original", "jpeg_q30", "lowres_96"]; lab = ["Original", "JPEG\nquality 30", "96 px,\nre-enlarged"]
x = np.arange(3); w = 0.36
for k, (m, col) in enumerate(zip(["convnext_tiny", "efficientnet_b0"], [C_PRE, C_GREEN])):
    v = [SH[(SH.model == m) & (SH.condition == c)].macro_f1.iloc[0] * 100 for c in conds]
    ax2.bar(x + (k - 0.5) * (w + 0.03), v, w, color=col, label=NICE[m], zorder=2)
    for xi, vi in zip(x, v): ax2.text(xi + (k - 0.5) * (w + 0.03), vi + 0.25, f"{vi:.1f}", ha="center", fontsize=6.8)
ax2.set_xticks(x, lab, fontsize=7.5); ax2.set_ylim(90, 100.5); ax2.set_ylabel("Macro-F1 (%), mean of 5 folds")
ax2.grid(axis="y", color=GRID, lw=0.6); ax2.set_axisbelow(True)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.2)
ax2.set_title("(b) Shortcut test: degraded test images\n(object-disjoint folds)", loc="left", fontsize=8.5)
save(fig, "fig_crossregion_shortcut")

# ============================================================ Tables (journal-style LaTeX)
def latex_escape(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("α", "$\\alpha$").replace("≥", "$\\geq$").replace("≤", "$\\leq$").replace("°", "$^\\circ$").replace("×", "$\\times$")

def write_table(name, caption, label, colspec, header, body, note=None, size="\\small", star=False):
    env = "table*" if star else "table"
    L = [f"\\begin{{{env}}}[!htbp]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         "\\begin{adjustbox}{max width=\\textwidth}",
         f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"] + body + ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if note: L.append(f"\\par\\vspace{{2pt}}\\footnotesize {note}")
    L.append(f"\\end{{{env}}}")
    (TAB / f"{name}.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

# T1 dataset comparison (curated from literature/literature_review.xlsx, sheet Table A + Papers)
DC = pd.DataFrame([
    ("TrashNet", "yang2016trashnet", 2016, "2,527", "6", "Real, staged on white board", "No", "No", "Not stated"),
    ("TACO", "proenca2020taco", 2020, "1,500", "60", "Real, crowdsourced litter", "No", "No", "Random, image-level"),
    ("TrashBox", "kumsetty2022trashbox", 2022, "17,785", "7", "Web-scraped", "Yes (web)", "No", "Not stated"),
    ("ZeroWaste-f", "bashkirova2022zerowaste", 2022, "4,661", "4", "Real, recycling conveyor", "No", "No", "Not stated"),
    ("Zhou et al.", "zhou2022medical", 2022, "3,480", "8", "Real, hospital (China)", "Yes (all)", "No", "Random, image-level"),
    ("RealWaste", "single2023realwaste", 2023, "4,808", "9", "Real, landfill", "No", "No", "Random, image-level"),
    ("Medical Waste 4.0", "bruno2023medical", 2023, ">1,400", "7", "Real, new unused items", "Yes (all)", "No", "Different-day test set"),
    ("BDWaste", "rahman2024bdwaste", 2024, "2,497", "20", "Real, street (Dhaka)", "Street-collected", "No", "Random, image-level"),
    ("TrashNeXt", "tanvir2025trashnext", 2025, "23,625", "9", "Merged public sets", "Yes (merged)", "No", "Random, image-level"),
    ("Garbage Dataset", "kunwar2026garbage", 2026, "12,259", "10", "App, web and community", "No", "No", "Random, image-level"),
    ("BanglaWaste-23 (ours)", "", 2026, "4,160", "23", "Real, households and hospital", "Yes (4 classes, hospital)", "Yes (172 objects)", "Object-disjoint 5-fold"),
], columns=["Dataset", "key", "Year", "Images", "Classes", "Source", "Medical waste", "Object IDs", "Evaluation split"])
DC.drop(columns="key").to_csv(TAB / "table_dataset_comparison.csv", index=False)
body = []
for r in DC.itertuples():
    ds = f"{r.Dataset} \\citep{{{r.key}}}" if r.key else f"\\textbf{{{r.Dataset}}}"
    cells = [ds, str(r.Year), r.Images, r.Classes, r.Source, r._7, r._8, r._9]
    if not r.key: cells = [cells[0]] + [f"\\textbf{{{latex_escape(c)}}}" for c in cells[1:]]
    else: cells = [cells[0]] + [latex_escape(c) for c in cells[1:]]
    body.append(" & ".join(cells) + " \\\\")
write_table("tex_table_dataset_comparison", "Public waste image datasets compared with BanglaWaste-23 (evaluation split as used in the original paper).",
            "tab:datasets", "@{}lrrrllll@{}",
            "Dataset & Year & Images & Classes & Source & Medical waste & Object IDs & Evaluation split", body, size="\\scriptsize", star=True)

# T2 main results
T = pd.read_csv(TAB / "table_main_results.csv")
body = ["\\multicolumn{7}{@{}l}{\\textit{ImageNet-pretrained, fully fine-tuned (40 epochs)}} \\\\"]
for i, r in T.iterrows():
    if r.Training == "from scratch" and "\\textit{Trained from scratch" not in "".join(body):
        body += ["\\midrule", "\\multicolumn{7}{@{}l}{\\textit{Trained from scratch (150 epochs)}} \\\\"]
    name = latex_escape(r.Model).replace(" (scratch)", "")
    p = "--" if r["p (Holm) vs best"] == "—" else ("$<$0.001" if "<" in str(r["p (Holm) vs best"]) else f'{float(r["p (Holm) vs best"]):.2f}')
    pooled = r["Pooled macro-F1 % [95% CI]"]
    if i == 0: pooled = "\\textbf{" + pooled.split(" ")[0] + "} " + " ".join(pooled.split(" ")[1:])
    body.append(" & ".join([name, f"{r['Params (M)']:.2f}", f"{int(r['MACs (M)']):,}", f"{float(r.LR):g}",
                            r["Macro-F1 % (mean ± SD)"].replace("±", "$\\pm$"), pooled, p]) + " \\\\")
write_table("tex_table_main_results",
            "Classification of unseen objects (object-disjoint 5-fold cross-validation, 4,160 images). Macro-F1 is given as the mean $\\pm$ SD over folds "
            "and pooled over all out-of-fold predictions with a 95\\% confidence interval from an object-level bootstrap (2,000 resamples). "
            "$p$: McNemar test against the best model (ConvNeXt-T), Holm-corrected.",
            "tab:main", "@{}lrrlccc@{}",
            "Model & Params (M) & MACs (M) & LR & Macro-F1 (\\%), fold mean & Pooled macro-F1 (\\%) [95\\% CI] & $p$", body, size="\\footnotesize")

# T3 groups and medical
G = pd.read_csv(TAB / "table_groups_medical.csv").set_index("Model").reindex(T.Model).reset_index()
body = []
for _, r in G.iterrows():
    if r.Model in ("Custom CNN (V0)",): body.append("\\midrule")
    body.append(" & ".join([latex_escape(r.Model.replace(" (scratch)", " (from scratch)").replace("Custom CNN (V0)", "Custom CNN V0 (from scratch)")),
                            f"{r['3-group accuracy %']:.2f}", f"{r['Medical recall %']:.2f}", f"{r['Medical precision %']:.2f}",
                            str(int(r['Medical → non-medical'])), str(int(r['Non-medical → medical']))]) + " \\\\")
write_table("tex_table_groups_medical",
            "Accuracy at the level of the three waste groups (food and plant, household non-food, medical) and errors involving medical waste "
            "(pooled out-of-fold predictions; 781 medical images).",
            "tab:medical", "@{}lccccc@{}",
            "Model & Group acc. (\\%) & Medical recall (\\%) & Medical precision (\\%) & Medical $\\rightarrow$ other & Other $\\rightarrow$ medical",
            body, size="\\footnotesize")

# Supplementary tables: simple LaTeX from the existing CSVs
SUPP_T = [("table_training_settings", "Training and evaluation settings.", "tab:s_settings"),
          ("table_lr_tuning", "Learning-rate selection on the fold-0 validation objects: mean validation macro-F1 (\\%) over epochs 16--40.", "tab:s_lr"),
          ("table_leakage", "Random image-level split versus object-disjoint evaluation (pooled out-of-fold macro-F1, \\%).", "tab:s_leakage"),
          ("table_hardest_objects", "The 15 hardest objects (lowest accuracy averaged over the 11 pretrained models).", "tab:s_hardest"),
          ("table_shortcut", "Shortcut test: macro-F1 (\\%, mean of 5 folds) after strong JPEG compression and after resolution loss.", "tab:s_shortcut"),
          ("table_device_check", "Per-device check: accuracy difference (percentage points) between the two phones used for a class.", "tab:s_device"),
          ("table_bdwaste", "Cross-dataset test on BDWaste (Dhaka): models trained on all BanglaWaste-23 images.", "tab:s_bdwaste"),
          ("table_per_class_f1", "Per-class F1 (\\%) of every model (pooled out-of-fold).", "tab:s_perclass")]
for name, cap, lab in SUPP_T:
    d = pd.read_csv(TAB / f"{name}.csv")
    if name == "table_per_class_f1": d = d.drop(columns=["Class"])
    hdr = " & ".join(latex_escape(c).replace("±", "$\\pm$").replace("→", "$\\rightarrow$") for c in d.columns)
    def fmt(v):
        if isinstance(v, float): return "--" if np.isnan(v) else f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 1e5 else f"{v:g}"
        return latex_escape(v).replace("±", "$\\pm$").replace("→", "$\\rightarrow$").replace("—", "--")
    body = [" & ".join(fmt(v) for v in row) + " \\\\" for row in d.itertuples(index=False)]
    spec = "@{}" + ("p{3.2cm}p{11cm}" if name == "table_training_settings" else "l" + "r" * (len(d.columns) - 1)) + "@{}"
    size = "\\tiny" if name == "table_per_class_f1" else "\\scriptsize"
    write_table(f"tex_{name}", cap, lab, spec, hdr, body, size=size, star=name == "table_per_class_f1")

# ============================================================ Copy into manuscript folders
MAIN_FIG = {"fig_samples": ROOT / "manuscript/figures/dataset/Fig1", "fig_split_protocol": FIG / "fig_split_protocol",
            "fig_accuracy_vs_macs": FIG / "fig_accuracy_vs_macs", "fig_leakage": FIG / "fig_leakage",
            "fig_crossregion_shortcut": FIG / "fig_crossregion_shortcut"}
SUPP_FIG = {"fig_s_workflow": ROOT / "manuscript/figures/dataset/Fig3", "fig_s_class_distribution": ROOT / "manuscript/figures/dataset/Fig2",
            "fig_s_fold_composition": FIG / "fig_fold_composition", "fig_s_tuning_curves": FIG / "fig_supp_tuning_curves",
            "fig_s_per_class_f1": FIG / "fig_per_class_f1", "fig_s_confusion_best": FIG / "fig_confusion_best",
            "fig_s_confusion_custom_v0": FIG / "fig_confusion_custom_v0", "fig_s_bdwaste_per_class": FIG / "fig_bdwaste_per_class"}
GRADCAM = {f"fig_s_gradcam_{m}_{k}": FIG / f"fig_gradcam_{m}_{k}" for m in ["convnext_tiny", "efficientnet_b0", "custom_v0", "mobilenetv3_scratch"]
           for k in ["correct", "wrong"]}
MAIN_TAB = ["tex_table_dataset_comparison", "tex_table_main_results", "tex_table_groups_medical"]
SUPP_TAB = [f"tex_{n}" for n, _, _ in SUPP_T]

if not (ROOT / 'manuscript').exists():
    print('manuscript/ not found: figures and tables written to results/paper only'); raise SystemExit
for base in (SH_, OV):
    for sub in (("figures/main", "figures/supp") if base == SH_ else ("figures/main", "figures/supp", "tables/main", "tables/supp")): (base / sub).mkdir(parents=True, exist_ok=True)
for dst, src in MAIN_FIG.items():
    shutil.copy(f"{src}.pdf", OV / "figures/main" / f"{dst}.pdf")
    for ext in ("pdf", "png"): shutil.copy(f"{src}.{ext}", SH_ / "figures/main" / f"{dst}.{ext}")
for dst, src in SUPP_FIG.items():
    shutil.copy(f"{src}.pdf", OV / "figures/supp" / f"{dst}.pdf")
    for ext in ("pdf", "png"): shutil.copy(f"{src}.{ext}", SH_ / "figures/supp" / f"{dst}.{ext}")
for dst, src in GRADCAM.items():                  # photo-heavy: JPEG keeps Overleaf small
    im = Image.open(f"{src}.png").convert("RGB"); im.thumbnail((2400, 2400))
    for base in (SH_, OV): im.save(base / "figures/supp" / f"{dst}.jpg", quality=88)
for n in MAIN_TAB:
    for base in (OV,): shutil.copy(TAB / f"{n}.tex", base / "tables/main" / f"{n.replace('tex_', '')}.tex")
for n in SUPP_TAB:
    for base in (OV,): shutil.copy(TAB / f"{n}.tex", base / "tables/supp" / f"{n.replace('tex_', '')}.tex")
shutil.copy(ROOT / "literature" / "references.bib", OV / "references.bib")
print("main figures:", list(MAIN_FIG)); print("main tables:", MAIN_TAB)
print("supp figures:", len(SUPP_FIG) + len(GRADCAM), "supp tables:", len(SUPP_TAB))
