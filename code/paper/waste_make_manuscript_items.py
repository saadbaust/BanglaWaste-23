"""Build the manuscript-only items and copy every figure/table into the manuscript folders. No GPU needed.

Run after waste_make_figures_tables.py, from the Claude folder:
    python code/paper/waste_make_manuscript_items.py

Makes   results/paper/figures/fig_samples (needs the raw images), fig_split_protocol, fig_crossregion_shortcut,
        fig_s_workflow, fig_s_class_distribution (PDF + PNG)
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

C_PRE, C_SCR, C_GREEN = "#2a78d6", "#eb6834", "#1baf7a"     # blue, orange, green (colours as before)
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import waste_figstyle as FS                     # paper font, text width, (a)/(b) under panels
FS.apply(); W = FS.TEXTWIDTH
from matplotlib.patches import Patch
NICE = {"convnext_tiny": "ConvNeXt-T", "mobilenetv3_large_100": "MobileNetV3-L", "efficientnet_b0": "EfficientNet-B0",
        "custom_v0": "Custom CNN (V0)"}
def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)

# ============================================================ Figure 1: one image per class (sub-figures (a)-(w), item names)
# Order: food and plant (a-n), household non-food (o-s), medical (t-w); alphabetical by item within each group.
import os
SAMPLE = {"Banana_peeling": "Banana_peeling_img_00034.jpg", "Bottle": "Bottle_img_00158.jpg", "cloth": "cloth_img_00067.jpg",
          "cotton": "cotton_img_00139.jpg", "cucumber_peeling": "cucumber_peeling_img_00060.jpg", "Egg": "Egg_img_00152.jpg",
          "Injection": "Injection_img_00122.jpg", "Lemon": "Lemon_img_00102.jpg", "malta_peeling": "malta_peeling_img_00133.jpg",
          "mango_leaf": "mango_leaf_img_00112.jpg", "mango_peeling": "mango_peeling_img_00147.jpg", "medicine": "medicine_img_00087.jpg",
          "nut": "nut_img_00139.jpg", "orange_peeling": "orange_peeling_img_00147.jpg", "packet": "packet_img_00124.jpg",
          "pen": "pen_img_00018.jpg", "pointed_gourd_peeling": "pointed_gourd_peeling_img_00110.jpg", "polythene": "polythene_img_00151.jpg",
          "potato_peeling": "potato_peeling_img_00010.jpg", "pumpkin": "pumpkin_img_00009.jpg", "saline": "saline_img_00068.jpg",
          "sugarcane": "sugarcane_img_00087.jpg", "watermelon": "watermelon_img_00027.jpg"}   # same images as the earlier figure
GROUP_ORDER = ["Food and plant", "Household non-food", "Medical"]
CLS = CM.assign(g=CM.group.map(GROUP_ORDER.index), i=CM.item_name.str.lower()).sort_values(["g", "i"]).reset_index(drop=True)
LETTERS = "abcdefghijklmnopqrstuvwxyz"
RAWS = [Path(p) / "BanglaWaste-23-raw" for p in [os.environ.get("BANGLAWASTE_ROOT", ""), r"E:\students\BanglaWaste-23", str(ROOT.parent / "BanglaWaste-23")] if p]
RAW = next((p for p in RAWS if p.exists()), None)
if RAW is None:
    print("raw images not found (set BANGLAWASTE_ROOT): Figure 1 not rebuilt")
else:
    from PIL import ImageOps
    cols, rows = 6, 4; tile = W / cols
    fig, axes = plt.subplots(rows, cols, figsize=(W, rows * (tile + 0.3)))
    for a in axes.ravel(): a.axis("off")
    for k, r in CLS.iterrows():
        with Image.open(RAW / r.class_folder / SAMPLE[r.class_folder]) as im:
            im.draft("RGB", (800, 800)); im = ImageOps.exif_transpose(im).convert("RGB")
            s_ = min(im.size); l_, t_ = (im.size[0] - s_) // 2, (im.size[1] - s_) // 2
            im = im.crop((l_, t_, l_ + s_, t_ + s_)).resize((480, 480), Image.LANCZOS)
        ax = axes.ravel()[k]; ax.imshow(np.asarray(im))
        name = r.item_name.replace(" (IV infusion)", "")
        if len(name) > 14: name = "\n".join(name.rsplit(" ", 1))          # two lines for long names
        ax.text(0.5, -0.04, f"({LETTERS[k]}) {name}", transform=ax.transAxes, ha="center", va="top", fontsize=7.5, linespacing=1.0)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.06, wspace=0.06, hspace=0.36)
    save(fig, "fig_samples")
    CLS.assign(letter=[LETTERS[k] for k in range(len(CLS))])[["letter", "class_folder", "item_name", "group"]].to_csv(TAB / "fig_samples_panels.csv", index=False)

# ============================================================ Figure 2: split protocol (no titles; explanation in the caption)
OBJ_COL = ["#2a78d6", "#1baf7a", "#eb6834", "#9b59b6", "#d4a017"]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(W, 1.95), gridspec_kw={"width_ratios": [1, 1.45]})
for ax in (axa, axb):
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.2); ax.axis("off")
rng = np.random.default_rng(3)
for r_, (name, col) in enumerate(zip(["Object 1", "Object 2", "Object 3"], OBJ_COL[:3])):
    y = 6.6 - r_ * 1.6
    axa.text(0.0, y + 0.3, name, fontsize=8, va="center")
    test = set(rng.choice(7, 2, replace=False))
    for j in range(7):
        x = 2.6 + j * 1.0
        axa.add_patch(Rectangle((x, y), 0.75, 0.75, facecolor=col if j not in test else "white", edgecolor=col, lw=1.1,
                                hatch="//////" if j in test else None))
axa.add_patch(Rectangle((0.2, 0.9), 0.7, 0.7, facecolor=FS.MID, edgecolor=FS.MID)); axa.text(1.15, 1.25, "training image", fontsize=8, va="center")
axa.add_patch(Rectangle((5.3, 0.9), 0.7, 0.7, facecolor="white", edgecolor=FS.MID, hatch="//////")); axa.text(6.25, 1.25, "test image", fontsize=8, va="center")
x0, w_, gap = 1.75, 1.45, 0.2
for k in range(5):
    x = x0 + k * (w_ + gap); test = k == 0
    axb.add_patch(FancyBboxPatch((x, 0.75), w_, 6.6, boxstyle="round,pad=0.02,rounding_size=0.12",
                                 facecolor="#fdece4" if test else "#eaf2fc", edgecolor=C_SCR if test else C_PRE, lw=1.3 if test else 0.8))
    axb.text(x + w_ / 2, 6.85, f"Fold {k + 1}", ha="center", fontsize=8, fontweight="bold")
    axb.text(x + w_ / 2, 1.1, "test" if test else "train", ha="center", fontsize=8, color=C_SCR if test else C_PRE)
for r_, cls in enumerate(["Class A", "Class B", "Class C"]):
    y = 5.25 - r_ * 1.35
    axb.text(0.0, y + 0.25, cls, fontsize=8, va="center")
    for k in range(5):
        x = x0 + k * (w_ + gap) + 0.16
        n = 1 if (k + r_) % 3 else 2
        for s_ in range(n):
            col = OBJ_COL[(r_ * 2 + k + s_) % 5]
            for d in range(3):
                axb.add_patch(Rectangle((x + s_ * 0.6 + d * 0.06, y + d * 0.06), 0.4, 0.4, facecolor=col, edgecolor="white", lw=0.5))
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.12, wspace=0.08)
FS.label_panels(fig, [(axa, "a"), (axb, "b")], pad=0.0)
save(fig, "fig_split_protocol")

# ============================================================ Figure 5: cross-region (a) + shortcut (b), no titles
BP = pd.read_csv(B3 / "bdwaste_per_class.csv"); BS = pd.read_csv(B3 / "bdwaste_summary.csv")
order_m = ["mobilenetv3_large_100", "convnext_tiny", "efficientnet_b0", "custom_v0"]
bp = BP.pivot(index="class", columns="model", values="acc_restricted")[order_m] * 100
bp = bp.loc[sorted(bp.index, key=lambda c: -bp.loc[c].mean())]
overall = BS.set_index("model").acc_restricted_to_shared.reindex(order_m) * 100
mat = np.vstack([bp.values, overall.values]); rows_ = [ITEM[c] for c in bp.index] + ["All 11 classes"]
SH = pd.read_csv(B3 / "shortcut_summary.csv")
NICE2 = {k: v.replace("Custom CNN (V0)", "Custom CNN V0") for k, v in NICE.items()}
fig = plt.figure(figsize=(W, 3.35)); gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.8)
ax = fig.add_subplot(gs[0]); im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
ax.axhline(len(bp) - 0.5, color="white", lw=2.5)
ax.set_xticks(range(4), [NICE2[m] for m in order_m], rotation=35, ha="right", fontsize=7.5); ax.set_yticks(range(len(rows_)), rows_, fontsize=7.5)
ax.get_yticklabels()[-1].set_fontweight("bold")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]; ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 55 else FS.INK,
                                fontweight="bold" if i == len(rows_) - 1 else None)
ax.spines[:].set_visible(False); ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03); cb.set_label("Accuracy on BDWaste (%)", fontsize=8); cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
ax2 = fig.add_subplot(gs[1]); conds = ["original", "jpeg_q30", "lowres_96"]; lab = ["Original", "JPEG\nq = 30", "96 px"]
x = np.arange(3); w = 0.36
for k, (m, fc_) in enumerate(zip(["convnext_tiny", "efficientnet_b0"], [C_PRE, C_GREEN])):
    v = [SH[(SH.model == m) & (SH.condition == c)].macro_f1.iloc[0] * 100 for c in conds]
    ax2.bar(x + (k - 0.5) * (w + 0.03), v, w, color=fc_, label=NICE2[m], zorder=2)
    for xi, vi in zip(x, v): ax2.text(xi + (k - 0.5) * (w + 0.03), vi + 0.2, f"{vi:.1f}", ha="center", fontsize=6.8)
ax2.set_xticks(x, lab, fontsize=7.5); ax2.set_ylim(90, 100.5); ax2.set_ylabel("Macro-F1 (%), mean of 5 folds", fontsize=8)
ax2.tick_params(axis="y", labelsize=7.5); ax2.grid(axis="y", color=FS.GRID, lw=0.5); ax2.set_axisbelow(True)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, fontsize=7.5)
FS.label_panels(fig, [(ax, "a"), (ax2, "b")])
save(fig, "fig_crossregion_shortcut")

import textwrap
# ============================================================ Fig. S1: collection and curation workflow (redrawn in the paper font)
STEPS = [("1. Planning", "List of target\nwaste items"), ("2. Collection", "Households, hostels,\ncanteen, street,\nhospital"),
         ("3. Photography", "Open Camera app\non smartphones"), ("4. Merging", "All images merged\ninto one collection"),
         ("5. Labelling", "One folder per class"), ("6. Verification", "Medical classes\nverified by a nurse"),   # as in the text (supervisor, 20 Nov 2026)
         ("7. Screening", "Blurred, dark,\nunintentional and\nwrong-item images\nremoved"),
         ("8. Clean-up", "Renamed; duplicates\nand images < 512 px\nremoved (821 of\n4,981 excluded)"),
         ("9. Release", "172 objects with\nconditions; 4,160\nimages on Zenodo")]
fig, ax = plt.subplots(figsize=(W, 3.0)); ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.axis("off")
bw, bh = 0.86, 0.8
pos = [(c, 2.1) for c in range(3)] + [(2 - c, 1.1) for c in range(3)] + [(c, 0.1) for c in range(3)]
for k, ((h, body), (cx, y)) in enumerate(zip(STEPS, pos)):
    x = cx + (1 - bw) / 2
    ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.04", facecolor="#eaf2fc", edgecolor=C_PRE, lw=0.9))
    ax.text(x + bw / 2, y + bh - 0.08, h, ha="center", va="top", fontsize=7.5, fontweight="bold")
    ax.text(x + bw / 2, y + bh - 0.26, "\n".join(textwrap.wrap(body.replace("\n", " "), 22)), ha="center", va="top", fontsize=7, linespacing=1.15, wrap=False)
for k in range(len(pos) - 1):
    (cx0, y0), (cx1, y1) = pos[k], pos[k + 1]
    if y0 == y1:
        xa, xb = (cx0 + (1 + bw) / 2, cx1 + (1 - bw) / 2) if cx1 > cx0 else (cx0 + (1 - bw) / 2, cx1 + (1 + bw) / 2)
        ax.annotate("", xy=(xb, y0 + bh / 2), xytext=(xa, y0 + bh / 2), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=FS.INK2))
    else:
        ax.annotate("", xy=(cx1 + 0.5, y1 + bh), xytext=(cx0 + 0.5, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=FS.INK2))
save(fig, "fig_s_workflow")

# ============================================================ Fig. S2: images per class (groups in green / blue / orange, as before)
CS = CLS.copy(); cnt = pd.read_csv(RES / "reference" / "fold_counts.csv", index_col=0)
CS["n"] = CS.class_folder.map(cnt["total"] if "total" in cnt.columns else cnt.sum(1)).astype(int)
STY = {"Food and plant": dict(facecolor=C_GREEN), "Household non-food": dict(facecolor=C_PRE), "Medical": dict(facecolor=C_SCR)}
fig, ax = plt.subplots(figsize=(W, 4.0)); y = np.arange(len(CS))[::-1]
for yi, r in zip(y, CS.itertuples()):
    ax.barh(yi, r.n, 0.72, **STY[r.group])
    ax.text(r.n + 2, yi, str(r.n), va="center", fontsize=7.5)
ax.set_yticks(y, CS.item_name, fontsize=7.5); ax.set_xlabel("Number of images"); ax.set_xlim(0, 235); ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=FS.GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(handles=[Patch(label=g + " waste", **STY[g]) for g in GROUP_ORDER],
          loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, fontsize=7.5)
save(fig, "fig_s_class_distribution")

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
    ("BanglaWaste-23 (ours)", "", 2026, "4,160", "23", "Real, households and hospital", "Yes (4 classes)", "Yes (172 objects)", "Object-disjoint 5-fold"),
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
          ("table_lr_tuning", "Learning-rate selection on the fold-1 validation objects: mean validation macro-F1 (\\%) over epochs 16--40.", "tab:s_lr"),
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

# Table S9: custom-model pilot on the fold-1 tuning split (every run; final-epoch EMA weights)
PL = pd.read_csv(RES / "custom_pilot" / "pilot_summary.csv")
PNAME = {"convnext_tiny": "ConvNeXt-T (teacher)", "custom_v0": "Custom CNN V0", "custom_v1": "Custom CNN V1 (redesign)",
         "mobilenetv3_large_100": "MobileNetV3-L", "hyb_mnv3_msa": "MobileNetV3-L + multi-scale/attention neck"}
PINIT = {"convnext_tiny": "pretrained", "custom_v0": "from scratch", "custom_v1": "from scratch",
         "mobilenetv3_large_100": "pretrained", "hyb_mnv3_msa": "pretrained backbone"}
order = ["custom_v0", "custom_v1", "mobilenetv3_large_100", "hyb_mnv3_msa", "convnext_tiny"]
PL["o"] = PL.model.map(order.index); PL = PL.sort_values(["o", "kd"])
rows = []
for r in PL.itertuples():
    rows.append({"Model": PNAME[r.model], "Initialisation": PINIT[r.model], "KD": "yes" if r.kd else "no",
                 "Params (M)": round(r.params / 1e6, 2), "MACs (M)": int(round(r.macs_M)), "Epochs": int(r.epochs),
                 "Val. accuracy (%)": round(r.final_val_acc * 100, 1), "Val. macro-F1 (%)": round(r.final_val_macro_f1 * 100, 1)})
PT = pd.DataFrame(rows); PT.to_csv(TAB / "table_pilot.csv", index=False)
body = [" & ".join([latex_escape(v["Model"]), v["Initialisation"], v["KD"], f'{v["Params (M)"]:.2f}', f'{v["MACs (M)"]:,}',
                    str(v["Epochs"]), f'{v["Val. accuracy (%)"]:.1f}', f'{v["Val. macro-F1 (%)"]:.1f}']) + " \\\\" for v in rows]
write_table("tex_table_pilot",
            "Custom-model pilot on the fold-1 tuning split (623 validation images of 23 held-out objects): final-epoch results. "
            "KD: knowledge distillation from the ConvNeXt-T teacher (temperature 4, weight 0.7).",
            "tab:s_pilot", "@{}lllrrrrr@{}",
            "Model & Initialisation & KD & Params (M) & MACs (M) & Epochs & Val. accuracy (\\%) & Val. macro-F1 (\\%)", body, size="\\scriptsize")

# ============================================================ Copy into manuscript folders
MAIN_FIG = {"fig_samples": FIG / "fig_samples", "fig_split_protocol": FIG / "fig_split_protocol",
            "fig_accuracy_vs_macs": FIG / "fig_accuracy_vs_macs", "fig_leakage": FIG / "fig_leakage",
            "fig_crossregion_shortcut": FIG / "fig_crossregion_shortcut"}
SUPP_FIG = {"fig_s_workflow": FIG / "fig_s_workflow", "fig_s_class_distribution": FIG / "fig_s_class_distribution",
            "fig_s_fold_composition": FIG / "fig_fold_composition", "fig_s_tuning_curves": FIG / "fig_supp_tuning_curves",
            "fig_s_per_class_f1": FIG / "fig_per_class_f1", "fig_s_confusion_best": FIG / "fig_confusion_best",
            "fig_s_confusion_custom_v0": FIG / "fig_confusion_custom_v0", "fig_s_bdwaste_per_class": FIG / "fig_bdwaste_per_class"}
GRADCAM = {f"fig_s_gradcam_{m}_{k}": FIG / f"fig_gradcam_{m}_{k}" for m in ["convnext_tiny", "efficientnet_b0", "custom_v0", "mobilenetv3_scratch"]
           for k in ["correct", "wrong"]}
MAIN_TAB = ["tex_table_dataset_comparison", "tex_table_main_results", "tex_table_groups_medical"]
SUPP_TAB = [f"tex_{n}" for n, _, _ in SUPP_T] + ["tex_table_pilot"]

if not (ROOT / 'manuscript').exists():
    print('manuscript/ not found: figures and tables written to results/paper only'); raise SystemExit
for base in (SH_, OV):
    for sub in (("figures/main", "figures/supp") if base == SH_ else ("figures/main", "figures/supp", "tables/main", "tables/supp")): (base / sub).mkdir(parents=True, exist_ok=True)
for dst, src in MAIN_FIG.items():
    shutil.copy(f"{src}.pdf", OV / "figures/main" / f"{dst}.pdf")
    for ext in ("pdf", "png"): shutil.copy(f"{src}.{ext}", SH_ / "figures/main" / f"{dst}.{ext}")
(SH_ / "figures/submission").mkdir(parents=True, exist_ok=True)          # separate figure files for the journal upload
for k, dst in enumerate(["fig_samples", "fig_split_protocol", "fig_accuracy_vs_macs", "fig_leakage", "fig_crossregion_shortcut"], 1):
    shutil.copy(f"{MAIN_FIG[dst]}.pdf", SH_ / "figures/submission" / f"Figure_{k}.pdf")
for dst, src in SUPP_FIG.items():
    shutil.copy(f"{src}.pdf", OV / "figures/supp" / f"{dst}.pdf")
    for ext in ("pdf", "png"): shutil.copy(f"{src}.{ext}", SH_ / "figures/supp" / f"{dst}.{ext}")
for dst, src in GRADCAM.items():                  # photo-heavy: JPEG keeps Overleaf small
    im = Image.open(f"{src}.png").convert("RGB")
    for base in (SH_, OV): im.save(base / "figures/supp" / f"{dst}.jpg", quality=90, dpi=(300, 300))   # 300 dpi = true size in LaTeX
import os
if os.environ.get("WASTE_FIGURES_ONLY"):             # set this to refresh figures without touching Overleaf tables / references
    print("WASTE_FIGURES_ONLY set: Overleaf tables and references.bib left unchanged"); MAIN_TAB, SUPP_TAB = [], []
for n in MAIN_TAB:
    for base in (OV,): shutil.copy(TAB / f"{n}.tex", base / "tables/main" / f"{n.replace('tex_', '')}.tex")
for n in SUPP_TAB:
    for base in (OV,): shutil.copy(TAB / f"{n}.tex", base / "tables/supp" / f"{n.replace('tex_', '')}.tex")
if not os.environ.get("WASTE_FIGURES_ONLY"): shutil.copy(ROOT / "literature" / "references.bib", OV / "references.bib")
print("main figures:", list(MAIN_FIG)); print("main tables:", MAIN_TAB)
print("supp figures:", len(SUPP_FIG) + len(GRADCAM), "supp tables:", len(SUPP_TAB))
