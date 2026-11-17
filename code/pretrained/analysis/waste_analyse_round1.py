"""Round 1 (tuning) analysis: choose the learning rate per model and the shared epoch budget.

Reads  runs/round1_pilot/extracted/runs/fold0_val/<model>_lr<lr>/fold0val/{history,metrics,job}.*
Writes runs/round1_pilot/round1_lr_table.csv     (all runs: final, best, window means)
       runs/round1_pilot/round1_choice.csv       (chosen LR per model + epoch budget)
Rule:  score = mean validation macro-F1 (EMA weights) over epochs 16-40 of the 80-epoch run;
       the LR with the highest score is chosen; epoch budget = 40 for all pretrained models.
Run:   python code/pretrained/analysis/waste_analyse_round1.py   (from the Claude folder)
"""
import glob, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
R1 = ROOT / "runs" / "round1_pilot"
EPOCHS = 40
rows = []
for f in sorted(glob.glob(str(R1 / "extracted" / "runs" / "fold0_val" / "*" / "fold0val" / "history.csv"))):
    d = Path(f).parent
    job = json.load(open(d / "job.json")); m = pd.read_csv(d / "metrics.csv").iloc[0]
    h = pd.read_csv(f).dropna(subset=["val_macro_f1_ema"]).set_index("epoch").val_macro_f1_ema
    r = {"model": job["model"], "lr": job["lr"], "score_ep16_40": h.loc[16:40].mean(),
         "best_val_f1": h.max(), "best_epoch": int(h.idxmax()), "final_val_f1": m.macro_f1,
         "final_val_acc": m.accuracy, "minutes_80ep": m.minutes, "params": m.params, "macs_M": m.macs_M}
    for a, b in [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80)]:
        r[f"val_f1_ep{a}_{b}"] = h.loc[a:b].mean()
    rows.append(r)
t = pd.DataFrame(rows).sort_values(["model", "lr"])
t.to_csv(R1 / "round1_lr_table.csv", index=False)
grid = {1e-3, 3e-4, 1e-4}
ch = (t.sort_values("score_ep16_40", ascending=False).groupby("model").head(1)
        .assign(epochs=EPOCHS)[["model", "lr", "epochs", "score_ep16_40", "minutes_80ep"]])
ch["lrs_tested"] = ch.model.map(t.groupby("model").lr.apply(lambda s: ",".join(f"{x:g}" for x in sorted(s))))
ch["grid_complete"] = ch.model.map(t.groupby("model").lr.apply(lambda s: set(s) == grid))
ch.sort_values("model").to_csv(R1 / "round1_choice.csv", index=False)
print(ch.sort_values("model").to_string(index=False))
