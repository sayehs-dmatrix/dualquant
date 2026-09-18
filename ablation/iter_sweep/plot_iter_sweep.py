"""Plot DualQuant PPL vs iteration for each model × format."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).parent / "results_dualQuant_with_Iter"
OUT_FILE = Path(__file__).parent / "dualquant_iter_sweep.pdf"

FORMATS  = ["rtn_int4", "mxfp4"]
MARKERS  = {"rtn_int4": "o", "mxfp4": "s"}
COLORS   = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# ── Load all iter CSVs ────────────────────────────────────────────────────────
records = []
for f in DATA_DIR.glob("*_iter*.csv"):
    m = re.search(r"_iter(\d+)\.csv$", f.name)
    if m is None:
        continue
    iteration = int(m.group(1))
    df = pd.read_csv(f)
    df["iteration"] = iteration
    records.append(df)

data = pd.concat(records, ignore_index=True)
data = data[data["weight_fmt"].isin(FORMATS)].copy()
# average duplicate (model, weight_fmt, iteration) rows
data = (
    data.groupby(["model", "weight_fmt", "iteration"], as_index=False)["ppl_wikitext"]
    .mean()
)

models = sorted(data["model"].unique())
color_map = {model: COLORS[i % len(COLORS)] for i, model in enumerate(models)}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), sharey=False)

for ax, model in zip(axes, models):
    short = model.split("/")[-1]
    for fmt in FORMATS:
        subset = (
            data[(data["model"] == model) & (data["weight_fmt"] == fmt)]
            .sort_values("iteration")
        )
        if subset.empty:
            continue
        ax.plot(
            subset["iteration"],
            subset["ppl_wikitext"],
            color=color_map[model],
            marker=MARKERS[fmt],
            linestyle="-" if fmt == "rtn_int4" else "--",
            label=fmt,
        )
    ax.set_title(short)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PPL (WikiText-2)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("DualQuant: PPL vs Iteration", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_FILE)
print(f"Saved: {OUT_FILE}")
