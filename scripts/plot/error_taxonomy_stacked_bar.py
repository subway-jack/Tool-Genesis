"""
Figure: Error Taxonomy Stacked Bar Chart.
Groups runs by strategy (Direct vs Code-Agent), aggregates error counts,
normalizes to percentages, and renders a stacked horizontal bar chart.

For Tool-Genesis ICML paper.
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# 1.  Data paths (relative to repo root; script is run from repo root)
# ---------------------------------------------------------------------------
SUMMARY_PATH = "data/analysis_c_summary.json"
OUT_DIR = "scripts/plot"

# ---------------------------------------------------------------------------
# 2.  Error taxonomy definition (order = bottom → top of stacked bar)
# ---------------------------------------------------------------------------
ERROR_TYPES_ORDERED = [
    # L1
    "format_non_compliant",
    "launch_fail_runtime",
    # L2
    "tool_missing",
    "tool_extra",
    "arg_missing",
    "arg_extra",
    "arg_type_error",
    "description_drift",
    # L3
    "ut_wrong_output",
    "ut_execution_error",
    # L4
    "task_not_solved",
]

COLOR_MAP = {
    "format_non_compliant":  "#e74c3c",   # red
    "launch_fail_runtime":   "#e67e22",   # orange
    "tool_missing":          "#f1c40f",   # yellow
    "tool_extra":            "#d4ac0d",   # gold
    "arg_missing":           "#aed6f1",   # light blue
    "arg_extra":             "#1abc9c",   # cyan
    "arg_type_error":        "#117a65",   # teal
    "description_drift":     "#95a5a6",   # gray
    "ut_wrong_output":       "#2471a3",   # dark blue
    "ut_execution_error":    "#8e44ad",   # purple
    "task_not_solved":       "#f1948a",   # pink
}

LABELS = {
    "format_non_compliant":  "L1: Format Non-Compliant",
    "launch_fail_runtime":   "L1: Launch Fail (Runtime)",
    "tool_missing":          "L2: Tool Missing",
    "tool_extra":            "L2: Tool Extra",
    "arg_missing":           "L2: Arg Missing",
    "arg_extra":             "L2: Arg Extra",
    "arg_type_error":        "L2: Arg Type Error",
    "description_drift":     "L2: Description Drift",
    "ut_wrong_output":       "L3: UT Wrong Output",
    "ut_execution_error":    "L3: UT Execution Error",
    "task_not_solved":       "L4: Task Not Solved",
}

# ---------------------------------------------------------------------------
# 3.  Load & aggregate summary data
# ---------------------------------------------------------------------------
with open(SUMMARY_PATH) as f:
    summary = json.load(f)

per_model = summary["per_model_summary"]

group_counts = {
    "Direct":      {et: 0 for et in ERROR_TYPES_ORDERED},
    "Code-Agent":  {et: 0 for et in ERROR_TYPES_ORDERED},
}

for run_name, err_dict in per_model.items():
    if run_name.startswith("direct_"):
        grp = "Direct"
    elif run_name.startswith("coder_agent_"):
        grp = "Code-Agent"
    else:
        continue
    for et in ERROR_TYPES_ORDERED:
        group_counts[grp][et] += err_dict.get(et, 0)

# Normalize to percentages within each group
group_pct = {}
for grp, counts in group_counts.items():
    total = sum(counts.values())
    group_pct[grp] = {et: (counts[et] / total * 100 if total else 0) for et in ERROR_TYPES_ORDERED}

# ---------------------------------------------------------------------------
# 4.  Plot
# ---------------------------------------------------------------------------
groups = ["Direct", "Code-Agent"]
x = np.arange(len(groups))
bar_width = 0.55

fig, ax = plt.subplots(figsize=(9, 5.5))

bottoms = np.zeros(len(groups))
for et in ERROR_TYPES_ORDERED:
    vals = np.array([group_pct[g][et] for g in groups])
    bars = ax.bar(x, vals, bar_width, bottom=bottoms,
                  color=COLOR_MAP[et], label=LABELS[et],
                  edgecolor="white", linewidth=0.4)
    # Annotate segments that are wide enough to read
    for xi, (val, bot) in enumerate(zip(vals, bottoms)):
        if val > 2.5:
            ax.text(xi, bot + val / 2, f"{val:.1f}%",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=13, fontweight="bold")
ax.set_ylabel("Percentage of Total Errors (%)", fontsize=11)
ax.set_title("Error Taxonomy Distribution by Strategy", fontsize=13, fontweight="bold")
ax.set_ylim(0, 105)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend outside, ordered same as stack (bottom → top → reversed for readability)
handles = [
    mpatches.Patch(color=COLOR_MAP[et], label=LABELS[et])
    for et in reversed(ERROR_TYPES_ORDERED)
]
ax.legend(
    handles=handles,
    title="Error Type",
    title_fontsize=9,
    fontsize=8,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    framealpha=0.9,
)

plt.tight_layout()

os.makedirs(OUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUT_DIR, "error_taxonomy.pdf")
png_path = os.path.join(OUT_DIR, "error_taxonomy.png")
plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
plt.savefig(png_path, bbox_inches="tight", dpi=300)
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

# Print summary for sanity-check
print("\n=== Aggregated Error Counts ===")
for grp in groups:
    total = sum(group_counts[grp].values())
    print(f"\n{grp} (total errors counted: {total})")
    for et in ERROR_TYPES_ORDERED:
        cnt = group_counts[grp][et]
        pct = group_pct[grp][et]
        print(f"  {et:<30s}  {cnt:>6d}  ({pct:5.1f}%)")
