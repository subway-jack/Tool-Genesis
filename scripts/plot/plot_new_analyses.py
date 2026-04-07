"""
Generate four publication-quality figures for the Tool-Genesis paper.
=======================================================================
1. Judge Re-eval Scatter Plot         -> images/judge_reeval_scatter.pdf
2. Solver Ablation Comparison         -> images/solver_ablation_comparison.pdf
3. Task Dependency Distribution       -> images/task_dependency_dist.pdf
4. Structural Equivalence Histogram   -> images/structural_equivalence_hist.pdf
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/Users/subway/code/python/项目/Multi-agent/tool-genesis/tool-genesis")
TEMP = ROOT / "temp"
DATA = ROOT / "data"
IMG  = ROOT / "images"
IMG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
TITLE_SIZE = 13
LABEL_SIZE = 11
TICK_SIZE  = 10
COLORS = plt.cm.tab10.colors

# ===================================================================
# Figure 1: Judge Re-eval Scatter Plot
# ===================================================================
def plot_judge_reeval_scatter():
    models = [
        ("coder_agent_openai_gpt-5-1",              "GPT-5.1"),
        ("coder_agent_google_gemini-3-flash-preview", "Gemini-3-Flash"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    all_x, all_y = [], []
    for idx, (model_key, label) in enumerate(models):
        # GPT-5.4 judge SR per server
        judge_path = TEMP / "judge_reeval_gpt54_full" / model_key / "per_server.json"
        judge_data = json.loads(judge_path.read_text())

        # Original SR per server (Qwen3-14B judge)
        orig_path = TEMP / "eval_results_v3" / model_key / "results.json"
        orig_list = json.loads(orig_path.read_text())
        orig_sr = {
            item["server_slug"]: item["metrics"]["trajectory_level_validation_rate_soft"]
            for item in orig_list
        }

        xs, ys = [], []
        for server, info in judge_data.items():
            if server in orig_sr:
                x = orig_sr[server]
                y = info["sr_soft"]
                xs.append(x)
                ys.append(y)

        ax.scatter(xs, ys, s=28, alpha=0.7, color=COLORS[idx],
                   label=f"{label} (n={len(xs)})", edgecolors="white", linewidths=0.3)
        all_x.extend(xs)
        all_y.extend(ys)

    # Diagonal y = x
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, alpha=0.7, label="y = x")

    # Spearman rho
    rho, pval = stats.spearmanr(all_x, all_y)
    ax.annotate(f"Spearman $\\rho$ = {rho:.3f}  (p = {pval:.1e})",
                xy=(0.03, 0.95), xycoords="axes fraction",
                fontsize=LABEL_SIZE, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.set_xlabel("Original SR (Qwen3-14B judge)", fontsize=LABEL_SIZE)
    ax.set_ylabel("GPT-5.4 Judge SR", fontsize=LABEL_SIZE)
    ax.set_title("Judge reliability: per-server SR under Qwen3-14B vs GPT-5.4 judge",
                 fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=TICK_SIZE)
    fig.tight_layout()
    out = IMG / "judge_reeval_scatter.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# Figure 2: Solver Ablation Comparison
# ===================================================================
def is_valid(v):
    """Filter out servers where API quota failure caused sr_soft=0."""
    if v.get("sr_soft", 0) > 0:
        return True
    details = v.get("details", [])
    if not details:
        return False
    return not all(d.get("score", 0) == 0 for d in details)


def plot_solver_ablation():
    models = [
        ("coder_agent_deepseek_deepseek-v3-2",        "DeepSeek-V3"),
        ("coder_agent_google_gemini-3-flash-preview",  "Gemini-3-Flash"),
        ("coder_agent_openai_gpt-4-1",                "GPT-4.1"),
        ("coder_agent_qwen3-32b",                     "Qwen3-32B"),
        ("coder_agent_openai_gpt-5-1",                "GPT-5.1"),
    ]

    orig_sr_all = []
    gpt54_sr_all = []
    labels = []

    for model_key, short_name in models:
        # GPT-5.4 solver per-server
        ablation_path = TEMP / "solver_ablation_gpt54_multi" / model_key / "per_server.json"
        ablation_data = json.loads(ablation_path.read_text())

        # Original per-server SR from eval_results_v3
        orig_path = TEMP / "eval_results_v3" / model_key / "results.json"
        orig_list = json.loads(orig_path.read_text())
        orig_sr_map = {
            item["server_slug"]: item["metrics"]["trajectory_level_validation_rate_soft"]
            for item in orig_list
        }

        # Filter valid servers and compute mean SR for GPT-5.4 solver
        valid_servers = {s: v for s, v in ablation_data.items() if is_valid(v)}
        if not valid_servers:
            gpt54_mean = 0.0
        else:
            gpt54_mean = np.mean([v["sr_soft"] for v in valid_servers.values()])

        # Original SR: average only over the same valid servers for fair comparison
        orig_vals = [orig_sr_map[s] for s in valid_servers if s in orig_sr_map]
        orig_mean = np.mean(orig_vals) if orig_vals else 0.0

        orig_sr_all.append(orig_mean)
        gpt54_sr_all.append(gpt54_mean)
        labels.append(short_name)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, orig_sr_all, width, label="Original (Qwen3-14B judge)",
                   color=COLORS[0], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, gpt54_sr_all, width, label="GPT-5.4 solver",
                   color=COLORS[1], edgecolor="white", linewidth=0.5)

    # Delta labels
    for i in range(len(labels)):
        delta = gpt54_sr_all[i] - orig_sr_all[i]
        sign = "+" if delta >= 0 else ""
        y_pos = max(orig_sr_all[i], gpt54_sr_all[i]) + 0.02
        ax.text(x[i], y_pos, f"{sign}{delta:.2f}", ha="center", va="bottom",
                fontsize=TICK_SIZE, fontweight="bold", color="black")

    ax.set_xlabel("Model", fontsize=LABEL_SIZE)
    ax.set_ylabel("Success Rate (SR)", fontsize=LABEL_SIZE)
    ax.set_title("Solver ablation: Qwen3-14B vs GPT-5.4 solver", fontsize=TITLE_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE, loc="upper left")
    ax.set_ylim(0, max(max(orig_sr_all), max(gpt54_sr_all)) + 0.12)
    ax.tick_params(labelsize=TICK_SIZE)
    fig.tight_layout()
    out = IMG / "solver_ablation_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# Figure 3: Task Dependency Distribution
# ===================================================================
def plot_task_dependency():
    dep_data = json.loads((DATA / "task_dependency_analysis.json").read_text())
    per_server = dep_data["per_server"]

    # Only servers with no_tools_sr data (non-null)
    servers = {k: v for k, v in per_server.items() if v.get("no_tools_sr") is not None}
    # Sort by no_tools_sr ascending
    sorted_items = sorted(servers.items(), key=lambda kv: kv[1]["no_tools_sr"])

    names = [k.replace("-", " ").title()[:35] for k, _ in sorted_items]
    sr_vals = [v["no_tools_sr"] for _, v in sorted_items]
    is_dep = [v.get("tool_dependent", True) for _, v in sorted_items]
    bar_colors = [COLORS[3] if dep else COLORS[2] for dep in is_dep]  # red vs green

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.barh(range(len(names)), sr_vals, color=bar_colors, edgecolor="white", linewidth=0.5)

    ax.axvline(x=0.7, color="black", linestyle="--", linewidth=1, alpha=0.7, label="Threshold = 0.7")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=TICK_SIZE - 1)
    ax.set_xlabel("No-Tools SR", fontsize=LABEL_SIZE)
    ax.set_title("Task dependency: no-tools SR by server", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[3], label="Tool-dependent"),
        Patch(facecolor=COLORS[2], label="Tool-independent"),
        plt.Line2D([0], [0], color="black", linestyle="--", lw=1, label="Threshold = 0.7"),
    ]
    ax.legend(handles=legend_elements, fontsize=TICK_SIZE, loc="lower right")
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    out = IMG / "task_dependency_dist.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# Figure 4: Structural Equivalence Histogram
# ===================================================================
def plot_structural_equivalence():
    eq_data = json.loads((DATA / "structural_equivalence_analysis.json").read_text())
    count_hist = eq_data["tool_count_diff_histogram"]  # str keys -> int counts
    f1_by_diff = eq_data["schema_f1_by_diff"]           # str keys -> float f1

    # Sort by numeric key
    all_keys = sorted(set(list(count_hist.keys()) + list(f1_by_diff.keys())), key=lambda k: int(k))
    x_vals  = [int(k) for k in all_keys]
    counts  = [count_hist.get(k, 0) for k in all_keys]
    f1_vals = [f1_by_diff.get(k, None) for k in all_keys]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # Bar chart (left y-axis) - counts
    bar_color = COLORS[0]
    ax1.bar(x_vals, counts, color=bar_color, alpha=0.75, edgecolor="white", linewidth=0.5,
            label="Count", zorder=2)
    ax1.set_xlabel("Tool Count Difference", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Count (model-server pairs)", fontsize=LABEL_SIZE, color=bar_color)
    ax1.tick_params(axis="y", labelcolor=bar_color, labelsize=TICK_SIZE)
    ax1.tick_params(axis="x", labelsize=TICK_SIZE)

    # Right y-axis - Schema-F1
    ax2 = ax1.twinx()
    line_color = COLORS[3]
    # Filter None values for line plot
    x_line = [x for x, f in zip(x_vals, f1_vals) if f is not None]
    y_line = [f for f in f1_vals if f is not None]
    ax2.plot(x_line, y_line, color=line_color, marker="o", markersize=5, linewidth=2,
             label="Avg Schema-F1", zorder=3)
    ax2.set_ylabel("Avg Schema-F1", fontsize=LABEL_SIZE, color=line_color)
    ax2.tick_params(axis="y", labelcolor=line_color, labelsize=TICK_SIZE)
    ax2.set_ylim(0, 1.05)

    ax1.set_title("Schema-F1 vs tool count difference", fontsize=TITLE_SIZE)

    # Combined legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=bar_color, alpha=0.75, label="Count"),
        Line2D([0], [0], color=line_color, marker="o", markersize=5, lw=2, label="Avg Schema-F1"),
    ]
    ax1.legend(handles=legend_elements, fontsize=TICK_SIZE, loc="upper right")

    fig.tight_layout()
    out = IMG / "structural_equivalence_hist.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating figures...")
    plot_judge_reeval_scatter()
    plot_solver_ablation()
    plot_task_dependency()
    plot_structural_equivalence()
    print("Done.")
