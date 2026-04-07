"""
Scaling Law Analysis for Tool-Genesis Benchmark (Qwen3 Family)
==============================================================
Loads evaluation results from temp/eval_results_v3/*/results.json,
computes per-level metrics for the Qwen3 model family (4B–235B),
generates a publication-quality figure, and saves data to JSON.
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/Users/subway/code/python/项目/Multi-agent/tool-genesis")
EVAL_DIR  = REPO_ROOT / "temp" / "eval_results_v3"
PLOT_DIR  = REPO_ROOT / "scripts" / "plot"
DATA_DIR  = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Qwen3 model family definition
# ---------------------------------------------------------------------------
QWEN3_RUNS = [
    ("qwen3-4b",                       4),
    ("qwen3-8b",                       8),
    ("qwen3-14b",                      14),
    ("qwen3-30b-a3b-instruct-2507",    30),
    ("qwen3-32b",                      32),
    ("qwen3-235b-a22b-instruct-2507", 235),
]

STRATEGIES = {
    "direct":       ("direct",       "Direct"),
    "coder_agent":  ("coder_agent",  "Coder-Agent"),
}

# ---------------------------------------------------------------------------
# All models with approximate active-parameter size (B) for sorting
# ---------------------------------------------------------------------------
ALL_MODEL_SIZES = {
    # Qwen3 family
    "qwen3-4b":                       4,
    "qwen3-8b":                       8,
    "qwen3-8b-sft":                   8,
    "qwen3-14b":                      14,
    "qwen3-30b-a3b-instruct-2507":    3,    # MoE active ~3B
    "qwen3-32b":                      32,
    "qwen3-235b-a22b-instruct-2507":  22,   # MoE active ~22B
    # Frontier / proprietary
    "anthropic_claude-3-5-haiku":     20,
    "anthropic_claude-sonnet-4":      200,
    "deepseek_deepseek-v3-2":         37,   # MoE active
    "google_gemini-3-flash-preview":  15,
    "moonshotai_kimi-k2":             32,   # MoE active
    "openai_gpt-4-1-mini":            15,
    "openai_gpt-4-1":                 200,
    "openai_gpt-4o":                  200,
    "openai_gpt-5-1":                 1000,
    "openai_gpt-5-2":                 1000,
    "openai_gpt-o3":                  1000,
}

# ---------------------------------------------------------------------------
# Helper: load and average metrics from a results.json
# ---------------------------------------------------------------------------
def load_results(run_dir: Path):
    path = run_dir / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        records = json.load(f)

    n = len(records)
    if n == 0:
        return None

    sums = dict(
        compliance=0.0,
        server_launch_success=0.0,
        schema_f1=0.0,
        tool_call_success_rate_soft=0.0,
        tool_call_success_rate_hard=0.0,
        trajectory_level_validation_rate_soft=0.0,
        trajectory_level_validation_rate_hard=0.0,
    )

    for rec in records:
        m = rec["metrics"]
        sums["compliance"]                         += float(m.get("compliance", 0) or 0)
        sums["server_launch_success"]              += float(m.get("server_launch_success", 0) or 0)
        sums["schema_f1"]                          += float(m.get("schema_f1", 0) or 0)
        sums["tool_call_success_rate_soft"]        += float(m.get("tool_call_success_rate_soft", 0) or 0)
        sums["tool_call_success_rate_hard"]        += float(m.get("tool_call_success_rate_hard", 0) or 0)
        sums["trajectory_level_validation_rate_soft"] += float(
            m.get("trajectory_level_validation_rate_soft", 0) or 0)
        sums["trajectory_level_validation_rate_hard"] += float(
            m.get("trajectory_level_validation_rate_hard", 0) or 0)

    avgs = {k: v / n for k, v in sums.items()}
    avgs["n_tasks"] = n
    return avgs


# ---------------------------------------------------------------------------
# Build Qwen3 scaling data
# ---------------------------------------------------------------------------
def build_qwen3_data():
    data = {}
    for strategy_key, (prefix, label) in STRATEGIES.items():
        data[strategy_key] = {"label": label, "sizes": [], "metrics": {}}
        metrics_lists = {
            "L1_compliance": [],
            "L1_launch": [],
            "L2_schema_f1": [],
            "L3_ut_soft": [],
            "L3_ut_hard": [],
            "L4_sr_soft": [],
            "L4_sr_hard": [],
        }
        sizes_found = []
        for run_name, size_b in QWEN3_RUNS:
            run_dir = EVAL_DIR / f"{prefix}_{run_name}"
            avgs = load_results(run_dir)
            if avgs is None:
                print(f"  [WARN] missing: {run_dir}")
                continue
            sizes_found.append(size_b)
            metrics_lists["L1_compliance"].append(avgs["compliance"])
            metrics_lists["L1_launch"].append(avgs["server_launch_success"])
            metrics_lists["L2_schema_f1"].append(avgs["schema_f1"])
            metrics_lists["L3_ut_soft"].append(avgs["tool_call_success_rate_soft"])
            metrics_lists["L3_ut_hard"].append(avgs["tool_call_success_rate_hard"])
            metrics_lists["L4_sr_soft"].append(avgs["trajectory_level_validation_rate_soft"])
            metrics_lists["L4_sr_hard"].append(avgs["trajectory_level_validation_rate_hard"])
            print(f"  Loaded {prefix}_{run_name}: "
                  f"launch={avgs['server_launch_success']:.3f} "
                  f"f1={avgs['schema_f1']:.3f} "
                  f"ut_soft={avgs['tool_call_success_rate_soft']:.3f} "
                  f"sr_soft={avgs['trajectory_level_validation_rate_soft']:.3f}")

        data[strategy_key]["sizes"] = sizes_found
        data[strategy_key]["metrics"] = {k: v for k, v in metrics_lists.items()}
    return data


# ---------------------------------------------------------------------------
# Build all-models data
# ---------------------------------------------------------------------------
def build_all_models_data():
    results = []
    for run_dir in sorted(EVAL_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        # Determine strategy prefix
        if name.startswith("direct_"):
            strategy = "direct"
            model_key = name[len("direct_"):]
        elif name.startswith("coder_agent_"):
            strategy = "coder_agent"
            model_key = name[len("coder_agent_"):]
        else:
            continue

        avgs = load_results(run_dir)
        if avgs is None:
            continue

        size_b = ALL_MODEL_SIZES.get(model_key, None)

        results.append({
            "run_name": name,
            "model_key": model_key,
            "strategy": strategy,
            "approx_active_params_B": size_b,
            "n_tasks": avgs["n_tasks"],
            "L1_compliance": avgs["compliance"],
            "L1_launch": avgs["server_launch_success"],
            "L2_schema_f1": avgs["schema_f1"],
            "L3_ut_soft": avgs["tool_call_success_rate_soft"],
            "L3_ut_hard": avgs["tool_call_success_rate_hard"],
            "L4_sr_soft": avgs["trajectory_level_validation_rate_soft"],
            "L4_sr_hard": avgs["trajectory_level_validation_rate_hard"],
        })

    # Sort by strategy then approximate size
    results.sort(key=lambda r: (r["strategy"], r["approx_active_params_B"] or 9999))
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_scaling_law(qwen3_data: dict, out_stem: str):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "lines.linewidth": 2.5,
        "lines.markersize": 9,
    })

    # ---- colour palette ----
    COLOR_DIRECT = "#1565C0"      # dark blue  – solid line
    COLOR_AGENT  = "#D32F2F"      # dark red   – dashed line

    METRICS = [
        ("L1_launch",    "L1: Server Launch Rate"),
        ("L2_schema_f1", "L2: Schema F1"),
        ("L3_ut_soft",   "L3: Unit-Test (soft)"),
        ("L4_sr_soft",   "L4: Success Rate (soft)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        for strategy_key, color, ls, marker in [
            ("direct",      COLOR_DIRECT, "-",  "o"),
            ("coder_agent", COLOR_AGENT,  "--", "^"),
        ]:
            info = qwen3_data[strategy_key]
            sizes  = np.array(info["sizes"])
            values = np.array(info["metrics"][metric_key])

            if len(sizes) == 0:
                continue

            ax.plot(
                sizes, values,
                color=color, linestyle=ls, marker=marker,
                label=info["label"],
                zorder=3,
            )

            # Log-linear trend line
            if len(sizes) >= 3:
                log_x = np.log10(sizes)
                coeffs = np.polyfit(log_x, values, 1)
                x_fit = np.logspace(np.log10(sizes.min() * 0.8),
                                    np.log10(sizes.max() * 1.5), 200)
                y_fit = np.polyval(coeffs, np.log10(x_fit))
                ax.plot(x_fit, y_fit, color=color, linestyle=ls,
                        alpha=0.3, linewidth=1.5, zorder=2)

        ax.set_xscale("log")
        ax.set_xlabel("Model Parameters (B)", fontsize=14)
        ax.set_ylabel(metric_label, fontsize=14)
        ax.set_title(metric_label, fontsize=15, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(2, 400)

        # Custom x-tick labels  (drop 30 and 100 to avoid crowding near 32B)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, pos: f"{int(v)}B" if v >= 1 else f"{v}B")
        )
        ax.set_xticks([4, 8, 14, 32, 235])
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.15, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", frameon=True, framealpha=0.85)

        # Annotate actual data points
        for strategy_key, color, ls, marker in [
            ("direct",      COLOR_DIRECT, "-",  "o"),
            ("coder_agent", COLOR_AGENT,  "--", "^"),
        ]:
            info = qwen3_data[strategy_key]
            sizes  = np.array(info["sizes"])
            values = np.array(info["metrics"][metric_key])
            for sx, vy in zip(sizes, values):
                ax.annotate(
                    f"{vy:.2f}",
                    xy=(sx, vy),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=color,
                    alpha=0.85,
                )

    fig.suptitle(
        "Qwen3 Scaling Law on Tool-Genesis Benchmark",
        fontsize=18, fontweight="bold",
    )

    for ext in (".pdf", ".png"):
        out_path = PLOT_DIR / (out_stem + ext)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved figure: {out_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Building Qwen3 scaling law data …")
    print("=" * 60)
    qwen3_data = build_qwen3_data()

    # Save Qwen3 JSON
    out_json_q3 = DATA_DIR / "scaling_law_qwen3.json"
    with open(out_json_q3, "w") as f:
        json.dump(qwen3_data, f, indent=2)
    print(f"\nSaved Qwen3 data: {out_json_q3}")

    print("\n" + "=" * 60)
    print("Building all-models data …")
    print("=" * 60)
    all_models = build_all_models_data()

    out_json_all = DATA_DIR / "all_models_by_size.json"
    with open(out_json_all, "w") as f:
        json.dump(all_models, f, indent=2)
    print(f"Saved all-models data: {out_json_all}")

    # ---- Print summary table ----
    print("\n" + "=" * 60)
    print("Qwen3 Scaling Summary")
    print("=" * 60)
    header = f"{'Run':<45} {'Size':>6}  {'Launch':>7}  {'F1':>6}  {'UT-soft':>8}  {'SR-soft':>8}"
    print(header)
    print("-" * len(header))
    for strategy_key in ("direct", "coder_agent"):
        info = qwen3_data[strategy_key]
        label = info["label"]
        sizes  = info["sizes"]
        m      = info["metrics"]
        for i, (run_name, _) in enumerate(QWEN3_RUNS):
            if i >= len(sizes):
                continue
            prefix = "direct" if strategy_key == "direct" else "coder_agent"
            tag = f"[{label}] {prefix}_{run_name}"
            print(
                f"{tag:<45} {sizes[i]:>5}B  "
                f"{m['L1_launch'][i]:>7.3f}  "
                f"{m['L2_schema_f1'][i]:>6.3f}  "
                f"{m['L3_ut_soft'][i]:>8.3f}  "
                f"{m['L4_sr_soft'][i]:>8.3f}"
            )
        print()

    print("\n" + "=" * 60)
    print("Generating figure …")
    print("=" * 60)
    plot_scaling_law(qwen3_data, "scaling_law_qwen3")
    print("\nDone.")


if __name__ == "__main__":
    main()
