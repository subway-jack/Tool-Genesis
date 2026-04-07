"""
Generate Figure: First-Fail Gate Distribution + Conditional Funnel.
For Tool-Genesis paper Section 6 rewrite.
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

eval_dir = "temp/eval_results_v3"

# Load all results
all_results = {}
for name in os.listdir(eval_dir):
    rpath = os.path.join(eval_dir, name, "results.json")
    if not os.path.exists(rpath):
        continue
    with open(rpath) as f:
        data = json.load(f)
    if data:
        all_results[name] = data

def classify_first_fail(metrics):
    if not metrics.get("compliance"):
        return "L1: Format"
    if metrics.get("server_launch_success", 0) == 0:
        return "L1: Launch"
    if metrics.get("schema_f1", 0) < 0.1:
        return "L2: Schema"
    if metrics.get("tool_call_success_rate_soft", 0) < 0.05:
        return "L3: Unit Test"
    if metrics.get("trajectory_level_validation_rate_soft", 0) < 0.1:
        return "L4: Task"
    return "Pass All"

GATE_ORDER = ["L1: Format", "L1: Launch", "L2: Schema", "L3: Unit Test", "L4: Task", "Pass All"]
GATE_COLORS = ["#e74c3c", "#e67e22", "#f39c12", "#3498db", "#9b59b6", "#2ecc71"]

# =====================================================================
# Figure 1: Aggregate first-fail distribution (Direct vs Coder-Agent)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for idx, strategy in enumerate(["direct", "coder_agent"]):
    runs = {k: v for k, v in all_results.items() if k.startswith(strategy + "_")}
    counts = {g: 0 for g in GATE_ORDER}
    total = 0
    for data in runs.values():
        for d in data:
            gate = classify_first_fail(d["metrics"])
            counts[gate] += 1
            total += 1
    
    pcts = [counts[g] / total * 100 for g in GATE_ORDER]
    bars = axes[idx].barh(GATE_ORDER[::-1], pcts[::-1], color=GATE_COLORS[::-1], edgecolor='white', linewidth=0.5)
    axes[idx].set_xlabel("Percentage (%)", fontsize=12)
    title = "Direct (Single-Pass)" if strategy == "direct" else "Code-Agent (Iterative)"
    axes[idx].set_title(title, fontsize=14, fontweight='bold')
    axes[idx].set_xlim(0, 80)
    
    for bar, pct in zip(bars, pcts[::-1]):
        if pct > 3:
            axes[idx].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                          f'{pct:.1f}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig("scripts/plot/first_fail_distribution.pdf", bbox_inches='tight', dpi=300)
plt.savefig("scripts/plot/first_fail_distribution.png", bbox_inches='tight', dpi=300)
print("Saved: scripts/plot/first_fail_distribution.pdf")

# =====================================================================
# Figure 2: Conditional funnel P(Lk|Lk-1) per model (Coder-Agent only)
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 6))

model_order = {
    "qwen3-4b": 4, "qwen3-8b": 8, "qwen3-14b": 14,
    "qwen3-32b": 32, "qwen3-235b-a22b-instruct-2507": 235,
    "openai_gpt-4-1-mini": 70, "openai_gpt-4o": 200,
    "openai_gpt-4-1": 200, "openai_gpt-5-1": 300,
    "anthropic_claude-3-5-haiku": 20,
    "google_gemini-3-flash-preview": 100,
    "deepseek_deepseek-v3-2": 671,
    "moonshotai_kimi-k2": 1000,
}

strategy = "coder_agent"
runs = {k: v for k, v in all_results.items() if k.startswith(strategy + "_")}

funnel_data = []
for run_name, data in sorted(runs.items()):
    model = run_name.replace(f"{strategy}_", "")
    n = len(data)
    l1_pass = [d for d in data if d["metrics"].get("compliance") and d["metrics"].get("server_launch_success", 0) > 0]
    l2_pass = [d for d in l1_pass if d["metrics"].get("schema_f1", 0) >= 0.3]
    l3_pass = [d for d in l2_pass if d["metrics"].get("tool_call_success_rate_soft", 0) >= 0.1]
    l4_pass = [d for d in l3_pass if d["metrics"].get("trajectory_level_validation_rate_soft", 0) >= 0.1]
    
    p_l1 = len(l1_pass) / n if n else 0
    p_l2_l1 = len(l2_pass) / len(l1_pass) if l1_pass else 0
    p_l3_l2 = len(l3_pass) / len(l2_pass) if l2_pass else 0
    p_l4_l3 = len(l4_pass) / len(l3_pass) if l3_pass else 0
    
    funnel_data.append({
        "model": model, "size": model_order.get(model, 0),
        "P(L1)": p_l1, "P(L2|L1)": p_l2_l1, "P(L3|L2)": p_l3_l2, "P(L4|L3)": p_l4_l3,
        "n": n,
    })

# Sort by model size
funnel_data.sort(key=lambda x: x["size"])

models = [d["model"].replace("_", " ").replace("a22b-instruct-2507", "").replace("3-flash-preview","3-flash")[:20] for d in funnel_data]
x = np.arange(len(models))
width = 0.2

colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
labels = ["P(L1 pass)", "P(L2|L1)", "P(L3|L2)", "P(L4|L3)"]
keys = ["P(L1)", "P(L2|L1)", "P(L3|L2)", "P(L4|L3)"]

for i, (key, color, label) in enumerate(zip(keys, colors, labels)):
    vals = [d[key] for d in funnel_data]
    ax.bar(x + i*width, vals, width, label=label, color=color, edgecolor='white', linewidth=0.5)

ax.set_ylabel("Conditional Pass Rate", fontsize=12)
ax.set_title("Conditional Funnel: Code-Agent Strategy", fontsize=14, fontweight='bold')
ax.set_xticks(x + 1.5*width)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=10, loc='upper left')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("scripts/plot/conditional_funnel.pdf", bbox_inches='tight', dpi=300)
plt.savefig("scripts/plot/conditional_funnel.png", bbox_inches='tight', dpi=300)
print("Saved: scripts/plot/conditional_funnel.pdf")

# =====================================================================
# Figure 3: Cumulative pass-through (stacked area)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, strategy in enumerate(["direct", "coder_agent"]):
    runs = {k: v for k, v in all_results.items() if k.startswith(strategy + "_")}
    
    model_data = []
    for run_name, data in sorted(runs.items()):
        model = run_name.replace(f"{strategy}_", "")
        n = len(data)
        pcts = []
        for gate in GATE_ORDER:
            count = sum(1 for d in data if classify_first_fail(d["metrics"]) == gate)
            pcts.append(count / n * 100)
        model_data.append({"model": model, "size": model_order.get(model, 0), "pcts": pcts})
    
    model_data.sort(key=lambda x: x["size"])
    models_short = [d["model"][:15] for d in model_data]
    
    bottom = np.zeros(len(model_data))
    for gi, gate in enumerate(GATE_ORDER):
        vals = [d["pcts"][gi] for d in model_data]
        axes[idx].bar(range(len(model_data)), vals, bottom=bottom, 
                     color=GATE_COLORS[gi], label=gate if idx == 0 else "", edgecolor='white', linewidth=0.3)
        bottom += vals
    
    axes[idx].set_ylabel("Percentage (%)", fontsize=11)
    title = "Direct" if strategy == "direct" else "Code-Agent"
    axes[idx].set_title(title, fontsize=13, fontweight='bold')
    axes[idx].set_xticks(range(len(model_data)))
    axes[idx].set_xticklabels(models_short, rotation=55, ha='right', fontsize=8)

axes[0].legend(fontsize=9, loc='upper left', bbox_to_anchor=(0, 1.0))
plt.tight_layout()
plt.savefig("scripts/plot/first_fail_stacked.pdf", bbox_inches='tight', dpi=300)
plt.savefig("scripts/plot/first_fail_stacked.png", bbox_inches='tight', dpi=300)
print("Saved: scripts/plot/first_fail_stacked.pdf")

# Save data for paper
with open("data/first_fail_funnel.json", "w") as f:
    json.dump({"funnel_coder_agent": funnel_data}, f, indent=2)
print("Saved: data/first_fail_funnel.json")

# =====================================================================
# P5: Schema-F1 sensitivity table
# =====================================================================
print("\n=== Schema-F1 Sensitivity Analysis ===")
print("(For paper appendix)")
print(f"{'Threshold':>10} | {'MiniLM Mean':>12} | {'Model Spread':>13} | {'Note':>20}")
print("-" * 65)
# Data from our earlier analysis
sensitivity = [
    (0.3, 0.850, 0.065, "Current default"),
    (0.4, 0.806, 0.075, ""),
    (0.5, 0.687, 0.133, "Better discrimination"),
    (0.6, 0.579, 0.174, "Best discrimination"),
]
for t, mean, spread, note in sensitivity:
    print(f"{t:>10.1f} | {mean:>12.3f} | {spread:>13.3f} | {note:>20}")
