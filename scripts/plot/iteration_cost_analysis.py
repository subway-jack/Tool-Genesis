"""
Code-Agent Iteration & Cost Analysis for Tool-Genesis paper.
Two figures:
1. Attempts distribution + success by attempt count
2. Cost-benefit frontier: tokens vs SR
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Load step analysis data
with open("data/coder_agent_step_analysis.json") as f:
    step_data = json.load(f)

# Load eval results for success rates
eval_dir = "temp/eval_results_v3"
eval_results = {}
for name in os.listdir(eval_dir):
    rpath = os.path.join(eval_dir, name, "results.json")
    if not os.path.exists(rpath):
        continue
    with open(rpath) as f:
        data = json.load(f)
    if data:
        eval_results[name] = {d["server_slug"]: d for d in data}

# =====================================================================
# Figure 1: Cost-Benefit Frontier (tokens vs SR)
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 6))

model_sizes = {
    "qwen3-4b": 4, "qwen3-8b": 8, "qwen3-14b": 14,
    "qwen3-32b": 32, "qwen3-30b-a3b-instruct-2507": 30,
    "qwen3-235b-a22b-instruct-2507": 235,
    "openai_gpt-4o": 200, "openai_gpt-4-1-mini": 70,
    "openai_gpt-4-1": 200, "openai_gpt-5-1": 300,
    "anthropic_claude-3-5-haiku": 20,
    "google_gemini-3-flash-preview": 100,
    "deepseek_deepseek-v3-2": 671,
    "moonshotai_kimi-k2": 1000,
}

# Compute avg tokens per server for coder_agent models
tokens_by_model = defaultdict(list)
for r in step_data:
    tokens_by_model[r["model"]].append(r["est_tokens"])

# Get SR for each model×strategy
points_direct = []
points_coder = []

for model in model_sizes:
    # Direct: tokens ≈ single prompt+response, estimate ~2000 tokens
    direct_key = f"direct_{model}"
    coder_key = f"coder_agent_{model}"
    
    if direct_key in eval_results:
        data = eval_results[direct_key]
        sr = np.mean([d["metrics"].get("trajectory_level_validation_rate_soft", 0) for d in data.values()])
        # Estimate direct tokens (single call)
        points_direct.append((2000, sr, model))
    
    if coder_key in eval_results:
        data = eval_results[coder_key]
        sr = np.mean([d["metrics"].get("trajectory_level_validation_rate_soft", 0) for d in data.values()])
        avg_tokens = np.mean(tokens_by_model.get(model, [5000])) if model in tokens_by_model else 5000
        points_coder.append((avg_tokens, sr, model))

# Plot
for tokens, sr, model in points_direct:
    size = model_sizes.get(model, 50)
    ax.scatter(tokens, sr, s=max(30, size/3), c='#3498db', alpha=0.7, zorder=5, edgecolors='white', linewidth=0.5)

for tokens, sr, model in points_coder:
    size = model_sizes.get(model, 50)
    ax.scatter(tokens, sr, s=max(30, size/3), c='#e74c3c', alpha=0.7, zorder=5, edgecolors='white', linewidth=0.5)
    short = model.replace("qwen3-", "Q").replace("openai_gpt-", "GPT-").replace("google_gemini-", "Gem-").replace("a22b-instruct-2507","").replace("anthropic_claude-","Claude-").replace("deepseek_deepseek-","DS-").replace("moonshotai_","")[:12]
    ax.annotate(short, (tokens, sr), fontsize=7, ha='left', va='bottom', 
                xytext=(5, 3), textcoords='offset points')

# Draw arrows from Direct to Coder for matching models
for d_tok, d_sr, d_model in points_direct:
    for c_tok, c_sr, c_model in points_coder:
        if d_model == c_model:
            ax.annotate('', xy=(c_tok, c_sr), xytext=(d_tok, d_sr),
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=0.8))

ax.scatter([], [], c='#3498db', s=60, label='Direct (single-pass)')
ax.scatter([], [], c='#e74c3c', s=60, label='Code-Agent (iterative)')
ax.set_xlabel("Estimated Tokens per Server", fontsize=12)
ax.set_ylabel("L4 Success Rate", fontsize=12)
ax.set_title("Cost–Benefit Frontier: Direct vs Code-Agent", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 0.75)

plt.tight_layout()
plt.savefig("scripts/plot/cost_benefit_frontier.pdf", bbox_inches='tight', dpi=300)
plt.savefig("scripts/plot/cost_benefit_frontier.png", bbox_inches='tight', dpi=300)
print("Saved: scripts/plot/cost_benefit_frontier.pdf")

# =====================================================================
# Figure 2: Direct vs Code-Agent Performance Comparison (paired)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metric_pairs = [
    ("server_launch_success", "L1: Server Launch Rate"),
    ("schema_f1", "L2: Schema-F1"),
    ("trajectory_level_validation_rate_soft", "L4: Success Rate"),
]

for idx, (metric_key, metric_name) in enumerate(metric_pairs):
    ax = axes[idx]
    
    for model in sorted(model_sizes.keys()):
        d_key = f"direct_{model}"
        c_key = f"coder_agent_{model}"
        
        if d_key in eval_results and c_key in eval_results:
            d_val = np.mean([d["metrics"].get(metric_key, 0) for d in eval_results[d_key].values()])
            c_val = np.mean([d["metrics"].get(metric_key, 0) for d in eval_results[c_key].values()])
            
            size = model_sizes.get(model, 50)
            ax.scatter(d_val, c_val, s=max(40, size/2), c='#2ecc71', alpha=0.7, 
                      edgecolors='white', linewidth=0.5, zorder=5)
            short = model.replace("qwen3-", "Q").replace("openai_gpt-", "").replace("a22b-instruct-2507","")[:10]
            ax.annotate(short, (d_val, c_val), fontsize=7, ha='left', va='bottom',
                       xytext=(3, 3), textcoords='offset points')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel(f"Direct", fontsize=11)
    ax.set_ylabel(f"Code-Agent", fontsize=11)
    ax.set_title(metric_name, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle("Direct vs Code-Agent: Per-Model Comparison", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("scripts/plot/direct_vs_coder_scatter.pdf", bbox_inches='tight', dpi=300)
plt.savefig("scripts/plot/direct_vs_coder_scatter.png", bbox_inches='tight', dpi=300)
print("Saved: scripts/plot/direct_vs_coder_scatter.pdf")

# =====================================================================
# Data: Token efficiency summary
# =====================================================================
print("\n=== Token Efficiency Summary ===")
print(f"{'Model':>35} | {'Direct tokens':>14} | {'Coder tokens':>14} | {'D→C SR gain':>12} | {'Token ratio':>12}")
print("-" * 95)
for model in sorted(model_sizes.keys()):
    d_key = f"direct_{model}"
    c_key = f"coder_agent_{model}"
    if d_key in eval_results and c_key in eval_results:
        d_sr = np.mean([d["metrics"].get("trajectory_level_validation_rate_soft", 0) for d in eval_results[d_key].values()])
        c_sr = np.mean([d["metrics"].get("trajectory_level_validation_rate_soft", 0) for d in eval_results[c_key].values()])
        
        d_tokens = 2000  # estimate
        c_tokens = np.mean(tokens_by_model.get(model, [5000])) if model in tokens_by_model else 5000
        
        gain = c_sr - d_sr
        ratio = c_tokens / d_tokens if d_tokens > 0 else 0
        print(f"{model:>35} | {d_tokens:>14.0f} | {c_tokens:>14.0f} | {gain:>+12.3f} | {ratio:>12.1f}x")
