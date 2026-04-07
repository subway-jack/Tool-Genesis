"""
Figure: Domain Difficulty Heatmap.
Rows = domains (primary_label), sorted by average SR across selected models.
Columns = selected Code-Agent models.
Cell value = average trajectory_level_validation_rate_soft for that domain × model.
Color = RdYlGn.

For Tool-Genesis ICML paper.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 1.  Configuration
# ---------------------------------------------------------------------------
GT_PATH        = "data/tool_genesis_v3.json"
EVAL_DIR       = "temp/eval_results_v3"
OUT_DIR        = "scripts/plot"
METRIC         = "trajectory_level_validation_rate_soft"

# Selected models (Code-Agent strategy only).
# Keys are the run-name suffixes after "coder_agent_".
SELECTED_MODELS = {
    "qwen3-8b":                        "Qwen3-8B",
    "qwen3-32b":                       "Qwen3-32B",
    "qwen3-235b-a22b-instruct-2507":   "Qwen3-235B",
    "openai_gpt-4-1":                  "GPT-4.1",
    "openai_gpt-5-1":                  "GPT-5.1",
    "google_gemini-3-flash-preview":   "Gemini-3-Flash",
    "moonshotai_kimi-k2":              "Kimi-K2",
    "deepseek_deepseek-v3-2":          "DeepSeek-V3.2",
}

# ---------------------------------------------------------------------------
# 2.  Build slug → primary_label mapping from GT data
# ---------------------------------------------------------------------------
print("Loading GT data …")
with open(GT_PATH) as f:
    gt_data = json.load(f)

slug_to_domain: dict[str, str] = {}
for item in gt_data:
    slug = item.get("server_slug") or item.get("server_name")
    domain = item.get("primary_label", "Others")
    if slug:
        slug_to_domain[slug] = domain

print(f"  GT entries: {len(gt_data)}, unique domains: {len(set(slug_to_domain.values()))}")

# ---------------------------------------------------------------------------
# 3.  Load eval results for selected Code-Agent runs
# ---------------------------------------------------------------------------
# data[model_key][server_slug] = metric_value
model_results: dict[str, dict[str, float]] = {}

for model_key in SELECTED_MODELS:
    run_name  = f"coder_agent_{model_key}"
    rpath     = os.path.join(EVAL_DIR, run_name, "results.json")
    if not os.path.exists(rpath):
        print(f"  WARNING: missing {rpath}")
        continue
    with open(rpath) as f:
        records = json.load(f)
    slug_vals: dict[str, float] = {}
    for rec in records:
        slug = rec.get("server_slug")
        val  = rec.get("metrics", {}).get(METRIC)
        if slug and val is not None:
            slug_vals[slug] = float(val)
    model_results[model_key] = slug_vals
    print(f"  Loaded {run_name}: {len(slug_vals)} servers")

# Keep only models that loaded successfully, in specification order
loaded_model_keys = [k for k in SELECTED_MODELS if k in model_results]

# ---------------------------------------------------------------------------
# 4.  Aggregate per-domain SR for each model
# ---------------------------------------------------------------------------
# Collect all domains that appear across the loaded runs
all_domains: set[str] = set()
for slug in slug_to_domain:
    for model_key in loaded_model_keys:
        if slug in model_results.get(model_key, {}):
            all_domains.add(slug_to_domain[slug])

# Build domain × model matrix
domain_list = sorted(all_domains)
matrix: dict[str, dict[str, list[float]]] = {d: {m: [] for m in loaded_model_keys} for d in domain_list}

for model_key in loaded_model_keys:
    for slug, val in model_results[model_key].items():
        domain = slug_to_domain.get(slug)
        if domain and domain in matrix:
            matrix[domain][model_key].append(val)

# Average per cell
avg_matrix = np.full((len(domain_list), len(loaded_model_keys)), np.nan)
for di, domain in enumerate(domain_list):
    for mi, model_key in enumerate(loaded_model_keys):
        vals = matrix[domain][model_key]
        if vals:
            avg_matrix[di, mi] = np.mean(vals)

# Sort domains by average SR across models (descending = easiest first → hardest last)
domain_avg = np.nanmean(avg_matrix, axis=1)
sort_idx   = np.argsort(domain_avg)          # ascending: hardest at top
sorted_domains   = [domain_list[i] for i in sort_idx]
sorted_avg_matrix = avg_matrix[sort_idx, :]

# ---------------------------------------------------------------------------
# 5.  Plot heatmap
# ---------------------------------------------------------------------------
col_labels = [SELECTED_MODELS[k] for k in loaded_model_keys]
n_rows, n_cols = sorted_avg_matrix.shape

fig_h = max(5, n_rows * 0.55 + 1.5)
fig_w = max(8, n_cols * 1.2 + 2.5)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

hm = sns.heatmap(
    sorted_avg_matrix,
    ax=ax,
    cmap="RdYlGn",
    vmin=0.0,
    vmax=1.0,
    linewidths=0.5,
    linecolor="white",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 8, "weight": "bold"},
    cbar_kws={"label": "Avg SR (trajectory soft)", "shrink": 0.75},
    xticklabels=col_labels,
    yticklabels=sorted_domains,
)

ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9.5, fontweight="bold")
ax.set_yticklabels(sorted_domains, rotation=0, fontsize=9)
ax.set_title(
    "Domain Difficulty Heatmap\n(Code-Agent strategy · avg SR by domain × model)",
    fontsize=12, fontweight="bold", pad=12,
)
ax.set_xlabel("Model", fontsize=10, labelpad=8)
ax.set_ylabel("Domain  (sorted hardest → easiest)", fontsize=10, labelpad=8)

plt.tight_layout()

os.makedirs(OUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUT_DIR, "domain_difficulty_heatmap.pdf")
png_path = os.path.join(OUT_DIR, "domain_difficulty_heatmap.png")
plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
plt.savefig(png_path, bbox_inches="tight", dpi=300)
print(f"\nSaved: {pdf_path}")
print(f"Saved: {png_path}")

# ---------------------------------------------------------------------------
# 6.  Print domain ranking table
# ---------------------------------------------------------------------------
print("\n=== Domain Difficulty Ranking (hardest → easiest) ===")
print(f"{'Domain':<35s}  {'Avg SR':>8s}  {'N servers':>10s}")
print("-" * 60)
for di, domain in enumerate(sorted_domains):
    avg = domain_avg[sort_idx[di]]
    n   = sum(len(matrix[domain][m]) for m in loaded_model_keys) // max(1, len(loaded_model_keys))
    print(f"{domain:<35s}  {avg:8.4f}  {n:>10d}")
