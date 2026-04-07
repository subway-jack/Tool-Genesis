#!/usr/bin/env python3
"""
Bootstrap confidence intervals for Tool-Genesis benchmark paper.

Outputs:
  data/bootstrap_ci_metrics.json   – per-run mean + 95 % CI for 6 metrics
  data/bootstrap_ci_funnel.json    – conditional funnel P(Lk | Lk-1) with CIs
  data/bottleneck_shift_test.json  – chi-squared significance test on first-fail distribution
"""

from __future__ import annotations

import json
import os
import glob
import numpy as np
from collections import defaultdict

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = "/Users/subway/code/python/项目/Multi-agent/tool-genesis"
EVAL_DIR = os.path.join(BASE_DIR, "temp/eval_results_v3")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

N_BOOT = 10_000
SEED = 42
rng = np.random.default_rng(SEED)

METRICS = [
    "compliance",
    "server_launch_success",
    "schema_f1",
    "tool_call_success_rate_soft",
    "tool_call_success_rate_hard",
    "trajectory_level_validation_rate_soft",
]
METRIC_LABELS = [
    "Compliance",
    "Server_Launch",
    "Schema_F1",
    "UT_soft",
    "UT_hard",
    "SR",
]

FUNNEL_GATES = [
    "L1_compliance",
    "L1_launch",
    "L2_schema",
    "L3_unit_test",
    "L4_task",
    "PASS_all",
]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def first_fail_label(m: dict) -> str:
    """Map a single server's metrics dict to a first-fail gate label."""
    if not m.get("compliance", False):
        return "L1_compliance"
    if m.get("server_launch_success", 0) == 0:
        return "L1_launch"
    if m.get("schema_f1", 0.0) < 0.1:
        return "L2_schema"
    if m.get("tool_call_success_rate_soft", 0.0) < 0.05:
        return "L3_unit_test"
    if m.get("trajectory_level_validation_rate_soft", 0.0) < 0.1:
        return "L4_task"
    return "PASS_all"


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, rng) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) using percentile bootstrap."""
    n = len(values)
    observed = float(np.mean(values))
    if n == 0:
        return observed, float("nan"), float("nan")
    boot_means = np.array([
        np.mean(values[rng.integers(0, n, size=n)])
        for _ in range(n_boot)
    ])
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return observed, ci_low, ci_high


def bootstrap_proportion_ci(counts: np.ndarray, total: int, n_boot: int, rng):
    """
    Bootstrap CI for a proportion vector.
    counts: array of shape (n_servers,) with label indices (0..K-1)
    Returns dict of label -> (mean, ci_low, ci_high).
    """
    n = len(counts)
    K = len(FUNNEL_GATES)

    def gate_props(sample):
        props = np.zeros(K)
        for k in range(K):
            props[k] = np.mean(sample == k)
        return props

    observed = gate_props(counts)
    boot_props = np.array([
        gate_props(counts[rng.integers(0, n, size=n)])
        for _ in range(n_boot)
    ])
    ci_low = np.percentile(boot_props, 2.5, axis=0)
    ci_high = np.percentile(boot_props, 97.5, axis=0)
    return {
        FUNNEL_GATES[k]: {
            "mean": float(observed[k]),
            "ci_low": float(ci_low[k]),
            "ci_high": float(ci_high[k]),
        }
        for k in range(K)
    }


# ──────────────────────────────────────────────
# Load all runs
# ──────────────────────────────────────────────
result_files = sorted(glob.glob(os.path.join(EVAL_DIR, "*/results.json")))
print(f"Found {len(result_files)} result files.\n")

runs: dict[str, list[dict]] = {}
for path in result_files:
    run_name = os.path.basename(os.path.dirname(path))
    with open(path) as f:
        data = json.load(f)
    runs[run_name] = [entry["metrics"] for entry in data]


def parse_run_name(name: str) -> tuple[str, str]:
    """Split 'strategy_model' from run directory name."""
    if name.startswith("coder_agent_"):
        strategy = "coder_agent"
        model = name[len("coder_agent_"):]
    elif name.startswith("direct_"):
        strategy = "direct"
        model = name[len("direct_"):]
    else:
        strategy = "unknown"
        model = name
    return strategy, model


# ──────────────────────────────────────────────
# 1. Per-run metric CIs
# ──────────────────────────────────────────────
print("=" * 90)
print(f"{'Model':<45} {'Strategy':<12} {'Metric':<25} {'Mean':>7} {'CI_low':>8} {'CI_high':>8}")
print("=" * 90)

ci_metrics: dict[str, dict] = {}

for run_name in sorted(runs):
    strategy, model = parse_run_name(run_name)
    metrics_list = runs[run_name]
    n = len(metrics_list)
    ci_metrics[run_name] = {"strategy": strategy, "model": model, "n_servers": n, "metrics": {}}

    for metric_key, metric_label in zip(METRICS, METRIC_LABELS):
        values = np.array([
            float(m.get(metric_key, 0.0)) if m.get(metric_key) is not None else 0.0
            for m in metrics_list
        ])
        mean, ci_low, ci_high = bootstrap_mean_ci(values, N_BOOT, rng)
        ci_metrics[run_name]["metrics"][metric_label] = {
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
        print(f"{model:<45} {strategy:<12} {metric_label:<25} {mean:>7.4f} {ci_low:>8.4f} {ci_high:>8.4f}")
    print()

# Save
out_path = os.path.join(DATA_DIR, "bootstrap_ci_metrics.json")
with open(out_path, "w") as f:
    json.dump(ci_metrics, f, indent=2)
print(f"\nSaved per-run metric CIs → {out_path}\n")


# ──────────────────────────────────────────────
# 2. First-fail analysis per strategy
# ──────────────────────────────────────────────
gate_index = {g: i for i, g in enumerate(FUNNEL_GATES)}

# Collect per-strategy arrays of first-fail labels (as indices)
strategy_gate_indices: dict[str, list[int]] = defaultdict(list)
# Also per-run for per-run first-fail
run_gate_indices: dict[str, np.ndarray] = {}

for run_name, metrics_list in runs.items():
    strategy, model = parse_run_name(run_name)
    indices = np.array([gate_index[first_fail_label(m)] for m in metrics_list])
    run_gate_indices[run_name] = indices
    strategy_gate_indices[strategy].extend(indices.tolist())

strategy_gate_arrays: dict[str, np.ndarray] = {
    s: np.array(v) for s, v in strategy_gate_indices.items()
}

print("=" * 60)
print("First-fail distribution (per strategy, bootstrap 95% CI)")
print("=" * 60)

first_fail_results: dict[str, dict] = {}

for strategy, arr in sorted(strategy_gate_arrays.items()):
    print(f"\nStrategy: {strategy}  (n={len(arr)} server-runs)")
    ff_ci = bootstrap_proportion_ci(arr, len(arr), N_BOOT, rng)
    first_fail_results[strategy] = {"n": len(arr), "gates": ff_ci}
    for gate in FUNNEL_GATES:
        d = ff_ci[gate]
        print(f"  {gate:<18}: {d['mean']:.4f}  [{d['ci_low']:.4f}, {d['ci_high']:.4f}]")

# Also compute per-run first-fail for the bottleneck test
# (merged into first_fail_results for reference)
first_fail_results["per_run"] = {}
for run_name in sorted(runs):
    strategy, model = parse_run_name(run_name)
    arr = run_gate_indices[run_name]
    ff_ci = bootstrap_proportion_ci(arr, len(arr), N_BOOT, rng)
    first_fail_results["per_run"][run_name] = {
        "strategy": strategy,
        "model": model,
        "n": len(arr),
        "gates": ff_ci,
    }


# ──────────────────────────────────────────────
# 3. Conditional funnel P(Lk | Lk-1)
# ──────────────────────────────────────────────
#
# Gate sequence for conditional funnel:
#   Lc  = compliance gate       P(pass L1_compliance)
#   Ll  = launch gate           P(pass L1_launch     | passed Lc)
#   Ls  = schema gate           P(pass L2_schema     | passed Ll)
#   Lu  = unit-test gate        P(pass L3_unit_test  | passed Ls)
#   Lt  = task gate             P(pass L4_task       | passed Lu)
#
# We compute this per-run and per-strategy.

COND_GATES = [
    ("Compliance",    "compliance",                          None,                     None),
    ("Launch | Comp", "server_launch_success",               "compliance",             None),
    ("Schema | Launch","schema_f1",                          "server_launch_success",  0.1),
    ("UT | Schema",   "tool_call_success_rate_soft",         "schema_f1",              0.05),
    ("SR | UT",       "trajectory_level_validation_rate_soft","tool_call_success_rate_soft", 0.1),
]


def compute_conditional_funnel(metrics_list: list[dict]) -> dict:
    """
    Returns dict: gate_label -> {"n_eligible", "n_pass", "p_cond"} as arrays
    for bootstrap.
    """
    # Convert to numpy for vectorised ops
    arr_comp   = np.array([float(m.get("compliance", False)) for m in metrics_list])
    arr_launch = np.array([float(m.get("server_launch_success", 0)) for m in metrics_list])
    arr_schema = np.array([float(m.get("schema_f1", 0.0)) for m in metrics_list])
    arr_ut     = np.array([float(m.get("tool_call_success_rate_soft", 0.0)) for m in metrics_list])
    arr_sr     = np.array([float(m.get("trajectory_level_validation_rate_soft", 0.0)) for m in metrics_list])

    pass_comp   = arr_comp > 0.5
    pass_launch = arr_launch > 0.5
    pass_schema = arr_schema >= 0.1
    pass_ut     = arr_ut >= 0.05
    pass_sr     = arr_sr >= 0.1

    # Eligibility mask for each conditional gate
    funnel = {
        "Compliance":     {"mask": np.ones(len(metrics_list), dtype=bool), "pass_arr": pass_comp},
        "Launch|Comp":    {"mask": pass_comp,                               "pass_arr": pass_launch},
        "Schema|Launch":  {"mask": pass_comp & pass_launch,                 "pass_arr": pass_schema},
        "UT|Schema":      {"mask": pass_comp & pass_launch & pass_schema,   "pass_arr": pass_ut},
        "SR|UT":          {"mask": pass_comp & pass_launch & pass_schema & pass_ut, "pass_arr": pass_sr},
    }
    return funnel


def bootstrap_conditional_funnel(metrics_list: list[dict], n_boot: int, rng):
    """Bootstrap CIs for conditional funnel probabilities."""
    n = len(metrics_list)
    funnel_def = compute_conditional_funnel(metrics_list)

    results = {}
    for gate_label, gd in funnel_def.items():
        mask = gd["mask"]
        pass_arr = gd["pass_arr"]
        # Observed
        n_elig = int(np.sum(mask))
        n_pass = int(np.sum(mask & pass_arr))
        p_obs = n_pass / n_elig if n_elig > 0 else float("nan")

        # Bootstrap: resample servers (indices), then compute conditional rate
        boot_ps = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            m_boot = mask[idx]
            p_boot = pass_arr[idx]
            ne = int(np.sum(m_boot))
            np_b = int(np.sum(m_boot & p_boot))
            boot_ps.append(np_b / ne if ne > 0 else float("nan"))

        boot_arr = np.array(boot_ps, dtype=float)
        valid = boot_arr[~np.isnan(boot_arr)]
        ci_low  = float(np.percentile(valid, 2.5))  if len(valid) > 0 else float("nan")
        ci_high = float(np.percentile(valid, 97.5)) if len(valid) > 0 else float("nan")

        results[gate_label] = {
            "n_eligible": n_elig,
            "n_pass": n_pass,
            "p_cond": p_obs,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return results


print("\n\n" + "=" * 90)
print("Conditional funnel P(Lk | Lk-1)")
print("=" * 90)

funnel_results: dict[str, dict] = {}

# Per-run
funnel_results["per_run"] = {}
for run_name in sorted(runs):
    strategy, model = parse_run_name(run_name)
    metrics_list = runs[run_name]
    funnnel_ci = bootstrap_conditional_funnel(metrics_list, N_BOOT, rng)
    funnel_results["per_run"][run_name] = {
        "strategy": strategy,
        "model": model,
        "n_servers": len(metrics_list),
        "funnel": funnnel_ci,
    }
    print(f"\n{model}  [{strategy}]")
    for gate, d in funnnel_ci.items():
        print(f"  {gate:<16}: p={d['p_cond']:.4f}  [{d['ci_low']:.4f}, {d['ci_high']:.4f}]  "
              f"(n_elig={d['n_eligible']}, n_pass={d['n_pass']})")

# Per-strategy (pool all servers across runs of same strategy)
funnel_results["per_strategy"] = {}
for strategy in sorted(strategy_gate_arrays.keys()):
    # Gather all metrics from all runs of this strategy
    all_metrics = []
    for run_name, metrics_list in runs.items():
        s, _ = parse_run_name(run_name)
        if s == strategy:
            all_metrics.extend(metrics_list)
    funnnel_ci = bootstrap_conditional_funnel(all_metrics, N_BOOT, rng)
    funnel_results["per_strategy"][strategy] = {
        "n_server_runs": len(all_metrics),
        "funnel": funnnel_ci,
    }
    print(f"\n[STRATEGY] {strategy}  (n_server_runs={len(all_metrics)})")
    for gate, d in funnnel_ci.items():
        print(f"  {gate:<16}: p={d['p_cond']:.4f}  [{d['ci_low']:.4f}, {d['ci_high']:.4f}]  "
              f"(n_elig={d['n_eligible']}, n_pass={d['n_pass']})")

# Save funnel
out_path_funnel = os.path.join(DATA_DIR, "bootstrap_ci_funnel.json")
with open(out_path_funnel, "w") as f:
    json.dump(funnel_results, f, indent=2)
print(f"\nSaved conditional funnel CIs → {out_path_funnel}\n")


# ──────────────────────────────────────────────
# 4. Bottleneck shift test: Direct vs Coder-Agent
#    Chi-squared on first-fail contingency table
#    + permutation test
# ──────────────────────────────────────────────
from scipy.stats import chi2_contingency  # type: ignore

direct_arr = strategy_gate_arrays.get("direct", np.array([], dtype=int))
coder_arr  = strategy_gate_arrays.get("coder_agent", np.array([], dtype=int))

K = len(FUNNEL_GATES)

def counts_vector(arr: np.ndarray) -> np.ndarray:
    counts = np.zeros(K, dtype=int)
    for k in range(K):
        counts[k] = int(np.sum(arr == k))
    return counts

direct_counts = counts_vector(direct_arr)
coder_counts  = counts_vector(coder_arr)

# Contingency table: shape (2, K)
contingency = np.vstack([direct_counts, coder_counts])
print("\n" + "=" * 60)
print("First-fail contingency table")
print("=" * 60)
print(f"{'Gate':<18} {'Direct':>8} {'Coder-Agent':>12}")
for k, gate in enumerate(FUNNEL_GATES):
    print(f"{gate:<18} {direct_counts[k]:>8} {coder_counts[k]:>12}")

# Drop zero-column gates before chi2 to avoid expected-frequency=0 error
nonzero_mask = (direct_counts + coder_counts) > 0
contingency_nz = contingency[:, nonzero_mask]
gates_nz = [g for g, m in zip(FUNNEL_GATES, nonzero_mask) if m]
print(f"\nGates used in chi2 (non-zero columns): {gates_nz}")
chi2, p_chi2, dof, expected = chi2_contingency(contingency_nz)
print(f"Chi-squared: {chi2:.4f}  df={dof}  p={p_chi2:.6f}")

# Permutation test
combined = np.concatenate([direct_arr, coder_arr])
n_direct = len(direct_arr)
n_coder  = len(coder_arr)

def chi2_stat(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
    """Chi2 on non-zero columns only to avoid expected=0 error."""
    cv_a = counts_vector(arr_a)
    cv_b = counts_vector(arr_b)
    ct = np.vstack([cv_a, cv_b])
    # Keep only columns where any count > 0
    nz = (ct.sum(axis=0)) > 0
    ct_nz = ct[:, nz]
    try:
        stat, _, _, _ = chi2_contingency(ct_nz)
        return float(stat)
    except Exception:
        return float("nan")

observed_chi2 = chi2_stat(direct_arr, coder_arr)
n_perm = 10_000
perm_stats = np.zeros(n_perm)
for i in range(n_perm):
    shuffled = rng.permutation(combined)
    perm_stats[i] = chi2_stat(shuffled[:n_direct], shuffled[n_direct:])

valid_perm = perm_stats[~np.isnan(perm_stats)]
p_perm = float(np.mean(valid_perm >= observed_chi2))
print(f"Permutation test (n={n_perm}): observed chi2={observed_chi2:.4f}  p_perm={p_perm:.6f}")

# Per-gate proportions with CIs for both strategies
print("\nPer-gate proportions with 95% CI:")
direct_ci = bootstrap_proportion_ci(direct_arr, len(direct_arr), N_BOOT, rng)
coder_ci  = bootstrap_proportion_ci(coder_arr,  len(coder_arr),  N_BOOT, rng)

print(f"{'Gate':<18} {'Direct mean':>12} {'Direct CI':>22} {'Coder mean':>12} {'Coder CI':>22}")
for gate in FUNNEL_GATES:
    dc = direct_ci[gate]
    cc = coder_ci[gate]
    print(f"{gate:<18} {dc['mean']:>12.4f} [{dc['ci_low']:.4f},{dc['ci_high']:.4f}]   "
          f"{cc['mean']:>12.4f} [{cc['ci_low']:.4f},{cc['ci_high']:.4f}]")

bottleneck_test = {
    "description": "Chi-squared test + permutation test for difference in first-fail distribution between Direct and Coder-Agent strategies",
    "n_direct": int(len(direct_arr)),
    "n_coder_agent": int(len(coder_arr)),
    "gates": FUNNEL_GATES,
    "direct_counts": direct_counts.tolist(),
    "coder_agent_counts": coder_counts.tolist(),
    "direct_proportions_ci": direct_ci,
    "coder_agent_proportions_ci": coder_ci,
    "chi2_test": {
        "statistic": float(chi2),
        "p_value": float(p_chi2),
        "df": int(dof),
        "gates_used": gates_nz,
        "note": "Zero-count columns (L4_task) excluded before chi2 computation",
        "observed": contingency_nz.tolist(),
        "expected": expected.tolist(),
    },
    "permutation_test": {
        "n_permutations": n_perm,
        "observed_chi2": float(observed_chi2),
        "p_value": float(p_perm),
    },
    "first_fail_per_strategy": first_fail_results,
}

out_path_test = os.path.join(DATA_DIR, "bottleneck_shift_test.json")
with open(out_path_test, "w") as f:
    json.dump(bottleneck_test, f, indent=2)
print(f"\nSaved bottleneck shift test → {out_path_test}")

print("\nAll done.")
