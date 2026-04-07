"""
Solver Agent Ablation: Compare SR rankings under different proxy agents.

Runs L4 trajectory evaluation with a different solver agent (GPT-4.1-mini)
on a representative subset of models, then computes rank correlation with
the default Qwen3-14B solver results.

Usage:
  # Set solver via env vars, then run evaluation for selected models
  SOLVER_PLATFORM=OPENAI SOLVER_MODEL=GPT_4_1_MINI \
    python scripts/experiment_journal/solver_ablation.py

Output:
  data/solver_ablation.json
"""

import json
import os
import sys
import subprocess
import argparse
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def get_existing_sr(results_dir: str, models: list) -> dict:
    """Load existing SR from l2_debug.json (Qwen3-14B solver)."""
    sr_map = {}
    for model_name in models:
        debug_dir = os.path.join(results_dir, model_name, "debug")
        if not os.path.isdir(debug_dir):
            continue
        sr_values = []
        for server in os.listdir(debug_dir):
            l2_path = os.path.join(debug_dir, server, "l2_debug.json")
            if not os.path.exists(l2_path):
                continue
            with open(l2_path) as f:
                l2 = json.load(f)
            traj = l2.get("trajectory", {})
            sr_values.append(traj.get("soft_avg", 0))
        if sr_values:
            sr_map[model_name] = sum(sr_values) / 86  # normalized to 86
    return sr_map


def run_eval_with_solver(
    model_name: str,
    solver_platform: str,
    solver_model: str,
    results_dir: str,
    out_dir: str,
    limit: int = 20,
):
    """Run evaluation with alternative solver agent."""
    env = {
        **os.environ,
        "SOLVER_PLATFORM": solver_platform,
        "SOLVER_MODEL": solver_model,
    }
    cmd = [
        sys.executable, "-u",
        str(PROJECT_ROOT / "scripts" / "run_benchmark" / "run_evaluation.py"),
        "--pred-path", str(PROJECT_ROOT / "temp" / "run_benchmark_v3"),
        "--out-root", out_dir,
        "--project-name", model_name,
        "--skip-l3",  # Skip L3 only; keep L1+L2 (trajectory is inside L2)
        "--workers", "1",
        "--reset",  # Force re-evaluation
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"  Running {model_name} with {solver_platform}/{solver_model}...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"  WARNING: {model_name} failed: {result.stderr[:200]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-platform", default="OPENAI")
    parser.add_argument("--solver-model", default="GPT_4_1_MINI")
    parser.add_argument("--results-dir", default="temp/eval_results_v3")
    parser.add_argument("--out-dir", default="temp/eval_results_v3_gpt_solver")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max servers per model (for speed)")
    args = parser.parse_args()

    # Representative subset: mix of Direct + CA, different model families
    models = [
        "direct_openai_gpt-4o",
        "direct_openai_gpt-5-1",
        "direct_qwen3-14b",
        "direct_qwen3-235b-a22b-instruct-2507",
        "direct_google_gemini-3-flash-preview",
        "direct_deepseek_deepseek-v3-2",
        "coder_agent_openai_gpt-5-1",
        "coder_agent_qwen3-14b",
        "coder_agent_qwen3-235b-a22b-instruct-2507",
        "coder_agent_google_gemini-3-flash-preview",
        "coder_agent_deepseek_deepseek-v3-2",
        "coder_agent_moonshotai_kimi-k2",
    ]

    # Step 1: Get existing SR (Qwen3-14B solver)
    print("Loading existing SR (Qwen3-14B solver)...")
    sr_qwen = get_existing_sr(args.results_dir, models)
    print(f"  Found SR for {len(sr_qwen)} models")

    # Step 2: Run eval with GPT solver
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nRunning L4 with {args.solver_platform}/{args.solver_model}...")
    for model_name in models:
        if model_name not in sr_qwen:
            print(f"  Skipping {model_name} (no baseline SR)")
            continue
        run_eval_with_solver(
            model_name, args.solver_platform, args.solver_model,
            args.results_dir, args.out_dir, args.limit,
        )

    # Step 3: Load GPT solver SR
    print("\nLoading GPT solver SR...")
    sr_gpt = get_existing_sr(args.out_dir, models)
    print(f"  Found SR for {len(sr_gpt)} models")

    # Step 4: Compute rank correlation
    common = sorted(set(sr_qwen.keys()) & set(sr_gpt.keys()))
    if len(common) < 3:
        print(f"Not enough common models ({len(common)}) for correlation")
        return

    qwen_vals = [sr_qwen[m] for m in common]
    gpt_vals = [sr_gpt[m] for m in common]

    spearman = stats.spearmanr(qwen_vals, gpt_vals)
    kendall = stats.kendalltau(qwen_vals, gpt_vals)

    print(f"\n=== Solver Ablation Results ===")
    print(f"Models compared: {len(common)}")
    print(f"Spearman ρ: {spearman.correlation:.3f} (p={spearman.pvalue:.4f})")
    print(f"Kendall τ:  {kendall.correlation:.3f} (p={kendall.pvalue:.4f})")
    print(f"\n{'Model':<45} {'Qwen3-14B':>10} {'GPT-4.1m':>10} {'Δ':>8}")
    print("-" * 78)
    for m in common:
        delta = sr_gpt[m] - sr_qwen[m]
        print(f"{m:<45} {sr_qwen[m]:>9.3f} {sr_gpt[m]:>9.3f} {delta:>+7.3f}")

    # Save
    output = {
        "solver_a": "Qwen3-14B (Bailian)",
        "solver_b": f"{args.solver_platform}/{args.solver_model}",
        "n_models": len(common),
        "spearman_rho": round(spearman.correlation, 4),
        "spearman_p": round(spearman.pvalue, 6),
        "kendall_tau": round(kendall.correlation, 4),
        "kendall_p": round(kendall.pvalue, 6),
        "per_model": {
            m: {"sr_qwen": round(sr_qwen[m], 4), "sr_gpt": round(sr_gpt[m], 4)}
            for m in common
        },
    }
    out_path = os.path.join(PROJECT_ROOT, "data", "solver_ablation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
