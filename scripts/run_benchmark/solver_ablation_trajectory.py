#!/usr/bin/env python3
"""
Solver Ablation: Trajectory-Only Evaluation
============================================

Re-runs ONLY L4 trajectory evaluation (solver + judge) on existing model outputs
with a *different* solver model.  MCP servers are initialized from the existing
registry.json produced by a previous run, so no sandbox re-creation is needed.

Usage example:
    python solver_ablation_trajectory.py \
        --pred-path temp/run_benchmark_v3 \
        --out-root temp/solver_ablation \
        --solver-platform OPENAI --solver-model GPT_5_4 \
        --models coder_agent_qwen3-14b,direct_openai_gpt-4-1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# IMPORTANT: Set solver/judge env vars BEFORE any project imports, because
# agentic_framework.py reads them at module-level import time.
# We parse a minimal set of CLI args here, then set env vars, then import.
# ---------------------------------------------------------------------------

def _pre_parse_env_args() -> argparse.Namespace:
    """Parse only the env-var-relevant flags so we can set them early."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--solver-platform", type=str, default="BAILIAN")
    p.add_argument("--solver-model", type=str, default="BAILIAN_QWEN3_14B")
    p.add_argument("--judge-platform", type=str, default="")
    p.add_argument("--judge-model", type=str, default="")
    ns, _ = p.parse_known_args()
    return ns

_env_ns = _pre_parse_env_args()
os.environ["SOLVER_PLATFORM"] = _env_ns.solver_platform
os.environ["SOLVER_MODEL"] = _env_ns.solver_model
os.environ["JUDGE_PLATFORM"] = _env_ns.judge_platform or _env_ns.solver_platform
os.environ["JUDGE_MODEL"] = _env_ns.judge_model or _env_ns.solver_model

# Now safe to import project code.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.env_evaluation import GTResolver
from src.env_evaluation.l2_semantic_correctness import ExecuteTask
from src.core.toolkits import MCPServerToolsToolkit

GT_DATA_PATH = "data/tool_genesis_v3.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_registry(pred_path: Path) -> Dict[str, Any]:
    reg_file = pred_path / "registry.json"
    if not reg_file.exists():
        return {}
    with open(reg_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_model_dirs(pred_root: Path) -> List[str]:
    """Return model directory names that contain a registry.json."""
    out: List[str] = []
    if not pred_root.is_dir():
        return out
    for child in sorted(pred_root.iterdir()):
        if child.is_dir() and (child / "registry.json").exists():
            out.append(child.name)
    return out


def evaluate_server_trajectory(
    server_slug: str,
    registry_path: str,
    gt_schema: Dict[str, Any],
    trajectory_root: Path,
) -> Dict[str, Any]:
    """
    Initialize MCP server, run trajectory evaluation via ExecuteTask,
    and return per-server results.
    """
    example_tasks = gt_schema.get("task_example") or []
    tasks: List[str] = [t for t in example_tasks if isinstance(t, str)]

    if not tasks:
        return {
            "sr_soft": 0.0,
            "sr_hard": 0.0,
            "n_tasks": 0,
            "details": [],
            "skipped": "no_tasks",
        }

    tk: Optional[MCPServerToolsToolkit] = None
    try:
        tk = MCPServerToolsToolkit(
            server_names=server_slug.strip(),
            registry_path=registry_path,
        )
        tk.initialize_servers()

        execute_task = ExecuteTask(
            tasks=tasks,
            server_name=server_slug.strip(),
            toolkit=tk,
            schema_path=None,
            agent_result_path=trajectory_root,
        )

        traj_score, traj_info = execute_task.get_execute_task_result()
        traj_details = traj_info.get("details", [])
        traj_soft = float(traj_score)
        traj_hard_count = sum(1 for d in traj_details if d.get("solved"))
        traj_hard = traj_hard_count / len(traj_details) if traj_details else 0.0

        return {
            "sr_soft": traj_soft,
            "sr_hard": traj_hard,
            "n_tasks": len(tasks),
            "details": traj_details,
        }
    except Exception as exc:
        return {
            "sr_soft": 0.0,
            "sr_hard": 0.0,
            "n_tasks": len(tasks),
            "details": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if tk is not None:
            try:
                tk.cleanup()
            except Exception:
                pass


def evaluate_model(
    model_name: str,
    pred_root: Path,
    out_root: Path,
    gt_resolver: GTResolver,
    limit: Optional[int],
    workers: int,
) -> Dict[str, Any]:
    """Evaluate all servers for a single model directory."""
    model_pred = pred_root / model_name
    registry = _load_registry(model_pred)
    if not registry:
        print(f"  [WARN] No registry.json found for {model_name}, skipping")
        return {"n_servers": 0, "avg_sr_soft": 0.0, "avg_sr_hard": 0.0, "per_server": {}}

    registry_path = str(model_pred / "registry.json")
    model_out = out_root / model_name
    os.makedirs(model_out, exist_ok=True)

    # Check which servers have already been evaluated (for resumability)
    partial_path = model_out / "per_server.json"
    already_done: Dict[str, Any] = {}
    if partial_path.exists():
        try:
            with open(partial_path, "r", encoding="utf-8") as f:
                already_done = json.load(f)
        except Exception:
            already_done = {}

    slugs = list(registry.keys())
    slugs = [s for s in slugs if s not in already_done]
    if limit is not None and limit > 0:
        slugs = slugs[:limit]

    print(f"  [{model_name}] {len(slugs)} servers to evaluate ({len(already_done)} already done)")

    per_server: Dict[str, Any] = dict(already_done)

    def _run_one(slug: str) -> Tuple[str, Dict[str, Any]]:
        entry = registry.get(slug, {})
        server_name = entry.get("server_name") or slug
        gt_schema = gt_resolver.get_gt_schema_obj(server_name) or {}
        traj_root = model_out / "trajectory" / slug
        os.makedirs(traj_root, exist_ok=True)
        result = evaluate_server_trajectory(slug, registry_path, gt_schema, traj_root)
        return slug, result

    if workers > 1 and len(slugs) > 1:
        effective_workers = min(workers, len(slugs))
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_run_one, s): s for s in slugs}
            for i, future in enumerate(as_completed(futures), 1):
                slug = futures[future]
                try:
                    slug_key, result = future.result()
                    per_server[slug_key] = result
                    status = f"soft={result['sr_soft']:.3f} hard={result['sr_hard']:.3f}"
                    print(f"    [{i}/{len(slugs)}] {slug}: {status}")
                except Exception as exc:
                    per_server[slug] = {
                        "sr_soft": 0.0, "sr_hard": 0.0, "n_tasks": 0,
                        "details": [], "error": str(exc),
                    }
                    print(f"    [{i}/{len(slugs)}] {slug}: FAILED ({exc})")
                # Save progress after each server
                with open(partial_path, "w", encoding="utf-8") as f:
                    json.dump(per_server, f, ensure_ascii=False, indent=2)
    else:
        for i, slug in enumerate(slugs, 1):
            try:
                slug_key, result = _run_one(slug)
                per_server[slug_key] = result
                status = f"soft={result['sr_soft']:.3f} hard={result['sr_hard']:.3f}"
                print(f"    [{i}/{len(slugs)}] {slug}: {status}")
            except Exception as exc:
                per_server[slug] = {
                    "sr_soft": 0.0, "sr_hard": 0.0, "n_tasks": 0,
                    "details": [], "error": str(exc),
                }
                print(f"    [{i}/{len(slugs)}] {slug}: FAILED ({exc})")
            # Save progress after each server
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(per_server, f, ensure_ascii=False, indent=2)

    # Compute aggregates
    scored = [v for v in per_server.values() if v.get("n_tasks", 0) > 0]
    avg_soft = sum(v["sr_soft"] for v in scored) / len(scored) if scored else 0.0
    avg_hard = sum(v["sr_hard"] for v in scored) / len(scored) if scored else 0.0

    return {
        "n_servers": len(per_server),
        "avg_sr_soft": avg_soft,
        "avg_sr_hard": avg_hard,
        "per_server": per_server,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Re-run L4 trajectory evaluation with a different solver model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pred-path", type=str, required=True,
        help="Root directory containing per-model output dirs (e.g. temp/run_benchmark_v3)",
    )
    parser.add_argument(
        "--out-root", type=str, required=True,
        help="Where to write ablation results (e.g. temp/solver_ablation)",
    )
    parser.add_argument("--solver-platform", type=str, default="BAILIAN")
    parser.add_argument("--solver-model", type=str, default="BAILIAN_QWEN3_14B")
    parser.add_argument("--judge-platform", type=str, default="")
    parser.add_argument("--judge-model", type=str, default="")
    parser.add_argument(
        "--models", type=str, default="all",
        help="Comma-separated model dir names to evaluate, or 'all'",
    )
    parser.add_argument("--workers", type=int, default=1, help="Thread parallelism per model")
    parser.add_argument("--limit", type=int, default=None, help="Max servers per model")
    parser.add_argument("--gt-path", type=str, default=GT_DATA_PATH, help="Path to GT data JSON")
    args = parser.parse_args()

    pred_root = Path(args.pred_path)
    out_root = Path(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    solver_desc = f"{os.environ['SOLVER_PLATFORM']}/{os.environ['SOLVER_MODEL']}"
    judge_desc = f"{os.environ['JUDGE_PLATFORM']}/{os.environ['JUDGE_MODEL']}"
    print(f"[SolverAblation] solver={solver_desc}  judge={judge_desc}")
    print(f"[SolverAblation] pred-path={pred_root}  out-root={out_root}")

    # Determine which models to evaluate
    if args.models.strip().lower() == "all":
        model_names = _list_model_dirs(pred_root)
    else:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    if not model_names:
        print("[SolverAblation] No model directories found. Exiting.")
        return

    print(f"[SolverAblation] Models to evaluate: {model_names}")

    gt_resolver = GTResolver(args.gt_path)
    all_results: Dict[str, Any] = {}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"[SolverAblation] Evaluating model: {model_name}")
        print(f"{'='*60}")
        t0 = time.time()
        model_result = evaluate_model(
            model_name, pred_root, out_root, gt_resolver,
            limit=args.limit, workers=args.workers,
        )
        elapsed = time.time() - t0
        all_results[model_name] = model_result
        print(f"  [{model_name}] Done in {elapsed:.1f}s  "
              f"avg_sr_soft={model_result['avg_sr_soft']:.4f}  "
              f"avg_sr_hard={model_result['avg_sr_hard']:.4f}")

    # Write final summary
    summary = {
        "solver": solver_desc,
        "judge": judge_desc,
        "n_models_evaluated": len(all_results),
        "results": all_results,
    }
    summary_path = out_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[SolverAblation] Summary written to {summary_path}")

    # Print quick overview table
    print(f"\n{'Model':<50} {'SR_soft':>8} {'SR_hard':>8} {'#Srv':>5}")
    print("-" * 75)
    for model_name, res in all_results.items():
        print(f"{model_name:<50} {res['avg_sr_soft']:>8.4f} {res['avg_sr_hard']:>8.4f} {res['n_servers']:>5}")


if __name__ == "__main__":
    main()
