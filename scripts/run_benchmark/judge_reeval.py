#!/usr/bin/env python3
"""
Judge Re-Evaluation
===================

Re-judges EXISTING trajectory solver conversations with a *different* judge model.
This is much cheaper than re-running solver+judge because the solver is NOT re-run;
only the judge LLM is called on the already-saved conversation logs.

If trajectory conversation logs are not found for a given model, falls back to
full trajectory re-execution (solver+judge) with the new judge model.

Conversation logs are expected at:
    temp/eval_results_v3/{model}/trajectory/{server_slug}/task_{N}/
        simulate_solve/{timestamp}/logs/conversation.json

Usage example:
    python judge_reeval.py \
        --pred-path temp/run_benchmark_v3 \
        --traj-path temp/eval_results_v3 \
        --out-root temp/judge_reeval_gpt54 \
        --judge-platform OPENAI --judge-model GPT_5_4 \
        --models coder_agent_qwen3-14b,direct_openai_gpt-4-1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Set judge env vars BEFORE imports (agentic_framework.py reads at import time).
# For judge-only re-eval the solver env vars are irrelevant, but we set them
# to the original defaults so that the fallback path (re-running solver+judge)
# uses the original solver.
# ---------------------------------------------------------------------------

def _pre_parse_env_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--solver-platform", type=str, default="BAILIAN")
    p.add_argument("--solver-model", type=str, default="BAILIAN_QWEN3_14B")
    p.add_argument("--judge-platform", type=str, default="OPENAI")
    p.add_argument("--judge-model", type=str, default="GPT_5_4")
    ns, _ = p.parse_known_args()
    return ns

_env_ns = _pre_parse_env_args()
os.environ["SOLVER_PLATFORM"] = _env_ns.solver_platform
os.environ["SOLVER_MODEL"] = _env_ns.solver_model
os.environ["JUDGE_PLATFORM"] = _env_ns.judge_platform
os.environ["JUDGE_MODEL"] = _env_ns.judge_model

# Now safe to import project code.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.env_evaluation import GTResolver
from src.env_evaluation.l2_semantic_correctness import ExecuteTask
from src.env_evaluation.prompt import build_judge_task_prompt
from src.env_evaluation.agentic_framework import initialize_agent
from src.core.toolkits import MCPServerToolsToolkit

GT_DATA_PATH = "data/tool_genesis_v3.json"


# ---------------------------------------------------------------------------
# Judge-only re-evaluation from saved conversation logs
# ---------------------------------------------------------------------------

def _parse_judge_response(xml_text: str) -> Tuple[int, str]:
    """Parse the judge XML response, returning (score, analysis).
    Mirrors ExecuteTask._parse_judge_response logic."""
    text = xml_text.strip()
    start_idx = text.find("<response")
    end_idx = text.rfind("</response>")
    if start_idx != -1 and end_idx != -1:
        end_idx += len("</response>")
        text = text[start_idx:end_idx]
    if not text:
        return 0, ""
    try:
        root = ET.fromstring(text)
    except Exception:
        return 0, text
    reasoning_elem = root.find(".//reasoning")
    rating_elem = root.find(".//rating")
    reasoning = "".join(reasoning_elem.itertext()).strip() if reasoning_elem is not None else text
    rating_raw = rating_elem.text.strip() if rating_elem is not None and rating_elem.text else ""
    rating_lower = rating_raw.lower()
    score_map = {
        "very incomplete": 1,
        "incomplete": 2,
        "partially complete": 3,
        "mostly complete": 4,
        "fully complete": 5,
    }
    if rating_lower.isdigit():
        try:
            score = int(rating_lower)
        except Exception:
            score = 0
    else:
        score = score_map.get(rating_lower, 0)
    if score < 0 or score > 5:
        score = 0
    return score, reasoning


def _find_solver_conversation(traj_task_dir: Path) -> Optional[List[Dict[str, Any]]]:
    """Find and load the solver conversation.json for a given task directory.
    Returns the OpenAI-format message list, or None if not found."""
    solve_dir = traj_task_dir / "simulate_solve"
    if not solve_dir.exists():
        return None
    # There may be multiple timestamp dirs; pick the latest one.
    candidates = sorted(solve_dir.iterdir(), reverse=True) if solve_dir.is_dir() else []
    for ts_dir in candidates:
        conv_path = ts_dir / "logs" / "conversation.json"
        if conv_path.exists():
            try:
                with open(conv_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
            except Exception:
                continue
    return None


def _create_judge_model():
    """Create a standalone judge model instance using the configured env vars."""
    from src.core.models import ModelFactory
    from src.core.types import ModelPlatformType, ModelType

    judge_platform_name = os.environ.get("JUDGE_PLATFORM", "BAILIAN")
    judge_model_name = os.environ.get("JUDGE_MODEL", "BAILIAN_QWEN3_14B")
    judge_platform = getattr(ModelPlatformType, judge_platform_name, ModelPlatformType.BAILIAN)
    judge_model_type = getattr(ModelType, judge_model_name, ModelType.BAILIAN_QWEN3_14B)
    return ModelFactory.create(
        model_platform=judge_platform,
        model_type=judge_model_type,
        model_config_dict={"temperature": 0},
    )


def _judge_single_task(
    task_text: str,
    conversation_history_str: str,
    judge_out_dir: Path,
) -> Tuple[int, str]:
    """Call the judge on a single task using the configured judge model.
    Returns (score, analysis)."""
    from src.core.agents import ChatAgent

    os.makedirs(judge_out_dir, exist_ok=True)
    judge_model = _create_judge_model()
    judge_agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=judge_model,
        tools=[],
        auto_save=True,
        results_base_dir=str(judge_out_dir),
    )

    prompt = build_judge_task_prompt(task_text, conversation_history_str)
    last_score, last_analysis = 0, ""
    for _ in range(3):
        try:
            raw_response = judge_agent.step(prompt)
            content = getattr(raw_response, "content", str(raw_response))
            score, analysis = _parse_judge_response(content)
            last_score, last_analysis = score, analysis
            if score > 0:
                return score, analysis
        except Exception:
            continue
    return last_score, last_analysis


def rejudge_from_logs(
    server_slug: str,
    traj_dir: Path,
    gt_schema: Dict[str, Any],
    out_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Re-judge a server's trajectory using saved solver conversations.
    Returns per-server result dict, or None if logs are missing.
    """
    example_tasks = gt_schema.get("task_example") or []
    tasks: List[str] = [t for t in example_tasks if isinstance(t, str)]
    if not tasks:
        return {"sr_soft": 0.0, "sr_hard": 0.0, "n_tasks": 0, "details": [], "skipped": "no_tasks"}

    server_traj = traj_dir / server_slug
    if not server_traj.exists():
        return None  # Signal: no saved logs

    details: List[Dict[str, Any]] = []
    scores: List[int] = []

    for idx, task_text in enumerate(tasks):
        task_dir = server_traj / f"task_{idx + 1}"
        conversation = _find_solver_conversation(task_dir)
        if conversation is None:
            # Missing conversation for this task
            details.append({
                "task": task_text,
                "response": "",
                "judge_output": "solver conversation log not found",
                "solved": False,
                "score": 0,
            })
            scores.append(0)
            continue

        # Build conversation history string (same format ExecuteTask uses)
        try:
            conv_str = json.dumps(conversation, ensure_ascii=False, indent=2)
        except Exception:
            conv_str = "\n".join(str(m) for m in conversation)

        judge_out = out_dir / "trajectory" / server_slug / f"task_{idx + 1}" / "judge"
        score, analysis = _judge_single_task(task_text, conv_str, judge_out)
        scores.append(score)
        details.append({
            "task": task_text,
            "response": "(from saved log)",
            "judge_output": analysis,
            "solved": score >= 3,
            "score": score,
        })

    n_tasks = len(tasks)
    task_scores_norm = [max(0.0, min(1.0, s / 5.0)) for s in scores]
    sr_soft = sum(task_scores_norm) / n_tasks if n_tasks > 0 else 0.0
    sr_hard = sum(1 for s in scores if s >= 3) / n_tasks if n_tasks > 0 else 0.0

    return {
        "sr_soft": sr_soft,
        "sr_hard": sr_hard,
        "n_tasks": n_tasks,
        "details": details,
        "mode": "rejudge_from_logs",
    }


# ---------------------------------------------------------------------------
# Fallback: full trajectory re-run with new judge (if logs not available)
# ---------------------------------------------------------------------------

def fallback_trajectory(
    server_slug: str,
    registry_path: str,
    gt_schema: Dict[str, Any],
    trajectory_root: Path,
) -> Dict[str, Any]:
    """Re-run full trajectory (solver + new judge) for a server.
    Same logic as solver_ablation_trajectory.evaluate_server_trajectory."""
    example_tasks = gt_schema.get("task_example") or []
    tasks: List[str] = [t for t in example_tasks if isinstance(t, str)]
    if not tasks:
        return {"sr_soft": 0.0, "sr_hard": 0.0, "n_tasks": 0, "details": [], "skipped": "no_tasks"}

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
            "mode": "fallback_full_trajectory",
        }
    except Exception as exc:
        return {
            "sr_soft": 0.0, "sr_hard": 0.0, "n_tasks": len(tasks),
            "details": [], "error": f"{type(exc).__name__}: {exc}",
            "mode": "fallback_full_trajectory",
        }
    finally:
        if tk is not None:
            try:
                tk.cleanup()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Model-level orchestration
# ---------------------------------------------------------------------------

def _load_registry(pred_path: Path) -> Dict[str, Any]:
    reg_file = pred_path / "registry.json"
    if not reg_file.exists():
        return {}
    with open(reg_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_model_dirs(pred_root: Path) -> List[str]:
    out: List[str] = []
    if not pred_root.is_dir():
        return out
    for child in sorted(pred_root.iterdir()):
        if child.is_dir() and (child / "registry.json").exists():
            out.append(child.name)
    return out


def evaluate_model(
    model_name: str,
    pred_root: Path,
    traj_root: Path,
    out_root: Path,
    gt_resolver: GTResolver,
    limit: Optional[int],
    workers: int,
) -> Dict[str, Any]:
    """Evaluate all servers for a single model."""
    model_pred = pred_root / model_name
    registry = _load_registry(model_pred)
    if not registry:
        print(f"  [WARN] No registry.json for {model_name}, skipping")
        return {"n_servers": 0, "avg_sr_soft": 0.0, "avg_sr_hard": 0.0, "per_server": {}}

    registry_path = str(model_pred / "registry.json")
    model_traj = traj_root / model_name / "trajectory"
    model_out = out_root / model_name
    os.makedirs(model_out, exist_ok=True)

    # Resumability: check what we already have
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

    has_traj_logs = model_traj.exists() and any(model_traj.iterdir()) if model_traj.exists() else False
    mode = "rejudge" if has_traj_logs else "fallback"
    print(f"  [{model_name}] {len(slugs)} servers pending ({len(already_done)} done) mode={mode}")

    per_server: Dict[str, Any] = dict(already_done)

    def _run_one(slug: str) -> Tuple[str, Dict[str, Any]]:
        entry = registry.get(slug, {})
        server_name = entry.get("server_name") or slug
        gt_schema = gt_resolver.get_gt_schema_obj(server_name) or {}

        result = None
        if has_traj_logs:
            result = rejudge_from_logs(slug, model_traj, gt_schema, model_out)

        if result is None:
            # Fallback: re-run full trajectory with new judge
            fallback_root = model_out / "trajectory" / slug
            os.makedirs(fallback_root, exist_ok=True)
            result = fallback_trajectory(slug, registry_path, gt_schema, fallback_root)

        return slug, result

    def _save_progress():
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(per_server, f, ensure_ascii=False, indent=2)

    if workers > 1 and len(slugs) > 1:
        effective_workers = min(workers, len(slugs))
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_run_one, s): s for s in slugs}
            for i, future in enumerate(as_completed(futures), 1):
                slug = futures[future]
                try:
                    slug_key, result = future.result()
                    per_server[slug_key] = result
                    m = result.get("mode", "?")
                    print(f"    [{i}/{len(slugs)}] {slug}: soft={result['sr_soft']:.3f} hard={result['sr_hard']:.3f} ({m})")
                except Exception as exc:
                    per_server[slug] = {
                        "sr_soft": 0.0, "sr_hard": 0.0, "n_tasks": 0,
                        "details": [], "error": str(exc),
                    }
                    print(f"    [{i}/{len(slugs)}] {slug}: FAILED ({exc})")
                _save_progress()
    else:
        for i, slug in enumerate(slugs, 1):
            try:
                slug_key, result = _run_one(slug)
                per_server[slug_key] = result
                m = result.get("mode", "?")
                print(f"    [{i}/{len(slugs)}] {slug}: soft={result['sr_soft']:.3f} hard={result['sr_hard']:.3f} ({m})")
            except Exception as exc:
                per_server[slug] = {
                    "sr_soft": 0.0, "sr_hard": 0.0, "n_tasks": 0,
                    "details": [], "error": str(exc),
                }
                print(f"    [{i}/{len(slugs)}] {slug}: FAILED ({exc})")
            _save_progress()

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
        description="Re-judge existing trajectory conversations with a different judge model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pred-path", type=str, required=True,
        help="Root with per-model output dirs (e.g. temp/run_benchmark_v3)",
    )
    parser.add_argument(
        "--traj-path", type=str, default="temp/eval_results_v3",
        help="Root with existing trajectory logs (e.g. temp/eval_results_v3)",
    )
    parser.add_argument(
        "--out-root", type=str, required=True,
        help="Where to write re-judged results",
    )
    parser.add_argument("--solver-platform", type=str, default="BAILIAN",
                        help="Solver platform (used only in fallback mode)")
    parser.add_argument("--solver-model", type=str, default="BAILIAN_QWEN3_14B",
                        help="Solver model (used only in fallback mode)")
    parser.add_argument("--judge-platform", type=str, default="OPENAI")
    parser.add_argument("--judge-model", type=str, default="GPT_5_4")
    parser.add_argument(
        "--models", type=str, default="all",
        help="Comma-separated model dir names, or 'all'",
    )
    parser.add_argument("--workers", type=int, default=1, help="Thread parallelism per model")
    parser.add_argument("--limit", type=int, default=None, help="Max servers per model")
    parser.add_argument("--gt-path", type=str, default=GT_DATA_PATH)
    args = parser.parse_args()

    pred_root = Path(args.pred_path)
    traj_root = Path(args.traj_path)
    out_root = Path(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    solver_desc = f"{os.environ['SOLVER_PLATFORM']}/{os.environ['SOLVER_MODEL']}"
    judge_desc = f"{os.environ['JUDGE_PLATFORM']}/{os.environ['JUDGE_MODEL']}"
    print(f"[JudgeReeval] solver={solver_desc} (for fallback)  judge={judge_desc}")
    print(f"[JudgeReeval] pred-path={pred_root}  traj-path={traj_root}  out-root={out_root}")

    if args.models.strip().lower() == "all":
        model_names = _list_model_dirs(pred_root)
    else:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    if not model_names:
        print("[JudgeReeval] No model directories found. Exiting.")
        return

    print(f"[JudgeReeval] Models to evaluate: {model_names}")

    gt_resolver = GTResolver(args.gt_path)
    all_results: Dict[str, Any] = {}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"[JudgeReeval] Evaluating model: {model_name}")
        print(f"{'='*60}")
        t0 = time.time()
        model_result = evaluate_model(
            model_name, pred_root, traj_root, out_root, gt_resolver,
            limit=args.limit, workers=args.workers,
        )
        elapsed = time.time() - t0
        all_results[model_name] = model_result
        print(f"  [{model_name}] Done in {elapsed:.1f}s  "
              f"avg_sr_soft={model_result['avg_sr_soft']:.4f}  "
              f"avg_sr_hard={model_result['avg_sr_hard']:.4f}")

    summary = {
        "solver": solver_desc,
        "judge": judge_desc,
        "n_models_evaluated": len(all_results),
        "results": all_results,
    }
    summary_path = out_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[JudgeReeval] Summary written to {summary_path}")

    print(f"\n{'Model':<50} {'SR_soft':>8} {'SR_hard':>8} {'#Srv':>5}")
    print("-" * 75)
    for model_name, res in all_results.items():
        print(f"{model_name:<50} {res['avg_sr_soft']:>8.4f} {res['avg_sr_hard']:>8.4f} {res['n_servers']:>5}")


if __name__ == "__main__":
    main()
