"""
Experiment B: Multi-Round Evolution Runner.

For each (model, server):
  Round 0: Use existing generated code from temp/run_benchmark_v3/
  Round 1..N: Feed back train-task failures → LLM → improved code
  Each round: evaluate on train_tasks → collect feedback → next round
  Final: evaluate all rounds on test_tasks → evolution curve

Usage:
  python scripts/experiment_journal/evolution/run_evolution.py \
    --model openai/gpt-4-1 \
    --strategy coder_agent \
    --rounds 3 \
    --servers academic-author-network,airbnb-search-and-listing-details-server \
    --workers 1

Output:
  temp/evolution_results/{strategy}_{model}/{server_slug}/
    round_0/env_code.py, tool_schema.json, feedback.json, eval_debug/
    round_1/...
    round_N/...
    evolution_summary.json
"""

import json
import os
import re
import sys
import copy
import ast
import argparse
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_journal.evolution.feedback_collector import (
    collect_feedback,
    collect_ut_feedback,
    format_feedback_text,
)
from scripts.experiment_journal.evolution.prompt_template import (
    build_evolution_messages,
    build_evolution_history_messages,
)


def _inline_evaluate(
    code: str,
    gt_item: dict,
    ut_train_indices: dict,
    round_dir: str,
    tool_map: dict = None,
) -> Tuple[Optional[dict], str]:
    """Evaluate generated code inline: syntax check + UT execution on train split.

    Returns (eval_result_dict_or_None, feedback_text).
    eval_result_dict has keys: syntax_ok, ut_pass_rate, ut_total, ut_passed, ut_failures.
    """
    # 1. Syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        feedback = (
            f"## Evaluation Feedback\n"
            f"**Syntax Error** at line {e.lineno}: {e.msg}\n"
            f"Please fix the syntax error and try again."
        )
        return {"syntax_ok": False, "ut_pass_rate": 0.0, "ut_total": 0, "ut_passed": 0}, feedback

    # 2. Write code to temp file and try to launch as MCP server
    code_path = os.path.join(round_dir, "env_code.py")

    # 3. Run UT tests via subprocess to isolate server lifecycle
    ut = gt_item.get("unit_test", {})
    if not ut:
        return {"syntax_ok": True, "ut_pass_rate": 0.0, "ut_total": 0, "ut_passed": 0}, \
            "## Evaluation Feedback\nNo unit tests available for this server."

    # Build test cases from train split, remapping GT tool names → predicted tool names
    _tm = tool_map or {}  # {gt_tool_name: pred_tool_name}
    test_cases = []
    for tool_name, cases in ut.items():
        indices = ut_train_indices.get(tool_name, [])
        if not isinstance(cases, list):
            continue
        # Use mapped tool name if available, otherwise keep GT name
        mapped_name = _tm.get(tool_name, tool_name)
        for idx in indices:
            if idx < len(cases):
                case = cases[idx]
                test_cases.append({
                    "tool_name": mapped_name,
                    "gt_tool_name": tool_name,
                    "arguments": case.get("arguments", {}),
                    "expected": case.get("function_output_content", ""),
                })

    if not test_cases:
        return {"syntax_ok": True, "ut_pass_rate": 0.0, "ut_total": 0, "ut_passed": 0}, \
            "## Evaluation Feedback\nNo train-split test cases found."

    # Write a test runner that uses the MCP client to actually call tools
    runner_script = os.path.join(round_dir, "_ut_runner.py")
    # Resolve mcp_boot.py path relative to project root
    boot_script = os.path.join(PROJECT_ROOT, "src", "apps", "factory", "mcp_boot.py")

    runner_code = '''
import json, sys, os, subprocess, time, asyncio

async def _run_mcp_tests(code_path, test_cases):
    """Launch MCP server via mcp_boot.py and run real tool calls."""
    results = []
    try:
        from mcp.client.stdio import stdio_client
        from mcp import StdioServerParameters, ClientSession
    except ImportError:
        return _run_tests_fallback(code_path, test_cases)

    # Use mcp_boot.py to properly launch App-based servers
    # (handles `from src.apps import App`, factory pattern, etc.)
    boot_path = os.environ.get("MCP_BOOT_PATH", "")
    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    if boot_path and os.path.exists(boot_path):
        cmd_args = [boot_path, "--file-path", code_path]
    else:
        cmd_args = [code_path]

    env = {**os.environ, "PYTHONPATH": project_root}
    params = StdioServerParameters(
        command=sys.executable,
        args=cmd_args,
        env=env,
        cwd=project_root,
    )

    try:
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
                available_tools = {t.name for t in tools_resp.tools}

                # Build fuzzy tool name mapping: GT name -> best matching server tool
                def _best_match(gt_name, available):
                    """Find best matching tool by word overlap."""
                    if gt_name in available:
                        return gt_name
                    gt_words = set(gt_name.lower().replace("_", " ").split())
                    best, best_score = None, 0
                    for t in available:
                        t_words = set(t.lower().replace("_", " ").split())
                        overlap = len(gt_words & t_words)
                        if overlap > best_score:
                            best, best_score = t, overlap
                    return best if best_score > 0 else None

                tool_map = {}
                for tc in test_cases:
                    gt = tc["tool_name"]
                    if gt not in tool_map:
                        tool_map[gt] = _best_match(gt, available_tools)

                for tc in test_cases:
                    tool_name = tool_map.get(tc["tool_name"])
                    arguments = tc.get("arguments", {})
                    expected = tc.get("expected", "")

                    if tool_name is None:
                        results.append({
                            "passed": False,
                            "error": f"Tool '{tc['tool_name']}' not matched. Available: {sorted(available_tools)}",
                        })
                        continue

                    try:
                        result = await asyncio.wait_for(
                            session.call_tool(tool_name, arguments),
                            timeout=30,
                        )
                        # Extract text content from result
                        actual_parts = []
                        if hasattr(result, "content"):
                            for c in result.content:
                                if hasattr(c, "text"):
                                    actual_parts.append(c.text)
                        actual = " ".join(actual_parts)

                        # Basic similarity: check if result is non-empty and non-error
                        is_error = getattr(result, "isError", False)
                        has_content = bool(actual.strip())
                        results.append({
                            "passed": has_content and not is_error,
                            "actual": actual[:500],
                            "is_error": is_error,
                        })
                    except asyncio.TimeoutError:
                        results.append({"passed": False, "error": "Tool call timed out (30s)"})
                    except Exception as e:
                        results.append({"passed": False, "error": str(e)[:200]})
    except Exception as e:
        # Server failed to start or MCP handshake failed
        for tc in test_cases:
            results.append({"passed": False, "error": f"Server/MCP error: {str(e)[:200]}"})

    return results


def _run_tests_fallback(code_path, test_cases):
    """Fallback: launch server and check if it starts (no tool calls)."""
    results = []
    try:
        proc = subprocess.Popen(
            [sys.executable, code_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        time.sleep(2)
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="replace")
            for tc in test_cases:
                results.append({"passed": False, "error": f"Server exited: {stderr[:200]}"})
        else:
            proc.terminate()
            proc.wait(timeout=5)
            for tc in test_cases:
                results.append({"passed": None, "status": "server_launched"})
    except Exception as e:
        for tc in test_cases:
            results.append({"passed": False, "error": str(e)[:200]})
    return results


def run_tests(code_path, test_cases_path, output_path):
    with open(test_cases_path) as f:
        test_cases = json.load(f)
    results = asyncio.run(_run_mcp_tests(code_path, test_cases))
    with open(output_path, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_tests(sys.argv[1], sys.argv[2], sys.argv[3])
'''
    with open(runner_script, "w") as f:
        f.write(runner_code)

    test_cases_path = os.path.join(round_dir, "_test_cases.json")
    with open(test_cases_path, "w") as f:
        json.dump(test_cases, f, ensure_ascii=False)

    output_path = os.path.join(round_dir, "_ut_results.json")

    # Run the test (pass boot script + project root via env)
    run_env = {**os.environ, "MCP_BOOT_PATH": boot_script, "PROJECT_ROOT": str(PROJECT_ROOT)}
    try:
        result = subprocess.run(
            [sys.executable, runner_script, code_path, test_cases_path, output_path],
            capture_output=True, text=True, timeout=120, env=run_env,
        )
    except subprocess.TimeoutExpired:
        feedback = "## Evaluation Feedback\n**Timeout**: Server launch timed out after 60s."
        return {"syntax_ok": True, "ut_pass_rate": 0.0, "ut_total": len(test_cases), "ut_passed": 0}, feedback

    # Parse results
    ut_results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            ut_results = json.load(f)

    # Count passes
    n_total = len(test_cases)
    n_launched = sum(1 for r in ut_results if r.get("status") == "server_launched")
    n_failed = sum(1 for r in ut_results if r.get("passed") is False)
    n_passed = sum(1 for r in ut_results if r.get("passed") is True)
    server_ok = n_launched > 0 or n_passed > 0

    # Build detailed feedback (with real tool call results for actionable LLM feedback)
    lines = [f"## Evaluation Feedback (Round)"]
    if not server_ok and n_failed > 0:
        errors = [r.get("error", "") for r in ut_results if r.get("error")]
        unique_errors = list(set(errors))[:5]
        lines.append(f"**Server failed.** {n_failed}/{n_total} tests failed.")
        for err in unique_errors:
            lines.append(f"- Error: {err}")
        ut_pass_rate = 0.0
    elif server_ok:
        ut_pass_rate = n_passed / n_total if n_total else 0.0
        lines.append(f"Server launched successfully. Testing {n_total} tool calls.")
        lines.append(f"- **{n_passed}/{n_total} tool calls succeeded** ({ut_pass_rate:.1%})")
        if n_failed > 0:
            lines.append(f"- {n_failed} tool calls failed:")
            for i, (tc, res) in enumerate(zip(test_cases, ut_results)):
                if res.get("passed") is False:
                    tool = tc.get("tool_name", "?")
                    args = json.dumps(tc.get("arguments", {}), ensure_ascii=False)[:100]
                    err = res.get("error", "wrong output")[:200]
                    lines.append(f"  - `{tool}({args})` → {err}")
                    if i >= 7:
                        lines.append(f"  - ... and {n_failed - i - 1} more failures")
                        break
        if n_launched > 0 and n_passed == 0:
            lines.append("  Note: MCP client not available; only server launch verified.")
    else:
        ut_pass_rate = 0.0
        lines.append("No evaluation results available.")

    eval_result = {
        "syntax_ok": True,
        "server_launched": server_ok,
        "ut_pass_rate": ut_pass_rate,
        "ut_total": n_total,
        "ut_passed": n_passed,
        "ut_launched": n_launched,
        "ut_failed": n_failed,
    }
    return eval_result, "\n".join(lines)


def _extract_code(raw: str) -> str:
    """Extract Python code from LLM response (fenced block)."""
    # Try to use the project's extract_code if available
    try:
        from src.utils.llm import extract_code
        return extract_code(raw)
    except ImportError:
        pass

    # Fallback: manual extraction
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, raw, re.DOTALL)
    if matches:
        return "\n\n".join(m.strip() for m in matches)
    return raw.strip()


def _call_llm(messages: list, model: str, platform: str) -> str:
    """Call LLM with messages."""
    from src.utils.llm import call_llm

    system_msg = ""
    user_msg = ""
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        elif m["role"] == "user":
            user_msg = m["content"]

    response = call_llm(
        text=user_msg,
        system_prompt=system_msg,
        model=model,
        max_tokens=8192,
        temperature=0.2,
        platform=platform,
    )
    return response


def _model_dir_variants(model: str) -> List[str]:
    """Generate directory name variants for a model (dot/hyphen, with/without provider prefix)."""
    clean = model.replace("/", "_")
    variants = {clean, clean.replace(".", "-"), clean.replace("-", ".")}
    # Also try without provider prefix (e.g. "openai_gpt-4-1" → "gpt-4-1")
    if "_" in clean:
        suffix = clean.split("_", 1)[1]
        variants.update({suffix, suffix.replace(".", "-"), suffix.replace("-", ".")})
    return list(variants)


def _load_existing_code(
    benchmark_root: str, strategy: str, model: str, server_slug: str
) -> Optional[str]:
    """Load existing generated code for round 0."""
    variants = _model_dir_variants(model)

    # Try exact match first
    for v in variants:
        code_path = os.path.join(benchmark_root, f"{strategy}_{v}", server_slug, "env_code.py")
        if os.path.exists(code_path):
            with open(code_path, "r", encoding="utf-8") as f:
                return f.read()

    # Try fuzzy match against all directories
    for name in os.listdir(benchmark_root):
        if strategy not in name:
            continue
        if any(v in name for v in variants):
            alt_path = os.path.join(benchmark_root, name, server_slug, "env_code.py")
            if os.path.exists(alt_path):
                with open(alt_path, "r", encoding="utf-8") as f:
                    return f.read()

    return None


def _load_existing_schema(
    benchmark_root: str, strategy: str, model: str, server_slug: str
) -> Optional[str]:
    """Load existing generated schema for round 0."""
    variants = _model_dir_variants(model)

    for v in variants:
        schema_path = os.path.join(benchmark_root, f"{strategy}_{v}", server_slug, "tool_schema.json")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                return f.read()

    for name in os.listdir(benchmark_root):
        if strategy not in name:
            continue
        if any(v in name for v in variants):
            alt_path = os.path.join(benchmark_root, name, server_slug, "tool_schema.json")
            if os.path.exists(alt_path):
                with open(alt_path, "r", encoding="utf-8") as f:
                    return f.read()

    return None


def _load_existing_eval(
    eval_root: str, strategy: str, model: str, server_slug: str
) -> Optional[dict]:
    """Load existing L2 debug data for round 0 feedback."""
    variants = _model_dir_variants(model)

    for v in variants:
        l2_path = os.path.join(eval_root, f"{strategy}_{v}", "debug", server_slug, "l2_debug.json")
        if os.path.exists(l2_path):
            with open(l2_path, "r", encoding="utf-8") as f:
                return json.load(f)

    for name in os.listdir(eval_root):
        if strategy not in name:
            continue
        if any(v in name for v in variants):
            alt_path = os.path.join(eval_root, name, "debug", server_slug, "l2_debug.json")
            if os.path.exists(alt_path):
                with open(alt_path, "r", encoding="utf-8") as f:
                    return json.load(f)

    return None


def evolve_server(
    server_slug: str,
    gt_item: dict,
    split: dict,
    model: str,
    platform: str,
    strategy: str,
    n_rounds: int,
    out_dir: str,
    benchmark_root: str,
    eval_root: str,
    use_history: bool = False,
    api_model: str = "",
) -> Dict[str, Any]:
    """
    Run multi-round evolution for a single server.

    *model* is used for round-0 code lookup; *api_model* (if set) overrides
    the model ID sent to the LLM for evolution rounds 1-N.

    Returns evolution summary dict.
    """
    server_out = os.path.join(out_dir, server_slug)
    os.makedirs(server_out, exist_ok=True)

    requirement = gt_item.get("agent_input_prompt", "")
    task_descriptions = gt_item.get("task_example", [])
    train_indices = split.get("train_task_indices", [])
    ut_train_indices = split.get("unit_test_train_indices", {})

    code_versions = []
    feedback_history = []
    round_results = []

    for r in range(n_rounds + 1):
        round_dir = os.path.join(server_out, f"round_{r}")
        os.makedirs(round_dir, exist_ok=True)

        print(f"  [{server_slug}] Round {r}/{n_rounds}...")

        if r == 0:
            # Round 0: use existing code
            code = _load_existing_code(benchmark_root, strategy, model, server_slug)
            if code is None:
                print(f"  [{server_slug}] WARNING: No existing code found, skipping")
                return {"server_slug": server_slug, "error": "no_existing_code"}

            # Save round 0 code
            with open(os.path.join(round_dir, "env_code.py"), "w") as f:
                f.write(code)

            # Copy existing schema
            schema = _load_existing_schema(benchmark_root, strategy, model, server_slug)
            if schema:
                with open(os.path.join(round_dir, "tool_schema.json"), "w") as f:
                    f.write(schema)

            # Load existing eval for feedback + tool name mapping
            l2_debug = _load_existing_eval(eval_root, strategy, model, server_slug)
            # Extract GT→pred tool name mapping from schema matching
            _tool_map = {}
            if l2_debug:
                for tm in l2_debug.get("schema", {}).get("tool_matches", []):
                    gt_name = tm.get("gt", "")
                    pred_name = tm.get("pred", "")
                    score = tm.get("score", 0)
                    if gt_name and pred_name and score >= 0.3:
                        _tool_map[gt_name] = pred_name

            if l2_debug:
                task_fb = collect_feedback(l2_debug, train_indices, task_descriptions, round_num=0)
                ut_fb = collect_ut_feedback(l2_debug, ut_train_indices)
                feedback_text = format_feedback_text(task_fb, ut_fb)
            else:
                task_fb = {"round": 0, "total_tasks": 0, "passed": 0, "failed": 0,
                           "success_rate": 0.0, "failure_summary": []}
                ut_fb = {"total_cases": 0, "passed": 0, "failed": 0, "failures": []}
                feedback_text = "No evaluation data available for round 0."

            # Save feedback
            with open(os.path.join(round_dir, "feedback.json"), "w") as f:
                json.dump({"task": task_fb, "ut": ut_fb}, f, indent=2, ensure_ascii=False)
            with open(os.path.join(round_dir, "feedback.txt"), "w") as f:
                f.write(feedback_text)

            code_versions.append(code)
            feedback_history.append(feedback_text)

            round_results.append({
                "round": 0,
                "sr_train": task_fb.get("success_rate", 0),
                "ut_pass_rate": ut_fb.get("pass_rate", 0),
                "source": "existing",
            })

        else:
            # Evolution round: call LLM with feedback
            prev_code = code_versions[-1]

            # Skip evolution if previous round was already perfect
            prev_result = round_results[-1] if round_results else {}
            prev_ut = prev_result.get("ut_pass_rate", 0)
            if prev_ut is not None and prev_ut >= 1.0:
                print(f"  [{server_slug}] Round {r}: skipping (previous round 100%)")
                code_versions.append(prev_code)
                feedback_history.append(feedback_history[-1])
                round_results.append({
                    "round": r, "ut_pass_rate": prev_ut,
                    "syntax_ok": True, "server_launched": True,
                    "source": "carried_over", "code_length": len(prev_code),
                })
                with open(os.path.join(round_dir, "env_code.py"), "w") as f:
                    f.write(prev_code)
                continue

            if use_history and len(code_versions) > 1:
                messages = build_evolution_history_messages(
                    requirement=requirement,
                    code_versions=code_versions,
                    feedback_history=feedback_history,
                    round_num=r,
                )
            else:
                messages = build_evolution_messages(
                    requirement=requirement,
                    current_code=prev_code,
                    feedback_text=feedback_history[-1],
                    round_num=r,
                    include_code=True,
                )

            # Save the prompt
            with open(os.path.join(round_dir, "prompt.json"), "w") as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)

            response = ""
            try:
                response = _call_llm(messages, model=api_model or model, platform=platform)
                new_code = _extract_code(response)
            except Exception as e:
                print(f"  [{server_slug}] ERROR in round {r}: {e}")
                new_code = prev_code  # Fall back to previous version

            # Save generated code
            with open(os.path.join(round_dir, "env_code.py"), "w") as f:
                f.write(new_code)
            with open(os.path.join(round_dir, "llm_response.txt"), "w") as f:
                f.write(response if response else "")

            code_versions.append(new_code)

            # Real inline evaluation: syntax + server launch + UT on train split
            print(f"  [{server_slug}] Round {r}: evaluating generated code...")
            eval_result, feedback_text = _inline_evaluate(
                code=new_code,
                gt_item=gt_item,
                ut_train_indices=ut_train_indices,
                round_dir=round_dir,
                tool_map=_tool_map,
            )

            new_ut = eval_result.get("ut_pass_rate", 0) if eval_result else 0
            prev_ut = round_results[-1].get("ut_pass_rate", 0) or 0

            # Elitist selection: if new code is worse, revert to previous best
            if new_ut < prev_ut - 0.01:
                print(f"  [{server_slug}] Round {r}: regression ({prev_ut:.1%}→{new_ut:.1%}), reverting")
                code_versions[-1] = code_versions[-2]  # revert code
                round_results.append({
                    "round": r,
                    "ut_pass_rate": prev_ut,
                    "source": "reverted",
                    "reverted_from": new_ut,
                    "code_length": len(code_versions[-1]),
                })
                # Keep previous feedback so next round gets correct context
                feedback_history.append(feedback_history[-1])
                # Overwrite the env_code with reverted version
                with open(os.path.join(round_dir, "env_code.py"), "w") as f:
                    f.write(code_versions[-1])
            else:
                round_results.append({
                    "round": r,
                    "ut_pass_rate": new_ut,
                    "syntax_ok": eval_result.get("syntax_ok", False) if eval_result else False,
                    "server_launched": eval_result.get("server_launched", False) if eval_result else False,
                    "source": "evolved",
                    "code_length": len(new_code),
                    "eval": eval_result,
                })
                feedback_history.append(feedback_text)

            # Save evaluation results
            with open(os.path.join(round_dir, "eval_result.json"), "w") as f:
                json.dump(eval_result, f, indent=2, ensure_ascii=False)

            with open(os.path.join(round_dir, "feedback.txt"), "w") as f:
                f.write(feedback_text)

    # Save evolution summary
    summary = {
        "server_slug": server_slug,
        "model": model,
        "strategy": strategy,
        "n_rounds": n_rounds,
        "rounds": round_results,
        "code_lengths": [len(c) for c in code_versions],
    }
    with open(os.path.join(server_out, "evolution_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment B: Multi-Round Evolution")
    parser.add_argument("--model", type=str, required=True, help="Base model for round-0 code lookup (e.g. openai/gpt-5-1)")
    parser.add_argument("--api-model", type=str, default="", help="Model ID for API calls in evolution rounds (default: same as --model)")
    parser.add_argument("--platform", type=str, default="openai", help="LLM platform")
    parser.add_argument("--strategy", type=str, default="coder_agent", help="Original generation strategy")
    parser.add_argument("--rounds", type=int, default=3, help="Number of evolution rounds")
    parser.add_argument("--data-path", type=str, default="data/tool_genesis_v3.json")
    parser.add_argument("--split-path", type=str, default="data/task_split.json")
    parser.add_argument("--benchmark-root", type=str, default="temp/run_benchmark_v3")
    parser.add_argument("--eval-root", type=str, default="temp/eval_results_v3")
    parser.add_argument("--out-root", type=str, default="temp/evolution_results")
    parser.add_argument("--servers", type=str, default="", help="Comma-separated server slugs (empty=all)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--use-history", action="store_true", help="Include full history in evolution prompt")
    args = parser.parse_args()

    # Load data
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(args.split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    gt_lookup = {d["server_slug"]: d for d in data}

    # Filter servers
    if args.servers:
        server_list = [s.strip() for s in args.servers.split(",")]
    else:
        server_list = list(splits.keys())

    # Output directory uses the API model name (the model actually doing evolution)
    effective_model = args.api_model or args.model
    model_clean = effective_model.replace("/", "_").replace(".", "-")
    out_dir = os.path.join(args.out_root, f"{args.strategy}_{model_clean}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Evolution experiment: base={args.model}, api_model={effective_model}, "
          f"strategy={args.strategy}, rounds={args.rounds}, servers={len(server_list)}")

    all_summaries = []

    if args.workers <= 1:
        for slug in server_list:
            gt_item = gt_lookup.get(slug)
            split = splits.get(slug)
            if not gt_item or not split:
                print(f"  Skipping {slug}: not found in data/splits")
                continue

            summary = evolve_server(
                server_slug=slug,
                gt_item=gt_item,
                split=split,
                model=args.model,
                platform=args.platform,
                strategy=args.strategy,
                n_rounds=args.rounds,
                out_dir=out_dir,
                benchmark_root=args.benchmark_root,
                eval_root=args.eval_root,
                use_history=args.use_history,
                api_model=args.api_model,
            )
            all_summaries.append(summary)
    else:
        # Parallel execution
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for slug in server_list:
                gt_item = gt_lookup.get(slug)
                split = splits.get(slug)
                if not gt_item or not split:
                    continue
                fut = pool.submit(
                    evolve_server,
                    server_slug=slug,
                    gt_item=gt_item,
                    split=split,
                    model=args.model,
                    platform=args.platform,
                    strategy=args.strategy,
                    n_rounds=args.rounds,
                    out_dir=out_dir,
                    benchmark_root=args.benchmark_root,
                    eval_root=args.eval_root,
                    use_history=args.use_history,
                    api_model=args.api_model,
                )
                futures[fut] = slug

            for fut in as_completed(futures):
                slug = futures[fut]
                try:
                    summary = fut.result()
                    all_summaries.append(summary)
                except Exception as e:
                    print(f"  ERROR: {slug}: {e}")

    # Save global summary
    global_summary = {
        "model": args.model,
        "strategy": args.strategy,
        "n_rounds": args.rounds,
        "n_servers": len(all_summaries),
        "errors": sum(1 for s in all_summaries if s.get("error")),
        "summaries": all_summaries,
    }
    summary_path = os.path.join(out_dir, "global_evolution_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_summaries)} servers processed.")
    print(f"  Errors: {global_summary['errors']}")
    print(f"  Results: {summary_path}")


if __name__ == "__main__":
    main()
