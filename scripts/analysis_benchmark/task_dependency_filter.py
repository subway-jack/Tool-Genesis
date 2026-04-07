"""Identify benchmark tasks that truly require tool invocation.

Tasks where even weak/direct models (no agent framework) achieve high scores
WITHOUT tools are "tool-independent" and should be flagged for potential removal.

Usage:
    python scripts/analysis_benchmark/task_dependency_filter.py
    python scripts/analysis_benchmark/task_dependency_filter.py --direct-threshold 0.6
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _as_list(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list):
        return [it for it in x if isinstance(it, dict)]
    return []


def _num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _parse_strategy(dirname: str) -> str:
    if dirname.startswith("direct_"):
        return "direct"
    if dirname.startswith("coder_agent_"):
        return "coder_agent"
    return "unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_results(eval_root: Path) -> Dict[str, Dict[str, float]]:
    """Load per-model, per-server trajectory_soft from eval_results_v3.

    Returns: {model_dir_name: {server_slug: trajectory_soft}}
    """
    out: Dict[str, Dict[str, float]] = {}
    if not eval_root.is_dir():
        return out
    for entry in sorted(eval_root.iterdir()):
        if not entry.is_dir():
            continue
        res_path = entry / "results.json"
        if not res_path.exists():
            continue
        raw = _load_json(res_path)
        rows = _as_list(raw)
        if not rows:
            continue
        server_sr: Dict[str, float] = {}
        for r in rows:
            slug = str(r.get("server_slug") or "")
            if not slug:
                continue
            m = r.get("metrics") or {}
            sr = _num(m.get("trajectory_level_validation_rate_soft"))
            server_sr[slug] = sr
        out[entry.name] = server_sr
    return out


def load_no_tools_baseline(path: Path) -> Dict[str, float]:
    """Load no-tools baseline: {server_slug: normalized_sr}."""
    out: Dict[str, float] = {}
    raw = _load_json(path)
    if not raw or not isinstance(raw, dict):
        return out
    results = raw.get("results", [])
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            server = str(item.get("server", ""))
            nsr = _num(item.get("normalized_sr"))
            if server:
                out[server] = nsr
    return out


def load_all_models_by_size(path: Path) -> Dict[str, Dict[str, float]]:
    """Load all_models_by_size.json: {model_name: {metric: value}}."""
    raw = _load_json(path)
    if not raw or not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze(
    eval_root: Path,
    no_tools_path: Path,
    all_models_path: Path,
    direct_threshold: float = 0.7,
    low_disc_std_threshold: float = 0.10,
) -> Dict[str, Any]:
    # 1) Load all eval results
    all_results = load_eval_results(eval_root)
    if not all_results:
        print("ERROR: No eval results found.", file=sys.stderr)
        return {}

    # Separate direct vs all models
    direct_results: Dict[str, Dict[str, float]] = {}
    all_model_results: Dict[str, Dict[str, float]] = {}
    for model_dir, server_sr in all_results.items():
        strategy = _parse_strategy(model_dir)
        if strategy == "direct":
            direct_results[model_dir] = server_sr
        all_model_results[model_dir] = server_sr

    # 2) Collect all server slugs
    all_servers: set = set()
    for server_sr in all_results.values():
        all_servers.update(server_sr.keys())
    all_servers_sorted = sorted(all_servers)

    # 3) Load no-tools baseline
    no_tools = load_no_tools_baseline(no_tools_path)

    # 4) Per-server analysis
    per_server: Dict[str, Dict[str, Any]] = {}
    tool_dependent: List[str] = []
    tool_independent: List[str] = []
    low_discrimination: List[str] = []

    for slug in all_servers_sorted:
        # Collect direct model scores for this server
        direct_scores = []
        for model_dir, server_sr in direct_results.items():
            if slug in server_sr:
                direct_scores.append(server_sr[slug])

        # Collect ALL model scores for this server
        all_scores = []
        for model_dir, server_sr in all_model_results.items():
            if slug in server_sr:
                all_scores.append(server_sr[slug])

        avg_direct = _mean(direct_scores)
        std_direct = _std(direct_scores)
        avg_all = _mean(all_scores)
        std_all = _std(all_scores)
        no_tools_sr = no_tools.get(slug)

        # Classification logic
        is_tool_independent = False
        is_low_disc = False
        reason_parts: List[str] = []

        # Check 1: direct models do well => task is easy without agent framework
        if avg_direct >= direct_threshold:
            is_tool_independent = True
            reason_parts.append(
                f"avg_direct_sr={avg_direct:.3f} >= {direct_threshold}"
            )

        # Check 2: cross-reference with no-tools baseline
        if no_tools_sr is not None and no_tools_sr >= direct_threshold:
            is_tool_independent = True
            reason_parts.append(
                f"no_tools_sr={no_tools_sr:.3f} >= {direct_threshold}"
            )

        # Check 3: low variance across ALL models => doesn't discriminate
        if len(all_scores) >= 5 and std_all < low_disc_std_threshold:
            is_low_disc = True
            reason_parts.append(
                f"std_all={std_all:.3f} < {low_disc_std_threshold} "
                f"(all models perform similarly)"
            )

        # Build reason string
        if is_tool_independent:
            reason = "tool-independent: " + "; ".join(reason_parts)
        elif is_low_disc:
            reason = "low-discrimination: " + "; ".join(reason_parts)
        else:
            reason = "tool-dependent"

        # Categorize
        if is_tool_independent:
            tool_independent.append(slug)
        elif is_low_disc:
            low_discrimination.append(slug)
        else:
            tool_dependent.append(slug)

        per_server[slug] = {
            "avg_direct_sr": round(avg_direct, 4),
            "std_direct_sr": round(std_direct, 4),
            "avg_all_sr": round(avg_all, 4),
            "std_all_sr": round(std_all, 4),
            "n_direct_models": len(direct_scores),
            "n_all_models": len(all_scores),
            "no_tools_sr": round(no_tools_sr, 4) if no_tools_sr is not None else None,
            "tool_dependent": not is_tool_independent,
            "low_discrimination": is_low_disc,
            "reason": reason,
        }

    # Build output
    recommended_filter = len(tool_independent) + len(low_discrimination)
    output = {
        "total_servers": len(all_servers_sorted),
        "tool_dependent": sorted(tool_dependent),
        "tool_independent": sorted(tool_independent),
        "low_discrimination": sorted(low_discrimination),
        "summary": {
            "n_tool_dependent": len(tool_dependent),
            "n_tool_independent": len(tool_independent),
            "n_low_discrimination": len(low_discrimination),
            "recommended_filter_count": recommended_filter,
            "direct_threshold": direct_threshold,
            "low_disc_std_threshold": low_disc_std_threshold,
            "n_direct_models": len(direct_results),
            "n_total_models": len(all_model_results),
        },
        "per_server": per_server,
    }
    return output


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_summary_table(result: Dict[str, Any]) -> None:
    summary = result.get("summary", {})
    per_server = result.get("per_server", {})

    print("=" * 90)
    print("TASK DEPENDENCY ANALYSIS")
    print("=" * 90)
    print(f"Total servers:              {result.get('total_servers', 0)}")
    print(f"Direct models evaluated:    {summary.get('n_direct_models', 0)}")
    print(f"Total models evaluated:     {summary.get('n_total_models', 0)}")
    print(f"Direct SR threshold:        {summary.get('direct_threshold', 0.7)}")
    print(f"Low-disc std threshold:     {summary.get('low_disc_std_threshold', 0.10)}")
    print("-" * 90)
    print(f"Tool-dependent servers:     {summary.get('n_tool_dependent', 0)}")
    print(f"Tool-independent servers:   {summary.get('n_tool_independent', 0)}")
    print(f"Low-discrimination servers: {summary.get('n_low_discrimination', 0)}")
    print(f"Recommended to filter out:  {summary.get('recommended_filter_count', 0)}")
    print("=" * 90)

    # Detailed table
    header = (
        f"{'Server':<45} {'AvgDir':>7} {'StdDir':>7} {'AvgAll':>7} "
        f"{'StdAll':>7} {'NoTool':>7} {'Category':<18}"
    )
    print()
    print(header)
    print("-" * len(header))

    # Sort by avg_direct_sr descending so tool-independent appear first
    sorted_servers = sorted(
        per_server.items(),
        key=lambda x: (-x[1].get("avg_direct_sr", 0), x[0]),
    )
    for slug, info in sorted_servers:
        no_tools_str = (
            f"{info['no_tools_sr']:.3f}" if info.get("no_tools_sr") is not None else "  -  "
        )
        if not info.get("tool_dependent"):
            category = "INDEPENDENT"
        elif info.get("low_discrimination"):
            category = "LOW-DISC"
        else:
            category = "dependent"

        line = (
            f"{slug:<45} {info['avg_direct_sr']:>7.3f} {info['std_direct_sr']:>7.3f} "
            f"{info['avg_all_sr']:>7.3f} {info['std_all_sr']:>7.3f} "
            f"{no_tools_str:>7} {category:<18}"
        )
        print(line)

    # Print the lists
    print()
    print("--- Tool-independent servers (candidates for removal) ---")
    for s in sorted(result.get("tool_independent", [])):
        info = per_server.get(s, {})
        print(f"  {s}: avg_direct={info.get('avg_direct_sr', 0):.3f}, "
              f"no_tools={info.get('no_tools_sr', 'N/A')}")

    print()
    print("--- Low-discrimination servers ---")
    for s in sorted(result.get("low_discrimination", [])):
        info = per_server.get(s, {})
        print(f"  {s}: std_all={info.get('std_all_sr', 0):.3f}, "
              f"avg_all={info.get('avg_all_sr', 0):.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify tool-independent tasks in the benchmark."
    )
    parser.add_argument(
        "--eval-root",
        type=str,
        default=str(BASE_DIR / "temp" / "eval_results_v3"),
        help="Path to eval_results_v3 directory.",
    )
    parser.add_argument(
        "--no-tools-baseline",
        type=str,
        default=str(BASE_DIR / "data" / "no_tools_baseline_results.json"),
        help="Path to no-tools baseline JSON.",
    )
    parser.add_argument(
        "--all-models",
        type=str,
        default=str(BASE_DIR / "data" / "all_models_by_size.json"),
        help="Path to all_models_by_size.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BASE_DIR / "data" / "task_dependency_analysis.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--direct-threshold",
        type=float,
        default=0.7,
        help="Avg direct-model trajectory_soft above which a task is tool-independent.",
    )
    parser.add_argument(
        "--low-disc-std",
        type=float,
        default=0.10,
        help="Std threshold below which a task has low discrimination.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary table output.",
    )
    args = parser.parse_args()

    result = analyze(
        eval_root=Path(args.eval_root),
        no_tools_path=Path(args.no_tools_baseline),
        all_models_path=Path(args.all_models),
        direct_threshold=args.direct_threshold,
        low_disc_std_threshold=args.low_disc_std,
    )
    if not result:
        sys.exit(1)

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved analysis to {out_path}")

    # Print summary
    if not args.quiet:
        print()
        print_summary_table(result)


if __name__ == "__main__":
    main()
