#!/usr/bin/env python3
"""Format oracle SR data into a paper-ready LaTeX table and summary JSON.

The oracle SR serves as an upper bound / "human substitute" baseline -- it
measures how well a solver agent performs when given the GROUND TRUTH tools
(perfect tool creation).

Outputs
-------
1. ``data/oracle_sr_summary.json`` -- structured summary for downstream use.
2. LaTeX table printed to stdout.
3. Key statistics for paper text printed to stdout.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}", file=sys.stderr)
        return None


def _mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _pretty_model_name(raw_key: str) -> str:
    """Turn ``coder_agent_openai_gpt-4-1`` into ``gpt-4-1 (coder)``."""
    if raw_key.startswith("coder_agent_"):
        rest = raw_key[len("coder_agent_"):]
        strategy_tag = "coder"
    elif raw_key.startswith("direct_"):
        rest = raw_key[len("direct_"):]
        strategy_tag = "direct"
    else:
        return raw_key

    # Provider prefixes we can strip for display
    for provider in ("openai_", "anthropic_", "google_", "deepseek_", "moonshotai_"):
        if rest.startswith(provider):
            rest = rest[len(provider):]
            break

    return f"{rest} ({strategy_tag})"


def _escape_latex(s: str) -> str:
    """Minimal LaTeX escaping for table cells."""
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Build the summary dict from raw oracle_sr data."""

    per_model: List[Dict[str, Any]] = []
    # Collect per-server oracle SRs across all models
    server_oracle_srs: Dict[str, List[float]] = defaultdict(list)

    for model_key, info in data.items():
        n_servers = info.get("n_servers", 0)
        if n_servers == 0:
            continue

        avg_oracle = info.get("avg_sr_oracle", 0.0)
        avg_raw = info.get("avg_sr_raw", 0.0)
        gap = avg_oracle - avg_raw

        per_model.append({
            "model": model_key,
            "model_display": _pretty_model_name(model_key),
            "strategy": info.get("strategy", ""),
            "oracle_sr": round(avg_oracle, 4),
            "raw_sr": round(avg_raw, 4),
            "n_servers": n_servers,
            "gap": round(gap, 4),
        })

        per_server = info.get("per_server", {})
        for srv_name, srv_info in per_server.items():
            sr_oracle = srv_info.get("sr_oracle")
            if sr_oracle is not None:
                server_oracle_srs[srv_name].append(sr_oracle)

    # Sort models by oracle SR descending
    per_model.sort(key=lambda x: x["oracle_sr"], reverse=True)

    # Per-server averages
    per_server_avg: List[Dict[str, Any]] = []
    for srv_name, srs in server_oracle_srs.items():
        per_server_avg.append({
            "server": srv_name,
            "avg_oracle_sr": round(_mean(srs), 4),
            "n_models": len(srs),
            "domain": "",  # domain info not in data; leave blank
        })
    per_server_avg.sort(key=lambda x: x["avg_oracle_sr"])

    hardest = per_server_avg[:10]
    easiest = list(reversed(per_server_avg[-10:]))

    all_oracle_srs = [m["oracle_sr"] for m in per_model]
    overall_avg = round(_mean(all_oracle_srs), 4)

    return {
        "overall_avg_oracle_sr": overall_avg,
        "n_models": len(per_model),
        "n_servers": len(per_server_avg),
        "per_model": per_model,
        "per_server_avg": per_server_avg,
        "hardest_servers": hardest,
        "easiest_servers": easiest,
    }


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def format_latex_table(per_model: List[Dict[str, Any]]) -> str:
    """Return a LaTeX table string for the per-model oracle SR results."""

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Oracle SR (ground-truth tools) per solver model.}")
    lines.append(r"\label{tab:oracle_sr}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & Oracle SR & Raw SR & Gap & \#Servers \\")
    lines.append(r"\midrule")

    oracle_vals: List[float] = []
    raw_vals: List[float] = []
    gap_vals: List[float] = []
    n_srv_vals: List[int] = []

    for entry in per_model:
        name = _escape_latex(entry["model_display"])
        osr = entry["oracle_sr"]
        rsr = entry["raw_sr"]
        gap = entry["gap"]
        n_srv = entry["n_servers"]

        oracle_vals.append(osr)
        raw_vals.append(rsr)
        gap_vals.append(gap)
        n_srv_vals.append(n_srv)

        lines.append(
            f"{name} & {osr:.4f} & {rsr:.4f} & {gap:+.4f} & {n_srv} \\\\"
        )

    lines.append(r"\midrule")
    avg_o = _mean(oracle_vals)
    avg_r = _mean(raw_vals)
    avg_g = _mean(gap_vals)
    avg_n = _mean([float(v) for v in n_srv_vals])
    lines.append(
        f"\\textbf{{Average}} & \\textbf{{{avg_o:.4f}}} & \\textbf{{{avg_r:.4f}}} "
        f"& \\textbf{{{avg_g:+.4f}}} & {avg_n:.0f} \\\\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper text statistics
# ---------------------------------------------------------------------------

def print_paper_stats(summary: Dict[str, Any]) -> None:
    """Print key statistics suitable for paper text."""

    per_model = summary["per_model"]
    oracle_srs = [m["oracle_sr"] for m in per_model]
    raw_srs = [m["raw_sr"] for m in per_model]
    gaps = [m["gap"] for m in per_model]

    avg_oracle = _mean(oracle_srs)
    std_oracle = _std(oracle_srs)
    avg_raw = _mean(raw_srs)
    avg_gap = _mean(gaps)

    print("\n" + "=" * 72)
    print("KEY STATISTICS FOR PAPER TEXT")
    print("=" * 72)

    print(
        f"\nWith ground-truth tools, the average SR is {avg_oracle:.2f} "
        f"(sigma={std_oracle:.2f}), establishing an upper bound for "
        f"tool-creation quality."
    )

    print(
        f"\nThe average gap between oracle and raw SR is {avg_gap:+.4f}, "
        f"indicating how much tool quality matters."
    )

    print(f"\nOverall: avg_oracle={avg_oracle:.4f}, avg_raw={avg_raw:.4f}, "
          f"n_models={summary['n_models']}, n_servers={summary['n_servers']}")

    # Hardest servers
    print("\n--- Hardest servers (lowest avg oracle SR across models) ---")
    for srv in summary["hardest_servers"]:
        print(f"  {srv['server']:50s}  avg_oracle_sr={srv['avg_oracle_sr']:.4f}  "
              f"(n_models={srv['n_models']})")

    # Easiest servers
    print("\n--- Easiest servers (highest avg oracle SR across models) ---")
    for srv in summary["easiest_servers"]:
        print(f"  {srv['server']:50s}  avg_oracle_sr={srv['avg_oracle_sr']:.4f}  "
              f"(n_models={srv['n_models']})")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format oracle SR data into LaTeX table and summary JSON."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to oracle_sr.json (default: data/oracle_sr.json relative to repo root)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for summary JSON output (default: data/oracle_sr_summary.json)",
    )
    args = parser.parse_args()

    # Resolve paths relative to the tool-genesis working directory
    repo_root = Path(__file__).resolve().parent.parent.parent
    input_path = Path(args.input) if args.input else repo_root / "data" / "oracle_sr.json"
    output_path = Path(args.output) if args.output else repo_root / "data" / "oracle_sr_summary.json"

    # Load data
    data = _load_json(input_path)
    if data is None:
        print(f"Failed to load {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(data)} model entries from {input_path}")

    # Build summary
    summary = build_summary(data)

    # Write summary JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {output_path}")

    # Print LaTeX table
    print("\n" + "=" * 72)
    print("LATEX TABLE")
    print("=" * 72 + "\n")
    latex = format_latex_table(summary["per_model"])
    print(latex)

    # Print paper stats
    print_paper_stats(summary)


if __name__ == "__main__":
    main()
