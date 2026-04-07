#!/usr/bin/env python3
"""Structural equivalence analysis for Schema-F1.

Measures how often models generate a different number of tools than the
ground truth and how that tool-count mismatch affects Schema-F1 scores.
Addresses reviewer concern jyMS-W3.

Data sources
------------
* GT tool counts        : data/tool_genesis_v3.json   (tool_definitions per server_slug)
* Pred tool counts      : temp/run_benchmark_v3/{model}/{server}/tool_schema.json
* Schema-F1 + matches   : temp/eval_results_v3/{model}/debug/{server}/l2_debug.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GT_DATA_PATH = PROJECT_ROOT / "data" / "tool_genesis_v3.json"
EVAL_RESULTS_DIR = PROJECT_ROOT / "temp" / "eval_results_v3"
RUN_BENCHMARK_DIR = PROJECT_ROOT / "temp" / "run_benchmark_v3"
OUTPUT_PATH = PROJECT_ROOT / "data" / "structural_equivalence_analysis.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gt_tool_counts() -> Dict[str, int]:
    """Return {server_slug: n_gt_tools} from the ground-truth dataset."""
    with open(GT_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        item["server_slug"]: len(item.get("tool_definitions", []))
        for item in data
        if isinstance(item, dict) and "server_slug" in item
    }


def load_pred_tool_count(model: str, server: str) -> Optional[int]:
    """Return the number of predicted tools from tool_schema.json, or None."""
    path = RUN_BENCHMARK_DIR / model / server / "tool_schema.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tools = data.get("tools", [])
        return len(tools)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def load_l2_debug(model: str, server: str) -> Optional[Dict[str, Any]]:
    """Return the schema section of l2_debug.json, or None."""
    path = EVAL_RESULTS_DIR / model / "debug" / server / "l2_debug.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return None
        schema = data.get("schema")
        if not schema or not isinstance(schema, dict):
            return None
        return schema
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Record type
# ---------------------------------------------------------------------------

class Record:
    __slots__ = (
        "model", "server", "n_pred", "n_gt", "n_matched",
        "tool_count_diff", "schema_f1", "tool_f1", "args_f1",
    )

    def __init__(
        self,
        model: str,
        server: str,
        n_pred: int,
        n_gt: int,
        n_matched: int,
        schema_f1: float,
        tool_f1: float,
        args_f1: float,
    ):
        self.model = model
        self.server = server
        self.n_pred = n_pred
        self.n_gt = n_gt
        self.n_matched = n_matched
        self.tool_count_diff = n_pred - n_gt
        self.schema_f1 = schema_f1
        self.tool_f1 = tool_f1
        self.args_f1 = args_f1


# ---------------------------------------------------------------------------
# Main collection
# ---------------------------------------------------------------------------

def collect_records(gt_counts: Dict[str, int]) -> List[Record]:
    """Iterate over all (model, server) pairs and build Record list."""
    records: List[Record] = []

    if not EVAL_RESULTS_DIR.is_dir():
        print(f"[WARN] eval_results_v3 directory not found: {EVAL_RESULTS_DIR}")
        return records

    models = sorted(
        d for d in os.listdir(EVAL_RESULTS_DIR)
        if (EVAL_RESULTS_DIR / d / "debug").is_dir()
    )

    for model in models:
        debug_dir = EVAL_RESULTS_DIR / model / "debug"
        servers = sorted(
            s for s in os.listdir(debug_dir)
            if (debug_dir / s).is_dir()
        )
        for server in servers:
            # Load l2_debug schema section
            schema_debug = load_l2_debug(model, server)
            if schema_debug is None:
                continue  # empty or missing

            schema_f1 = schema_debug.get("schema_f1", 0.0)
            tool_f1 = schema_debug.get("tool_f1", 0.0)
            args_f1 = schema_debug.get("args_f1", 0.0)
            n_matched = len(schema_debug.get("tool_matches", []))

            # GT tool count
            n_gt = gt_counts.get(server)
            if n_gt is None:
                continue  # server not in GT dataset

            # Pred tool count from tool_schema.json
            n_pred = load_pred_tool_count(model, server)
            if n_pred is None:
                # Fallback: if no run_benchmark_v3 entry, skip
                continue

            records.append(Record(
                model=model,
                server=server,
                n_pred=n_pred,
                n_gt=n_gt,
                n_matched=n_matched,
                schema_f1=schema_f1,
                tool_f1=tool_f1,
                args_f1=args_f1,
            ))

    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(records: List[Record]) -> Dict[str, Any]:
    """Compute all aggregation metrics from records."""

    total = len(records)
    if total == 0:
        return {"error": "no records collected"}

    # --- Categorize ---
    exact = [r for r in records if r.tool_count_diff == 0]
    over = [r for r in records if r.tool_count_diff > 0]
    under = [r for r in records if r.tool_count_diff < 0]

    def avg(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    tool_count_distribution = {
        "exact_match": {
            "count": len(exact),
            "pct": round(len(exact) / total * 100, 2),
            "avg_schema_f1": round(avg([r.schema_f1 for r in exact]), 4),
            "avg_tool_f1": round(avg([r.tool_f1 for r in exact]), 4),
            "avg_args_f1": round(avg([r.args_f1 for r in exact]), 4),
        },
        "over_generation": {
            "count": len(over),
            "pct": round(len(over) / total * 100, 2),
            "avg_schema_f1": round(avg([r.schema_f1 for r in over]), 4),
            "avg_tool_f1": round(avg([r.tool_f1 for r in over]), 4),
            "avg_args_f1": round(avg([r.args_f1 for r in over]), 4),
            "avg_excess": round(avg([r.tool_count_diff for r in over]), 2),
        },
        "under_generation": {
            "count": len(under),
            "pct": round(len(under) / total * 100, 2),
            "avg_schema_f1": round(avg([r.schema_f1 for r in under]), 4),
            "avg_tool_f1": round(avg([r.tool_f1 for r in under]), 4),
            "avg_args_f1": round(avg([r.args_f1 for r in under]), 4),
            "avg_deficit": round(avg([r.tool_count_diff for r in under]), 2),
        },
    }

    # --- Histogram of tool_count_diff ---
    diff_counts: Dict[int, int] = defaultdict(int)
    for r in records:
        diff_counts[r.tool_count_diff] += 1
    # Sort by key for readability
    histogram = {str(k): diff_counts[k] for k in sorted(diff_counts.keys())}

    # --- Schema-F1 by diff bucket ---
    diff_f1s: Dict[int, List[float]] = defaultdict(list)
    for r in records:
        diff_f1s[r.tool_count_diff].append(r.schema_f1)
    schema_f1_by_diff = {
        str(k): round(avg(diff_f1s[k]), 4) for k in sorted(diff_f1s.keys())
    }

    # --- Per-model summary ---
    model_records: Dict[str, List[Record]] = defaultdict(list)
    for r in records:
        model_records[r.model].append(r)

    per_model_summary = {}
    for model in sorted(model_records.keys()):
        mrs = model_records[model]
        n_exact = sum(1 for r in mrs if r.tool_count_diff == 0)
        per_model_summary[model] = {
            "n_servers": len(mrs),
            "exact_match_pct": round(n_exact / len(mrs) * 100, 2),
            "avg_tool_count_diff": round(avg([r.tool_count_diff for r in mrs]), 2),
            "avg_schema_f1": round(avg([r.schema_f1 for r in mrs]), 4),
            "avg_tool_f1": round(avg([r.tool_f1 for r in mrs]), 4),
            "n_over": sum(1 for r in mrs if r.tool_count_diff > 0),
            "n_under": sum(1 for r in mrs if r.tool_count_diff < 0),
        }

    return {
        "total_model_server_pairs": total,
        "tool_count_distribution": tool_count_distribution,
        "tool_count_diff_histogram": histogram,
        "schema_f1_by_diff": schema_f1_by_diff,
        "per_model_summary": per_model_summary,
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_summary(result: Dict[str, Any]) -> None:
    """Print human-readable tables to stdout."""

    print("=" * 80)
    print("  Structural Equivalence Analysis — Schema-F1 vs Tool-Count Mismatch")
    print("=" * 80)
    print()

    total = result["total_model_server_pairs"]
    dist = result["tool_count_distribution"]

    print(f"Total (model, server) pairs analysed: {total}")
    print()

    # --- Distribution table ---
    print("--- Tool Count Distribution ---")
    fmt = "{:<20s} {:>6s} {:>7s} {:>12s} {:>10s} {:>10s}"
    print(fmt.format("Category", "Count", "Pct(%)", "Schema-F1", "Tool-F1", "Args-F1"))
    print("-" * 70)
    for cat, label in [("exact_match", "Exact match"),
                       ("over_generation", "Over-generation"),
                       ("under_generation", "Under-generation")]:
        d = dist[cat]
        extra = ""
        if cat == "over_generation" and "avg_excess" in d:
            extra = f"  (avg excess: +{d['avg_excess']:.1f})"
        if cat == "under_generation" and "avg_deficit" in d:
            extra = f"  (avg deficit: {d['avg_deficit']:.1f})"
        print(fmt.format(
            label + extra,
            str(d["count"]),
            f"{d['pct']:.1f}",
            f"{d['avg_schema_f1']:.4f}",
            f"{d['avg_tool_f1']:.4f}",
            f"{d['avg_args_f1']:.4f}",
        ))
    print()

    # --- Schema-F1 by diff ---
    print("--- Schema-F1 by Tool-Count Difference (pred - gt) ---")
    hdr = "{:<8s} {:>6s} {:>12s}"
    print(hdr.format("Diff", "Count", "Avg Schema-F1"))
    print("-" * 30)
    hist = result["tool_count_diff_histogram"]
    f1_by = result["schema_f1_by_diff"]
    for diff_str in sorted(hist.keys(), key=lambda x: int(x)):
        cnt = hist[diff_str]
        f1 = f1_by.get(diff_str, 0.0)
        print(hdr.format(diff_str, str(cnt), f"{f1:.4f}"))
    print()

    # --- Per-model table ---
    print("--- Per-Model Summary ---")
    mfmt = "{:<55s} {:>5s} {:>8s} {:>9s} {:>10s} {:>5s} {:>5s}"
    print(mfmt.format("Model", "N", "Exact%", "AvgDiff", "Schema-F1", "Over", "Under"))
    print("-" * 105)
    pms = result["per_model_summary"]
    for model in sorted(pms.keys()):
        m = pms[model]
        print(mfmt.format(
            model[:55],
            str(m["n_servers"]),
            f"{m['exact_match_pct']:.1f}",
            f"{m['avg_tool_count_diff']:+.2f}",
            f"{m['avg_schema_f1']:.4f}",
            str(m["n_over"]),
            str(m["n_under"]),
        ))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[INFO] Loading GT data from {GT_DATA_PATH}")
    gt_counts = load_gt_tool_counts()
    print(f"[INFO] GT servers: {len(gt_counts)}")

    print(f"[INFO] Scanning eval results in {EVAL_RESULTS_DIR}")
    records = collect_records(gt_counts)
    print(f"[INFO] Collected {len(records)} (model, server) records with valid data")

    if not records:
        print("[ERROR] No records found. Exiting.")
        sys.exit(1)

    result = analyze(records)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Results saved to {OUTPUT_PATH}")
    print()

    print_summary(result)


if __name__ == "__main__":
    main()
