"""
Recalculate L2 Schema-F1 scores for all existing evaluation results
using the fixed local sentence-transformers/all-MiniLM-L6-v2 embeddings.

This is a READ-ONLY recalculation — existing results.json files are NOT modified.

Output: temp/eval_results_v3/schema_f1_recalc.csv
Columns: run, server, old_f1, new_f1, delta
"""
import sys
import os
import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EVAL_DIR   = REPO_ROOT / "temp" / "eval_results_v3"
BENCH_DIR  = REPO_ROOT / "temp" / "run_benchmark_v3"
GT_PATH    = REPO_ROOT / "data" / "tool_genesis_v3.json"
OUTPUT_CSV = EVAL_DIR / "schema_f1_recalc.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("recalc_schema_f1")

# ---------------------------------------------------------------------------
# Import the metric function we want to re-use
# ---------------------------------------------------------------------------
from src.env_evaluation.l2_semantic_correctness import _schema_fidelity  # noqa: E402


# ---------------------------------------------------------------------------
# Load GT data indexed by server_slug
# ---------------------------------------------------------------------------
def load_gt(path: Path) -> Dict[str, Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    index: Dict[str, Dict[str, Any]] = {}
    for item in items:
        slug = item.get("server_slug") or item.get("server_name")
        if slug:
            # _schema_fidelity expects {"tools": [...]}
            tools = item.get("tool_definitions") or []
            index[slug] = {"tools": tools}
    return index


# ---------------------------------------------------------------------------
# Helper: read JSON file safely
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(test_mode: bool = False, test_run: str = None, test_max_servers: int = 3):
    log.info("Loading GT from %s", GT_PATH)
    gt_index = load_gt(GT_PATH)
    log.info("Loaded %d GT server entries", len(gt_index))

    # Enumerate runs — skip empty directories and non-run entries
    run_dirs = sorted(
        d for d in EVAL_DIR.iterdir()
        if d.is_dir() and d.name not in {"logs"}
        and not d.name.endswith((".csv", ".xlsx"))
    )

    if test_mode and test_run:
        run_dirs = [d for d in run_dirs if d.name == test_run]
    elif test_mode:
        # pick first non-empty run
        run_dirs = run_dirs[:1]

    rows: List[Dict[str, Any]] = []
    total_servers = 0
    skipped_no_results = 0
    skipped_no_schema = 0
    skipped_no_bench = 0

    for run_dir in run_dirs:
        run_name = run_dir.name
        results_path = run_dir / "results.json"

        # Skip runs with 0 results
        if not results_path.exists():
            log.info("Skipping %s — no results.json", run_name)
            skipped_no_results += 1
            continue

        results_data = _load_json(results_path)
        if not results_data or len(results_data) == 0:
            log.info("Skipping %s — 0 results", run_name)
            skipped_no_results += 1
            continue

        debug_root = run_dir / "debug"
        if not debug_root.exists():
            log.info("Skipping %s — no debug/ directory", run_name)
            skipped_no_results += 1
            continue

        # Build a quick index from results.json: slug -> old schema_f1
        old_f1_index: Dict[str, float] = {}
        for entry in results_data:
            slug = entry.get("server_slug") or entry.get("server_id") or ""
            metrics = entry.get("metrics") or {}
            old_f1_index[slug] = float(metrics.get("schema_f1", 0.0))

        # Corresponding benchmark run directory
        bench_run_dir = BENCH_DIR / run_name

        # Iterate over debug/*/l2_debug.json
        server_debug_dirs = sorted(debug_root.iterdir()) if debug_root.exists() else []
        if test_mode and test_max_servers:
            server_debug_dirs = server_debug_dirs[:test_max_servers]

        for server_dir in server_debug_dirs:
            if not server_dir.is_dir():
                continue
            server_slug = server_dir.name
            l2_path = server_dir / "l2_debug.json"
            if not l2_path.exists():
                continue

            l2_data = _load_json(l2_path)
            if l2_data is None:
                continue

            # Old schema_f1 — prefer l2_debug.json's value over results.json
            schema_section = l2_data.get("schema") or {}
            if schema_section and "schema_f1" in schema_section:
                old_f1 = float(schema_section["schema_f1"])
            else:
                old_f1 = old_f1_index.get(server_slug, 0.0)

            # Pred schema
            pred_schema_path = bench_run_dir / server_slug / "tool_schema.json"
            if not pred_schema_path.exists():
                log.debug("No pred schema for %s/%s — skipping", run_name, server_slug)
                skipped_no_bench += 1
                continue

            pred_schema = _load_json(pred_schema_path)
            if pred_schema is None:
                skipped_no_bench += 1
                continue

            # GT schema
            gt_schema = gt_index.get(server_slug)
            if gt_schema is None:
                log.debug("No GT entry for server %s — skipping", server_slug)
                skipped_no_schema += 1
                continue

            # Recompute Schema-F1
            try:
                _, _, new_f1, _ = _schema_fidelity(pred_schema, gt_schema)
            except Exception as exc:
                log.warning("Error computing schema_f1 for %s/%s: %s", run_name, server_slug, exc)
                new_f1 = 0.0

            delta = round(new_f1 - old_f1, 6)
            rows.append({
                "run": run_name,
                "server": server_slug,
                "old_f1": round(old_f1, 6),
                "new_f1": round(new_f1, 6),
                "delta": delta,
            })
            total_servers += 1

            if test_mode:
                log.info(
                    "  [TEST] %s / %s  old=%.4f  new=%.4f  delta=%+.4f",
                    run_name, server_slug, old_f1, new_f1, delta,
                )

        log.info(
            "Run %-45s  servers processed: %d",
            run_name, sum(1 for r in rows if r["run"] == run_name),
        )

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "server", "old_f1", "new_f1", "delta"])
        writer.writeheader()
        writer.writerows(rows)

    log.info("=" * 60)
    log.info("Wrote %d rows to %s", len(rows), OUTPUT_CSV)
    log.info("Skipped (no results):  %d", skipped_no_results)
    log.info("Skipped (no bench):    %d", skipped_no_bench)
    log.info("Skipped (no GT entry): %d", skipped_no_schema)
    log.info("Total servers processed: %d", total_servers)

    # Summary statistics
    if rows:
        import statistics
        deltas = [r["delta"] for r in rows]
        non_zero_deltas = [d for d in deltas if abs(d) > 1e-6]
        log.info("Delta stats: mean=%+.4f  max=%+.4f  min=%+.4f  changed=%d/%d",
                 statistics.mean(deltas),
                 max(deltas),
                 min(deltas),
                 len(non_zero_deltas),
                 len(deltas))

    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recalculate Schema-F1 scores")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode: 1 run, 3 servers",
    )
    parser.add_argument(
        "--test-run",
        default=None,
        help="Specific run name to test (implies --test)",
    )
    parser.add_argument(
        "--test-max-servers",
        type=int,
        default=3,
        help="Max servers per run in test mode (default: 3)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full recalculation (default when --test not specified)",
    )
    args = parser.parse_args()

    test_mode = args.test or (args.test_run is not None)
    if not test_mode and not args.full:
        # default to test mode for safety
        log.warning("No mode specified — defaulting to --test mode. Use --full for full recalc.")
        test_mode = True

    run(
        test_mode=test_mode,
        test_run=args.test_run,
        test_max_servers=args.test_max_servers,
    )
