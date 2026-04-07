#!/usr/bin/env python3
"""Regenerate data/all_models_by_size.json from ground-truth eval results.

Reads every temp/eval_results_v3/{model_name}/results.json, computes
per-model aggregate metrics, and writes the canonical JSON file.

Metric definitions (denominator is always N = total benchmark servers = 86):
  n           -- number of unique servers evaluated for this model
  compliance  -- count(compliance == true) / N
  exec        -- count(server_launch_success > 0) / N
  schema_f1   -- sum(schema_f1) / N
  cov         -- count(schema_f1 > 0) / N   (tool coverage)
  ut_soft     -- sum(tool_call_success_rate_soft) / N
  ut_hard     -- sum(tool_call_success_rate_hard) / N
  sr          -- sum(trajectory_level_validation_rate_soft) / N

Duplicate server slugs within a single model are deduplicated (last wins).
Models are sorted: direct_ first, then coder_agent_, alphabetically within.
"""

import json
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "temp" / "eval_results_v3"
OUTPUT_PATH = REPO_ROOT / "data" / "all_models_by_size.json"
BACKUP_PATH = OUTPUT_PATH.with_suffix(".json.bak")

# Total benchmark size (number of servers in the canonical benchmark set)
N_BENCHMARK = 86


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _num(x) -> float:
    """Coerce to float, treating None / bool / string gracefully."""
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _sort_key(name: str):
    """Sort direct_ models first, then coder_agent_, alphabetically within."""
    if name.startswith("direct_"):
        return (0, name)
    if name.startswith("coder_agent_"):
        return (1, name)
    return (2, name)


# ---------------------------------------------------------------------------
# Core: load one model's results and aggregate
# ---------------------------------------------------------------------------
def aggregate_model(results_path: Path) -> dict:
    """Return aggregated metrics dict for one model directory."""
    with results_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    # Deduplicate by server_slug, keeping the last occurrence
    by_slug: dict = {}
    for rec in records:
        slug = rec.get("server_slug") or rec.get("server_id") or ""
        by_slug[slug] = rec
    deduped = list(by_slug.values())

    n = len(deduped)
    N = N_BENCHMARK

    compliance_count = 0
    launched_count = 0
    schema_f1_sum = 0.0
    cov_count = 0
    ut_soft_sum = 0.0
    ut_hard_sum = 0.0
    sr_sum = 0.0

    for rec in deduped:
        m = rec.get("metrics") or {}

        # compliance: boolean flag
        if m.get("compliance"):
            compliance_count += 1

        # exec: did the server launch at least once?
        launch_success = _num(m.get("server_launch_success", 0))
        if launch_success > 0:
            launched_count += 1

        # schema_f1
        sf1 = _num(m.get("schema_f1", 0))
        schema_f1_sum += sf1
        if sf1 > 0:
            cov_count += 1

        # unit-test rates
        ut_soft_sum += _num(m.get("tool_call_success_rate_soft", 0))
        ut_hard_sum += _num(m.get("tool_call_success_rate_hard", 0))

        # trajectory-level success rate (soft)
        sr_sum += _num(m.get("trajectory_level_validation_rate_soft", 0))

    return {
        "n": n,
        "compliance": round(compliance_count / N, 3),
        "exec": round(launched_count / N, 3),
        "schema_f1": round(schema_f1_sum / N, 3),
        "cov": round(cov_count / N, 3),
        "ut_soft": round(ut_soft_sum / N, 3),
        "ut_hard": round(ut_hard_sum / N, 3),
        "sr": round(sr_sum / N, 3),
    }


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------
def compare_old_new(old: dict, new: dict):
    """Print a human-readable diff between old and new all_models dicts."""
    all_keys = sorted(set(list(old.keys()) + list(new.keys())), key=_sort_key)
    metric_keys = ["n", "compliance", "exec", "schema_f1", "cov", "ut_soft", "ut_hard", "sr"]

    added = [k for k in all_keys if k not in old]
    removed = [k for k in all_keys if k not in new]
    common = [k for k in all_keys if k in old and k in new]

    if added:
        print(f"\n  NEW models ({len(added)}):")
        for m in added:
            print(f"    + {m}")
    if removed:
        print(f"\n  REMOVED models ({len(removed)}):")
        for m in removed:
            print(f"    - {m}")

    changed_count = 0
    max_delta = 0.0
    max_delta_model = ""
    max_delta_metric = ""

    print(f"\n  {'Model':<52} {'Metric':<12} {'Old':>8} {'New':>8} {'Delta':>8}")
    print("  " + "-" * 92)

    for model in common:
        o = old[model]
        n_ = new[model]
        model_changed = False
        for mk in metric_keys:
            ov = o.get(mk, 0)
            nv = n_.get(mk, 0)
            if mk == "n":
                delta = nv - ov
                if delta != 0:
                    print(f"  {model:<52} {mk:<12} {ov:>8} {nv:>8} {delta:>+8}")
                    model_changed = True
            else:
                delta = nv - ov
                if abs(delta) > 0.0005:
                    print(f"  {model:<52} {mk:<12} {ov:>8.3f} {nv:>8.3f} {delta:>+8.3f}")
                    model_changed = True
                    if abs(delta) > abs(max_delta):
                        max_delta = delta
                        max_delta_model = model
                        max_delta_metric = mk
        if model_changed:
            changed_count += 1

    print()
    print(f"  Total models: old={len(old)}, new={len(new)}")
    print(f"  Changed models: {changed_count}")
    if max_delta_model:
        print(f"  Largest delta: {max_delta_model} / {max_delta_metric} = {max_delta:+.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not EVAL_DIR.is_dir():
        print(f"ERROR: eval directory not found: {EVAL_DIR}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Collect all model results
    # ------------------------------------------------------------------
    all_models: dict = {}
    skipped = []

    for entry in sorted(EVAL_DIR.iterdir()):
        if not entry.is_dir():
            continue
        results_path = entry / "results.json"
        if not results_path.exists():
            skipped.append(entry.name)
            continue

        model_name = entry.name
        agg = aggregate_model(results_path)
        all_models[model_name] = agg

    # Sort by strategy (direct first) then alphabetically
    sorted_models = dict(sorted(all_models.items(), key=lambda kv: _sort_key(kv[0])))

    print(f"Processed {len(sorted_models)} models from {EVAL_DIR}")
    if skipped:
        print(f"Skipped (no results.json): {', '.join(skipped)}")

    # ------------------------------------------------------------------
    # 2. Load old file (if any) for comparison
    # ------------------------------------------------------------------
    old_data: dict = {}
    if OUTPUT_PATH.exists():
        try:
            with OUTPUT_PATH.open("r", encoding="utf-8") as f:
                old_data = json.load(f)
        except Exception:
            old_data = {}

    # ------------------------------------------------------------------
    # 3. Save backup then write new file
    # ------------------------------------------------------------------
    if OUTPUT_PATH.exists():
        shutil.copy2(OUTPUT_PATH, BACKUP_PATH)
        print(f"Backup saved: {BACKUP_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(sorted_models, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Written: {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # 4. Print summary table
    # ------------------------------------------------------------------
    metric_keys = ["compliance", "exec", "schema_f1", "cov", "ut_soft", "ut_hard", "sr"]
    header = f"  {'Model':<52} {'n':>4}  " + "  ".join(f"{k:>10}" for k in metric_keys)
    print("\n" + header)
    print("  " + "-" * len(header))
    for model, vals in sorted_models.items():
        row = f"  {model:<52} {vals['n']:>4}  "
        row += "  ".join(f"{vals[k]:>10.3f}" for k in metric_keys)
        print(row)

    # ------------------------------------------------------------------
    # 5. Compare old vs new
    # ------------------------------------------------------------------
    if old_data:
        print("\n" + "=" * 96)
        print("  COMPARISON: old vs new")
        print("=" * 96)
        compare_old_new(old_data, sorted_models)

        # Specifically check direct_qwen3-32b
        print("\n" + "=" * 96)
        print("  SPOTLIGHT: direct_qwen3-32b")
        print("=" * 96)
        old_q = old_data.get("direct_qwen3-32b", {})
        new_q = sorted_models.get("direct_qwen3-32b", {})
        if old_q and new_q:
            for k in ["n"] + metric_keys:
                ov = old_q.get(k, "N/A")
                nv = new_q.get(k, "N/A")
                delta = ""
                if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
                    delta = f"  delta={nv - ov:+.3f}" if k != "n" else f"  delta={nv - ov:+d}"
                print(f"    {k:<12}  old={ov}  new={nv}{delta}")
    else:
        print("\n  (No old file to compare against)")


if __name__ == "__main__":
    main()
