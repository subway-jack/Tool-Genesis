"""
Schema-F1 Threshold Sensitivity Analysis
=========================================
Reviewer tC8V asked: how sensitive are model rankings to the cosine-similarity
threshold τ used to match predicted tools/args to GT tools/args?

The paper states τ=0.3 is used for tool/arg name matching.

Key finding from data inspection
----------------------------------
The stored l2_debug.json results were computed with an effective threshold of τ≈0
for both tool_matches and arg_matches (the bipartite Hungarian assignment was run
and ALL resulting pairs were accepted, regardless of score).  Specifically:
  - 3.3% of tool_matches have score < 0.3 (min observed: 0.12)
  - 10.2% of arg_matches have score < 0.3
Counting ALL stored pairs at τ=0 perfectly reproduces stored tool_f1 and args_f1.

Therefore the paper's τ=0.3 represents a STRICTER recomputation relative to the
stored evaluation.  This script:
  1. Uses τ=0 (all stored matches) as the baseline — matching stored paper results.
  2. Sweeps τ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} to show ranking sensitivity.
  3. Computes Spearman ρ between the τ=0 (paper) ranking and each other threshold.

Metric variants
---------------
  tool_f1         : F1 of matched tool counts; tau applied to tool_match scores.
  schema_f1       : (tool_f1 + args_f1) / 2; tau applied to tool scores only,
                    ALL stored arg_matches for surviving tool pairs count as TP.
                    This exactly reproduces stored schema_f1 at τ=0.

Output
------
data/schema_f1_threshold_sensitivity.json
"""

import json
import math
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO_ROOT / "temp" / "eval_results_v3"
BENCH_DIR = REPO_ROOT / "temp" / "run_benchmark_v3"
GT_PATH = REPO_ROOT / "data" / "tool_genesis_v3.json"
OUT_PATH = REPO_ROOT / "data" / "schema_f1_threshold_sensitivity.json"

THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TAU_REF = 0.0   # Paper's effective threshold (all stored matches accepted)
TAU_PAPER_STATED = 0.3  # Paper's stated threshold

# Models whose bench dir has a different name
BENCH_DIR_OVERRIDES: Dict[str, List[str]] = {
    "direct_anthropic_claude-sonnet-4": [
        "direct_us-anthropic-claude-sonnet-4-20250514-v1_0",
        "direct_us.anthropic.claude-sonnet-4-20250514-v1_0",
        "direct_us.anthropic.claude-sonnet-4-20250514-v1:0",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f1(tp: int, n_pred: int, n_gt: int) -> float:
    """Standard precision/recall/F1 from counts."""
    if tp == 0:
        return 0.0
    prec = tp / max(1, n_pred)
    rec = tp / max(1, n_gt)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _extract_args(tool: dict) -> List[str]:
    """Return arg names from a tool dict (handles input_schema / inputSchema / parameters)."""
    params = (
        tool.get("input_schema")
        or tool.get("inputSchema")
        or tool.get("parameters")
    )
    if not isinstance(params, dict):
        return []
    props = params.get("properties") or {}
    if isinstance(props, dict):
        return list(props.keys())
    if isinstance(props, list):
        return [p.get("name", "") for p in props if isinstance(p, dict) and p.get("name")]
    return []


def load_gt_index(gt_path: Path) -> Dict[str, Dict]:
    """{ server_slug: { "n_tools": int, "tool_args": {tool_name: [arg_names]} } }"""
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)
    index: Dict[str, Dict] = {}
    for item in data:
        slug = item.get("server_slug") or item.get("server_name") or ""
        tools = item.get("tool_definitions") or []
        tool_args: Dict[str, List[str]] = {}
        for t in tools:
            if not isinstance(t, dict):
                continue
            tool_args[t.get("name", "")] = _extract_args(t)
        index[slug] = {"n_tools": len(tools), "tool_args": tool_args}
    return index


def load_pred_schema(bench_root: Path, model: str, server: str) -> Optional[Dict]:
    """Load tool_schema.json from bench dir; try override names if primary missing."""
    candidates: List[Path] = [bench_root / model / server / "tool_schema.json"]
    for alt in BENCH_DIR_OVERRIDES.get(model, []):
        candidates.append(bench_root / alt / server / "tool_schema.json")
    for c in candidates:
        if c.exists():
            try:
                with open(c, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def build_pred_args_index(pred_schema: Optional[dict]) -> Dict[str, List[str]]:
    """{ tool_name: [arg_names] } from pred tool_schema.json."""
    if not pred_schema:
        return {}
    return {
        t.get("name", ""): _extract_args(t)
        for t in (pred_schema.get("tools") or [])
        if isinstance(t, dict)
    }


def recompute_at_tau(
    tool_matches: List[dict],
    arg_matches: List[dict],
    gt_tool_args: Dict[str, List[str]],
    pred_tool_args: Dict[str, List[str]],
    n_pred_tools: int,
    n_gt_tools: int,
    tau: float,
) -> Dict[str, float]:
    """
    Recompute schema_f1 at a given tool-match threshold tau.

    tool_f1
    -------
    tp_tools = # stored tool_matches with score >= tau  (tau=0 -> all matches)
    n_pred   = total pred tool count (from bench schema)
    n_gt     = total GT tool count (from GT file)
    tool_f1  = F1(tp_tools, n_pred, n_gt)

    args_f1
    -------
    For each surviving tool pair (score >= tau):
      - Count ALL stored arg_matches for that pair as TP (no arg-score filtering).
        This reproduces stored args_f1 when tau=0.
      - n_gt_args  = GT arg count for that GT tool (from GT file)
      - n_pred_args = pred arg count for that pred tool (from bench schema)
    args_f1 = F1(total_tp_args, total_n_pred_args, total_n_gt_args)

    schema_f1
    ---------
    (tool_f1 + args_f1) / 2  if either > 0, else 0.
    """
    from collections import defaultdict

    # ---- Tool F1 ----
    surviving = [m for m in tool_matches if m.get("score", 0.0) >= tau]
    tp_tools = len(surviving)
    tool_f1 = _f1(tp_tools, n_pred_tools, n_gt_tools)

    # ---- Build arg lookup ----
    pair_to_ams: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for am in arg_matches:
        p_fq = am.get("pred", "")
        g_fq = am.get("gt", "")
        p_tool = p_fq.split(".")[0] if "." in p_fq else ""
        g_tool = g_fq.split(".")[0] if "." in g_fq else ""
        pair_to_ams[(p_tool, g_tool)].append(am)

    # ---- Arg F1 ----
    total_tp_args = 0
    total_n_gt_args = 0
    total_n_pred_args = 0

    for tm in surviving:
        p_name = tm["pred"]
        g_name = tm["gt"]
        gt_args = gt_tool_args.get(g_name, [])
        pair_ams = pair_to_ams.get((p_name, g_name), [])
        tp_this = len(pair_ams)           # all stored arg_matches count

        total_n_gt_args += len(gt_args)
        total_tp_args += tp_this

        # n_pred_args: prefer bench schema; fall back to tp (assumes prec=1 for this pair)
        pred_args = pred_tool_args.get(p_name)
        total_n_pred_args += len(pred_args) if pred_args is not None else tp_this

    if total_n_gt_args > 0 or total_n_pred_args > 0:
        args_f1 = _f1(total_tp_args, total_n_pred_args, total_n_gt_args)
    else:
        args_f1 = 0.0

    # ---- Schema F1 ----
    if tool_f1 > 0 or args_f1 > 0:
        schema_f1 = (tool_f1 + args_f1) / 2
    else:
        schema_f1 = 0.0

    return {
        "tool_f1": tool_f1,
        "args_f1": args_f1,
        "schema_f1": schema_f1,
        "n_surviving_tool_pairs": len(surviving),
    }


def spearman_rho(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation (handles ties)."""
    n = len(x)
    if n <= 1:
        return float("nan")

    def _rank(v: List[float]) -> List[float]:
        si = sorted(range(n), key=lambda i: v[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and v[si[j]] == v[si[i]]:
                j += 1
            avg = (i + j + 1) / 2.0
            for k in range(i, j):
                ranks[si[k]] = avg
            i = j
        return ranks

    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def _safe(v: float, d: int = 6) -> Optional[float]:
    return round(v, d) if not math.isnan(v) else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading GT index from {GT_PATH}")
    gt_index = load_gt_index(GT_PATH)
    print(f"  → {len(gt_index)} GT servers")

    eval_models = sorted(
        d.name
        for d in EVAL_DIR.iterdir()
        if d.is_dir()
        and d.name not in {"logs"}
        and not d.name.endswith((".csv", ".xlsx"))
    )
    print(f"Found {len(eval_models)} model runs\n")

    METRICS = ["tool_f1", "schema_f1"]
    # per_model[model][metric][tau] = [per-server values]
    per_model: Dict[str, Dict[str, Dict[float, List[float]]]] = {}
    all_records: List[dict] = []

    missing_bench = 0
    total_ok = 0
    skipped_no_schema = 0
    skipped_no_gt = 0
    total_tool_matches = 0
    sub_tau_tool_matches = 0
    total_arg_matches = 0
    sub_tau_arg_matches = 0

    for model in eval_models:
        debug_root = EVAL_DIR / model / "debug"
        if not debug_root.exists():
            continue

        mdata: Dict[str, Dict[float, List[float]]] = {
            m: {tau: [] for tau in THRESHOLDS} for m in METRICS
        }

        for server_dir in sorted(d for d in debug_root.iterdir() if d.is_dir()):
            server = server_dir.name
            l2_path = server_dir / "l2_debug.json"
            if not l2_path.exists():
                continue

            with open(l2_path, encoding="utf-8") as f:
                l2 = json.load(f)

            schema = l2.get("schema") or {}
            if not schema or "tool_matches" not in schema:
                skipped_no_schema += 1
                continue

            tool_matches: List[dict] = schema.get("tool_matches") or []
            arg_matches: List[dict] = schema.get("arg_matches") or []

            # Accumulate threshold-violation stats
            total_tool_matches += len(tool_matches)
            sub_tau_tool_matches += sum(
                1 for m in tool_matches if m.get("score", 1.0) < TAU_PAPER_STATED
            )
            total_arg_matches += len(arg_matches)
            sub_tau_arg_matches += sum(
                1 for a in arg_matches if a.get("score", 1.0) < TAU_PAPER_STATED
            )

            gt_info = gt_index.get(server)
            if gt_info is None:
                skipped_no_gt += 1
                continue
            n_gt_tools = gt_info["n_tools"]
            gt_tool_args = gt_info["tool_args"]
            if n_gt_tools == 0:
                skipped_no_gt += 1
                continue

            pred_schema = load_pred_schema(BENCH_DIR, model, server)
            if pred_schema is None:
                missing_bench += 1
                pred_tool_args: Dict[str, List[str]] = {}
                # Best available n_pred_tools: count from tool_matches (lower bound)
                n_pred_tools = len(set(m["pred"] for m in tool_matches)) if tool_matches else 0
            else:
                pred_tool_args = build_pred_args_index(pred_schema)
                n_pred_tools = len(pred_schema.get("tools") or [])

            stored_schema_f1 = schema.get("schema_f1", 0.0)
            stored_tool_f1 = schema.get("tool_f1", 0.0)
            stored_args_f1 = schema.get("args_f1", 0.0)

            record: dict = {
                "model": model,
                "server": server,
                "n_pred_tools": n_pred_tools,
                "n_gt_tools": n_gt_tools,
                "bench_available": pred_schema is not None,
                "stored_tool_f1": stored_tool_f1,
                "stored_args_f1": stored_args_f1,
                "stored_schema_f1": stored_schema_f1,
                "thresholds": {},
            }

            for tau in THRESHOLDS:
                res = recompute_at_tau(
                    tool_matches=tool_matches,
                    arg_matches=arg_matches,
                    gt_tool_args=gt_tool_args,
                    pred_tool_args=pred_tool_args,
                    n_pred_tools=n_pred_tools,
                    n_gt_tools=n_gt_tools,
                    tau=tau,
                )
                record["thresholds"][str(tau)] = {k: _safe(v) for k, v in res.items()}
                for metric in METRICS:
                    v = res[metric]
                    if not math.isnan(v):
                        mdata[metric][tau].append(v)

            all_records.append(record)
            total_ok += 1

        per_model[model] = mdata

    print(f"Processed {total_ok} server×model combinations")
    print(f"Skipped (no schema):   {skipped_no_schema}")
    print(f"Skipped (no GT entry): {skipped_no_gt}")
    print(f"Missing bench schema:  {missing_bench} (n_pred lower-bounded)\n")

    pct_tool = 100 * sub_tau_tool_matches / max(1, total_tool_matches)
    pct_arg = 100 * sub_tau_arg_matches / max(1, total_arg_matches)
    print("Data quality finding:")
    print(f"  tool_matches with score < {TAU_PAPER_STATED}: "
          f"{sub_tau_tool_matches}/{total_tool_matches} ({pct_tool:.1f}%)")
    print(f"  arg_matches  with score < {TAU_PAPER_STATED}: "
          f"{sub_tau_arg_matches}/{total_arg_matches} ({pct_arg:.1f}%)")
    print(f"  → Stored results used effective τ≈0 (full bipartite assignment).")
    print(f"  → τ=0 in this table reproduces paper's stored schema_f1.\n")

    # -------------------------------------------------------------------------
    # Per-model averages
    # -------------------------------------------------------------------------
    model_avg: Dict[str, Dict[str, Dict[float, float]]] = {}
    for model, mdata in per_model.items():
        model_avg[model] = {}
        for metric in METRICS:
            model_avg[model][metric] = {
                tau: (sum(v) / len(v) if v else float("nan"))
                for tau, v in mdata[metric].items()
            }

    primary = "schema_f1"
    models_ok = [
        m for m in model_avg
        if not math.isnan(model_avg[m][primary].get(TAU_REF, float("nan")))
        and len(per_model[m][primary][TAU_REF]) > 0
    ]
    print(f"Models with data at τ={TAU_REF}: {len(models_ok)}\n")

    # -------------------------------------------------------------------------
    # Spearman ρ vs τ=0 (paper baseline)
    # -------------------------------------------------------------------------
    spearman: Dict[str, Dict[float, float]] = {}
    for metric in METRICS:
        ref = [model_avg[m][metric][TAU_REF] for m in models_ok]
        spearman[metric] = {}
        for tau in THRESHOLDS:
            taus = [model_avg[m][metric][tau] for m in models_ok]
            valid = [(r, t) for r, t in zip(ref, taus) if not math.isnan(t)]
            if len(valid) < 2:
                spearman[metric][tau] = float("nan")
            else:
                rv, tv = zip(*valid)
                spearman[metric][tau] = spearman_rho(list(rv), list(tv))

    # -------------------------------------------------------------------------
    # Print table
    # -------------------------------------------------------------------------
    col_w = 14
    n_cols = len(THRESHOLDS)
    total_w = 45 + col_w * n_cols
    print("=" * total_w)
    print(f"Per-model average {primary} at each threshold")
    print("(τ=0.0 = paper's stored result; higher τ = stricter match threshold)")
    print("=" * total_w)
    header = f"{'Model':<45}" + "".join(
        f"τ={t:.1f}{' (paper)':<{col_w-5} if t==TAU_PAPER_STATED else '':<{col_w-4}}"
        if False else f"τ={t:<{col_w-2}}"
        for t in THRESHOLDS
    )
    print(header)
    print("-" * total_w)

    sorted_models = sorted(
        models_ok,
        key=lambda m: model_avg[m][primary][TAU_REF],
        reverse=True,
    )
    for model in sorted_models:
        row = f"{model:<45}"
        for tau in THRESHOLDS:
            v = model_avg[model][primary][tau]
            row += f"{(f'{v:.4f}' if not math.isnan(v) else 'N/A'):<{col_w}}"
        print(row)

    print()
    print("Spearman ρ vs τ=0 (paper) rankings:")
    for metric in METRICS:
        print(f"\n  [{metric}]")
        for tau in THRESHOLDS:
            rho = spearman[metric][tau]
            marker = " ← reference (paper)" if tau == TAU_REF else (
                " ← paper-stated τ" if tau == TAU_PAPER_STATED else ""
            )
            rho_s = f"{rho:.6f}" if not math.isnan(rho) else "N/A"
            print(f"    τ={tau:.1f}: ρ = {rho_s}{marker}")

    # -------------------------------------------------------------------------
    # Rankings table
    # -------------------------------------------------------------------------
    tau_rankings: Dict[float, List[str]] = {}
    for tau in THRESHOLDS:
        valid = [
            (m, model_avg[m][primary][tau]) for m in models_ok
            if not math.isnan(model_avg[m][primary][tau])
        ]
        valid.sort(key=lambda x: x[1], reverse=True)
        tau_rankings[tau] = [m for m, _ in valid]

    print()
    print(f"Rank table (by {primary}, sorted by τ=0):")
    rk_w = 9
    print(f"{'Model':<45}" + "".join(f"τ={t:<{rk_w-2}}" for t in THRESHOLDS))
    print("-" * (45 + rk_w * n_cols))
    for model in sorted_models:
        row = f"{model:<45}"
        for tau in THRESHOLDS:
            rank = tau_rankings[tau].index(model) + 1 if model in tau_rankings[tau] else None
            row += f"{(str(rank) if rank is not None else 'N/A'):<{rk_w}}"
        print(row)

    # -------------------------------------------------------------------------
    # Max rank change summary
    # -------------------------------------------------------------------------
    print()
    ref_ranking = tau_rankings[TAU_REF]
    print(f"Rank stability relative to τ=0 ({primary}):")
    for tau in THRESHOLDS:
        if tau == TAU_REF:
            continue
        changes = [
            abs(ref_ranking.index(m) - tau_rankings[tau].index(m))
            for m in sorted_models
            if m in tau_rankings[tau]
        ]
        if changes:
            print(f"  τ=0 → τ={tau:.1f}: max_rank_change={max(changes):3d}, "
                  f"mean={sum(changes)/len(changes):.1f}, "
                  f"ρ={spearman[primary][tau]:.4f}")

    # -------------------------------------------------------------------------
    # Output JSON
    # -------------------------------------------------------------------------
    output = {
        "description": (
            "Schema-F1 threshold sensitivity analysis. "
            f"τ=0 reproduces stored paper results (all bipartite matches accepted). "
            f"τ={TAU_PAPER_STATED} is the threshold stated in the paper. "
            "For each τ, tool_matches with score<τ are dropped; "
            "all stored arg_matches for surviving tool pairs count as TP."
        ),
        "thresholds": THRESHOLDS,
        "reference_tau": TAU_REF,
        "paper_stated_tau": TAU_PAPER_STATED,
        "primary_metric": primary,
        "n_models": len(models_ok),
        "data_quality": {
            "total_tool_matches": total_tool_matches,
            "tool_matches_below_stated_tau": sub_tau_tool_matches,
            "pct_tool_matches_below_stated_tau": round(pct_tool, 2),
            "total_arg_matches": total_arg_matches,
            "arg_matches_below_stated_tau": sub_tau_arg_matches,
            "pct_arg_matches_below_stated_tau": round(pct_arg, 2),
            "interpretation": (
                f"Stored evaluation used effective τ≈0 (full bipartite assignment accepted). "
                f"The paper-stated τ={TAU_PAPER_STATED} represents a stricter re-thresholding. "
                f"τ=0.0 in this analysis exactly reproduces stored schema_f1 values."
            ),
        },
        "spearman_rho_vs_tau0": {
            metric: {
                str(tau): _safe(rho, 6)
                for tau, rho in spearman[metric].items()
            }
            for metric in METRICS
        },
        "model_avg_schema_f1": {
            model: {
                str(tau): _safe(model_avg[model]["schema_f1"][tau], 6)
                for tau in THRESHOLDS
            }
            for model in sorted_models
        },
        "model_avg_tool_f1": {
            model: {
                str(tau): _safe(model_avg[model]["tool_f1"][tau], 6)
                for tau in THRESHOLDS
            }
            for model in sorted_models
        },
        "model_ranks": {
            model: {
                str(tau): (tau_rankings[tau].index(model) + 1
                           if model in tau_rankings[tau] else None)
                for tau in THRESHOLDS
            }
            for model in sorted_models
        },
        "n_servers_per_model": {
            model: {str(tau): len(per_model[model]["schema_f1"][tau]) for tau in THRESHOLDS}
            for model in sorted_models
        },
        "per_server_detail": all_records,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
