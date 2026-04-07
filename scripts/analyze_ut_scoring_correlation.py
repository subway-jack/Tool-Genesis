#!/usr/bin/env python3
"""
Analyze correlation between struct_score and embed_score in UT evaluation.
Reads all l2_debug.json files from temp/eval_results_v3/*/debug/*/l2_debug.json
"""

import json
import os
import glob
import math
import statistics
from pathlib import Path


def pearson_corr(x, y):
    n = len(x)
    if n < 2:
        return float('nan')
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y))
    if den == 0:
        return float('nan')
    return num / den


def spearman_corr(x, y):
    n = len(x)
    if n < 2:
        return float('nan')

    def rank(arr):
        sorted_idx = sorted(range(n), key=lambda i: arr[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and arr[sorted_idx[j + 1]] == arr[sorted_idx[j]]:
                j += 1
            avg_rank = (i + j) / 2 + 1  # 1-indexed average rank
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(x), rank(y)
    return pearson_corr(rx, ry)


def distribution_stats(arr):
    if not arr:
        return {}
    n = len(arr)
    sorted_arr = sorted(arr)
    mean = sum(arr) / n
    variance = sum((v - mean) ** 2 for v in arr) / n
    std = math.sqrt(variance)

    def percentile(p):
        idx = (p / 100) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return sorted_arr[lo] * (1 - frac) + sorted_arr[hi] * frac

    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(sorted_arr[0], 6),
        "q1": round(percentile(25), 6),
        "median": round(percentile(50), 6),
        "q3": round(percentile(75), 6),
        "max": round(sorted_arr[-1], 6),
    }


def biserial_corr(continuous, binary):
    """Point-biserial correlation between a continuous variable and a binary variable."""
    n = len(continuous)
    if n < 2:
        return float('nan')
    ones = [c for c, b in zip(continuous, binary) if b]
    zeros = [c for c, b in zip(continuous, binary) if not b]
    n1, n0 = len(ones), len(zeros)
    if n1 == 0 or n0 == 0:
        return float('nan')
    m1 = sum(ones) / n1
    m0 = sum(zeros) / n0
    mean_all = sum(continuous) / n
    variance = sum((v - mean_all) ** 2 for v in continuous) / n
    sd = math.sqrt(variance) if variance > 0 else 0
    if sd == 0:
        return float('nan')
    prop1 = n1 / n
    prop0 = n0 / n
    return (m1 - m0) / sd * math.sqrt(prop1 * prop0)


def main():
    base_dir = Path("/Users/subway/code/python/项目/Multi-agent/tool-genesis/tool-genesis")
    pattern = str(base_dir / "temp/eval_results_v3/*/debug/*/l2_debug.json")

    files = glob.glob(pattern)
    print(f"Found {len(files)} l2_debug.json files")

    struct_scores = []
    embed_scores = []
    final_scores = []
    hard_passes = []

    file_count = 0
    detail_count = 0
    skipped_files = 0

    per_model_data = {}

    for fpath in sorted(files):
        parts = Path(fpath).parts
        # Extract model name and task name from path
        # .../eval_results_v3/<model>/debug/<task>/l2_debug.json
        try:
            debug_idx = parts.index("debug")
            model_name = parts[debug_idx - 1]
            task_name = parts[debug_idx + 1]
        except (ValueError, IndexError):
            model_name = "unknown"
            task_name = "unknown"

        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            skipped_files += 1
            continue

        unit_tests = data.get("unit_tests", {})
        details = unit_tests.get("details", [])

        if not details:
            continue

        file_count += 1

        if model_name not in per_model_data:
            per_model_data[model_name] = {
                "struct_scores": [], "embed_scores": [],
                "final_scores": [], "hard_passes": []
            }

        for item in details:
            ss = item.get("struct_score")
            es = item.get("embed_score")
            fs = item.get("final_score")
            hp = item.get("hard_pass")

            if ss is None or es is None:
                continue

            struct_scores.append(float(ss))
            embed_scores.append(float(es))
            final_scores.append(float(fs) if fs is not None else 0.5 * float(ss) + 0.5 * float(es))
            hard_passes.append(1 if hp else 0)

            per_model_data[model_name]["struct_scores"].append(float(ss))
            per_model_data[model_name]["embed_scores"].append(float(es))
            per_model_data[model_name]["final_scores"].append(float(fs) if fs is not None else 0.5 * float(ss) + 0.5 * float(es))
            per_model_data[model_name]["hard_passes"].append(1 if hp else 0)

            detail_count += 1

    print(f"Processed {file_count} files with unit test details (skipped {skipped_files} unreadable)")
    print(f"Total unit test cases: {detail_count}")
    print(f"Models: {len(per_model_data)}")

    # --- Global correlations ---
    pearson_se = pearson_corr(struct_scores, embed_scores)
    spearman_se = spearman_corr(struct_scores, embed_scores)

    pearson_s_hp = biserial_corr(struct_scores, hard_passes)
    pearson_e_hp = biserial_corr(embed_scores, hard_passes)

    # Use spearman for struct/embed vs hard_pass (binary, rank-based)
    spearman_s_hp = spearman_corr(struct_scores, hard_passes)
    spearman_e_hp = spearman_corr(embed_scores, hard_passes)

    spearman_f_hp = spearman_corr(final_scores, hard_passes)
    biserial_f_hp = biserial_corr(final_scores, hard_passes)

    # --- Distribution stats ---
    struct_dist = distribution_stats(struct_scores)
    embed_dist = distribution_stats(embed_scores)
    final_dist = distribution_stats(final_scores)

    hard_pass_rate = sum(hard_passes) / len(hard_passes) if hard_passes else 0

    # --- Score agreement analysis ---
    # Cases where struct=1 (perfect structure match)
    struct_perfect = [i for i, s in enumerate(struct_scores) if s == 1.0]
    embed_when_struct_perfect = [embed_scores[i] for i in struct_perfect]

    # Cases where struct=0 (no structure match)
    struct_zero = [i for i, s in enumerate(struct_scores) if s == 0.0]
    embed_when_struct_zero = [embed_scores[i] for i in struct_zero]

    # Hard pass rate by struct score bracket
    brackets = [(0.0, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 1.0)]
    bracket_analysis = {}
    for lo, hi in brackets:
        if lo == hi:
            idxs = [i for i, s in enumerate(struct_scores) if abs(s - lo) < 1e-9]
        else:
            idxs = [i for i, s in enumerate(struct_scores) if lo < s <= hi]
        if idxs:
            hp_rate = sum(hard_passes[i] for i in idxs) / len(idxs)
            avg_embed = sum(embed_scores[i] for i in idxs) / len(idxs)
            key = f"struct={'%.1f'%lo if lo==hi else '(%.1f,%.1f]'%(lo,hi)}"
            bracket_analysis[key] = {
                "n": len(idxs),
                "hard_pass_rate": round(hp_rate, 4),
                "avg_embed_score": round(avg_embed, 4),
            }

    # Embed score brackets
    embed_brackets = [(0.0, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 1.0)]
    embed_bracket_analysis = {}
    for lo, hi in embed_brackets:
        if lo == hi:
            idxs = [i for i, e in enumerate(embed_scores) if abs(e - lo) < 1e-9]
        else:
            idxs = [i for i, e in enumerate(embed_scores) if lo < e <= hi]
        if idxs:
            hp_rate = sum(hard_passes[i] for i in idxs) / len(idxs)
            avg_struct = sum(struct_scores[i] for i in idxs) / len(idxs)
            key = f"embed={'%.1f'%lo if lo==hi else '(%.1f,%.1f]'%(lo,hi)}"
            embed_bracket_analysis[key] = {
                "n": len(idxs),
                "hard_pass_rate": round(hp_rate, 4),
                "avg_struct_score": round(avg_struct, 4),
            }

    # --- Per-model correlations ---
    per_model_corr = {}
    for model, md in per_model_data.items():
        if len(md["struct_scores"]) >= 5:
            per_model_corr[model] = {
                "n": len(md["struct_scores"]),
                "pearson_struct_embed": round(pearson_corr(md["struct_scores"], md["embed_scores"]), 4),
                "spearman_struct_embed": round(spearman_corr(md["struct_scores"], md["embed_scores"]), 4),
                "hard_pass_rate": round(sum(md["hard_passes"]) / len(md["hard_passes"]), 4),
                "mean_struct": round(sum(md["struct_scores"]) / len(md["struct_scores"]), 4),
                "mean_embed": round(sum(md["embed_scores"]) / len(md["embed_scores"]), 4),
            }

    # --- Zero-inflated analysis: how many scores are exactly 0 or 1? ---
    struct_zeros_pct = sum(1 for s in struct_scores if s == 0.0) / len(struct_scores)
    struct_ones_pct = sum(1 for s in struct_scores if s == 1.0) / len(struct_scores)
    embed_zeros_pct = sum(1 for e in embed_scores if e == 0.0) / len(embed_scores)
    embed_ones_pct = sum(1 for e in embed_scores if e == 1.0) / len(embed_scores)

    result = {
        "metadata": {
            "files_processed": file_count,
            "files_skipped": skipped_files,
            "total_ut_cases": detail_count,
            "n_models": len(per_model_data),
        },
        "global_correlations": {
            "struct_vs_embed": {
                "pearson": round(pearson_se, 6),
                "spearman": round(spearman_se, 6),
                "interpretation": "Correlation between the two scoring components",
            },
            "struct_vs_hard_pass": {
                "point_biserial": round(pearson_s_hp, 6),
                "spearman": round(spearman_s_hp, 6),
            },
            "embed_vs_hard_pass": {
                "point_biserial": round(pearson_e_hp, 6),
                "spearman": round(spearman_e_hp, 6),
            },
            "final_score_vs_hard_pass": {
                "point_biserial": round(biserial_f_hp, 6),
                "spearman": round(spearman_f_hp, 6),
            },
        },
        "distribution_stats": {
            "struct_score": struct_dist,
            "embed_score": embed_dist,
            "final_score": final_dist,
            "hard_pass_rate": round(hard_pass_rate, 6),
        },
        "zero_inflation": {
            "struct_score_exact_0_pct": round(struct_zeros_pct, 4),
            "struct_score_exact_1_pct": round(struct_ones_pct, 4),
            "embed_score_exact_0_pct": round(embed_zeros_pct, 4),
            "embed_score_exact_1_pct": round(embed_ones_pct, 4),
        },
        "conditional_analysis": {
            "embed_score_when_struct_perfect": distribution_stats(embed_when_struct_perfect) if embed_when_struct_perfect else {},
            "embed_score_when_struct_zero": distribution_stats(embed_when_struct_zero) if embed_when_struct_zero else {},
        },
        "hard_pass_by_struct_bracket": bracket_analysis,
        "hard_pass_by_embed_bracket": embed_bracket_analysis,
        "per_model_correlations": per_model_corr,
    }

    out_path = base_dir / "data/ut_scoring_correlation.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("UT SCORING CORRELATION ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nDataset: {detail_count:,} unit test cases across {len(per_model_data)} models")
    print(f"Overall hard_pass rate: {hard_pass_rate:.1%}")

    print("\n--- Distribution of Scores ---")
    for name, dist in [("struct_score", struct_dist), ("embed_score", embed_dist), ("final_score", final_dist)]:
        print(f"  {name}: mean={dist['mean']:.3f}  std={dist['std']:.3f}  "
              f"Q1={dist['q1']:.3f}  med={dist['median']:.3f}  Q3={dist['q3']:.3f}")

    print("\n--- Zero/One Inflation ---")
    print(f"  struct_score exactly 0: {struct_zeros_pct:.1%}  |  exactly 1: {struct_ones_pct:.1%}")
    print(f"  embed_score  exactly 0: {embed_zeros_pct:.1%}  |  exactly 1: {embed_ones_pct:.1%}")

    print("\n--- Correlation: struct_score vs embed_score ---")
    print(f"  Pearson  r = {pearson_se:.4f}")
    print(f"  Spearman r = {spearman_se:.4f}")
    if abs(pearson_se) < 0.2:
        interp = "very weak (near-independent)"
    elif abs(pearson_se) < 0.4:
        interp = "weak"
    elif abs(pearson_se) < 0.6:
        interp = "moderate"
    else:
        interp = "strong"
    print(f"  => {interp} correlation")

    print("\n--- Correlation vs downstream hard_pass ---")
    print(f"  struct_score:  point-biserial={pearson_s_hp:.4f}  spearman={spearman_s_hp:.4f}")
    print(f"  embed_score:   point-biserial={pearson_e_hp:.4f}  spearman={spearman_e_hp:.4f}")
    print(f"  final_score:   point-biserial={biserial_f_hp:.4f}  spearman={spearman_f_hp:.4f}")

    print("\n--- hard_pass rate by struct_score bracket ---")
    for k, v in bracket_analysis.items():
        print(f"  {k}: n={v['n']:5d}  hard_pass={v['hard_pass_rate']:.1%}  avg_embed={v['avg_embed_score']:.3f}")

    print("\n--- hard_pass rate by embed_score bracket ---")
    for k, v in embed_bracket_analysis.items():
        print(f"  {k}: n={v['n']:5d}  hard_pass={v['hard_pass_rate']:.1%}  avg_struct={v['avg_struct_score']:.3f}")

    print("\n--- embed_score distribution conditioned on struct_score ---")
    if embed_when_struct_perfect:
        d = distribution_stats(embed_when_struct_perfect)
        print(f"  When struct=1.0 (n={d['n']}): embed mean={d['mean']:.3f}  std={d['std']:.3f}  med={d['median']:.3f}")
    if embed_when_struct_zero:
        d = distribution_stats(embed_when_struct_zero)
        print(f"  When struct=0.0 (n={d['n']}): embed mean={d['mean']:.3f}  std={d['std']:.3f}  med={d['median']:.3f}")

    print("\n--- Per-model Pearson(struct, embed) range ---")
    corrs = [v["pearson_struct_embed"] for v in per_model_corr.values() if not math.isnan(v["pearson_struct_embed"])]
    if corrs:
        print(f"  min={min(corrs):.4f}  max={max(corrs):.4f}  mean={sum(corrs)/len(corrs):.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
