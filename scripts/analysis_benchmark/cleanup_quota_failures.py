#!/usr/bin/env python3
"""
Clean up quota-failed entries from solver ablation and judge re-eval results.

After OpenAI API quota exhaustion, many servers got SR=0 not because
the tools were bad, but because the API call failed. This script removes
those entries so that resume will re-evaluate them.

Usage:
    python cleanup_quota_failures.py --dry-run   # preview what would be removed
    python cleanup_quota_failures.py              # actually remove
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SOLVER_ABLATION_ROOT = "temp/solver_ablation_gpt54_multi"
JUDGE_REEVAL_ROOT = "temp/judge_reeval_gpt54_full"


def is_quota_failure(server_result: Dict[str, Any]) -> bool:
    """Detect if a server result failed due to API quota exhaustion."""
    details = server_result.get("details", [])
    error = server_result.get("error", "")
    sr = server_result.get("sr_soft", 0)

    # Explicit error field
    if "quota" in error.lower() or "429" in error:
        return True

    # All tasks scored 0 and judge output contains error markers
    if sr == 0 and details:
        all_zero = all(d.get("score", 0) == 0 for d in details)
        if not all_zero:
            return False
        judge_texts = " ".join(str(d.get("judge_output", "")) for d in details)
        markers = ["429", "quota", "insufficient", "execution_failed", "rate_limit"]
        if any(m in judge_texts.lower() for m in markers):
            return True

    return False


def clean_per_server(path: Path, dry_run: bool) -> Tuple[int, int]:
    """Clean one per_server.json file. Returns (kept, removed)."""
    if not path.exists():
        return 0, 0

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kept = {}
    removed = []
    for slug, result in data.items():
        if is_quota_failure(result):
            removed.append(slug)
        else:
            kept[slug] = result

    if removed and not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)

    return len(kept), len(removed)


def main():
    parser = argparse.ArgumentParser(description="Clean up quota-failed entries.")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--solver-only", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    args = parser.parse_args()

    total_kept = 0
    total_removed = 0

    targets = []
    if not args.judge_only:
        if os.path.isdir(SOLVER_ABLATION_ROOT):
            for model in sorted(os.listdir(SOLVER_ABLATION_ROOT)):
                p = Path(SOLVER_ABLATION_ROOT) / model / "per_server.json"
                if p.exists():
                    targets.append(("solver", model, p))

    if not args.solver_only:
        if os.path.isdir(JUDGE_REEVAL_ROOT):
            for model in sorted(os.listdir(JUDGE_REEVAL_ROOT)):
                p = Path(JUDGE_REEVAL_ROOT) / model / "per_server.json"
                if p.exists():
                    targets.append(("judge", model, p))

    prefix = "[DRY RUN] " if args.dry_run else ""
    for kind, model, path in targets:
        kept, removed = clean_per_server(path, args.dry_run)
        total_kept += kept
        total_removed += removed
        if removed:
            print(f"{prefix}{kind}/{model}: kept {kept}, removed {removed} quota-failed entries")
        else:
            print(f"{prefix}{kind}/{model}: {kept} entries, 0 quota failures")

    print(f"\n{prefix}Total: kept {total_kept}, removed {total_removed}")
    if args.dry_run and total_removed > 0:
        print("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
