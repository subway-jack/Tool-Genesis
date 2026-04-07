import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _dump_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _is_results_file(path: Path) -> bool:
    return path.name in ("result.json", "results.json")


def _as_list(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list):
        return [it for it in x if isinstance(it, dict)]
    return []


def _extract_details(l2_debug: Any) -> List[Dict[str, Any]]:
    if not isinstance(l2_debug, dict):
        return []
    if isinstance(l2_debug.get("details"), list):
        return [it for it in l2_debug.get("details") if isinstance(it, dict)]
    unit_tests = l2_debug.get("unit_tests")
    if isinstance(unit_tests, dict) and isinstance(unit_tests.get("details"), list):
        return [it for it in unit_tests.get("details") if isinstance(it, dict)]
    return []


def _recompute_final_scores(details: List[Dict[str, Any]]) -> None:
    for d in details:
        s = d.get("struct_score")
        e = d.get("embed_score")
        try:
            s_val = float(s) if isinstance(s, (int, float)) else 0.0
        except Exception:
            s_val = 0.0
        try:
            e_val = float(e) if isinstance(e, (int, float)) else 0.0
        except Exception:
            e_val = 0.0
        d["final_score"] = (s_val + e_val) / 2.0


def _compute_soft_avg(details: List[Dict[str, Any]]) -> float:
    total = 0
    acc = 0.0
    for d in details:
        fs = d.get("final_score")
        if isinstance(fs, (int, float)):
            total += 1
            acc += float(fs)
    if total == 0:
        return 0.0
    return acc / total


def _compute_hard_rate(details: List[Dict[str, Any]], threshold: float) -> float:
    total = 0
    passed = 0
    for d in details:
        fs = d.get("final_score")
        if isinstance(fs, (int, float)):
            total += 1
            if fs > threshold:
                passed += 1
    if total == 0:
        return 0.0
    return passed / total


def _server_key(row: Dict[str, Any]) -> str:
    slug = row.get("server_slug")
    if isinstance(slug, str) and slug.strip():
        return slug.strip()
    sid = row.get("server_id")
    return str(sid or "").strip()


def recalibrate_project(project_dir: Path, threshold: float) -> Tuple[int, int]:
    results_path = project_dir / "results.json"
    if not results_path.exists():
        results_path = project_dir / "result.json"
    if not results_path.exists():
        return 0, 0
    data = _load_json(results_path)
    rows = _as_list(data)
    if not rows:
        return 0, 0
    updated = 0
    total = 0
    for row in rows:
        total += 1
        slug = _server_key(row)
        debug_path = project_dir / "debug" / slug / "l2_debug.json"
        l2_debug = _load_json(debug_path) if debug_path.exists() else None
        details = _extract_details(l2_debug)
        _recompute_final_scores(details)
        soft_avg = _compute_soft_avg(details)
        hard_rate = _compute_hard_rate(details, threshold)
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
            row["metrics"] = metrics
        metrics["tool_call_success_rate_soft"] = soft_avg
        metrics["tool_call_success_rate_hard"] = hard_rate
        if isinstance(l2_debug, dict):
            if isinstance(l2_debug.get("unit_tests"), dict):
                l2_debug["unit_tests"]["soft_avg"] = soft_avg
                l2_debug["unit_tests"]["hard_rate"] = hard_rate
            elif isinstance(l2_debug.get("details"), list):
                # If unit_tests is absent, write at top-level for backward compatibility
                l2_debug["soft_avg"] = soft_avg
                l2_debug["hard_rate"] = hard_rate
            _dump_json(debug_path, l2_debug)
        updated += 1
    _dump_json(results_path, rows)
    return updated, total


def iter_projects(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="temp/eval_results_v3")
    p.add_argument("--threshold", type=float, default=0.7)
    args = p.parse_args()
    root = Path(args.root)
    threshold = float(args.threshold)
    projects = iter_projects(root)
    if not projects:
        print(f"Not a directory: {root}")
        return
    total_projects = 0
    total_servers = 0
    updated_servers = 0
    for project in projects:
        if not any(_is_results_file(p) for p in project.iterdir()):
            continue
        total_projects += 1
        updated, total = recalibrate_project(project, threshold)
        total_servers += total
        updated_servers += updated
    print(json.dumps({
        "root": str(root),
        "projects": total_projects,
        "servers": total_servers,
        "updated": updated_servers
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
