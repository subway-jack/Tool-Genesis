"""
LLM Judge vs "Human" (GPT-5.4) Agreement Analysis.

Samples trajectory evaluation results, re-judges them with GPT-5.4 as a
stronger independent evaluator (simulated human), and computes agreement
metrics with the original Qwen3-14B judge.

Usage:
  python scripts/experiment_journal/judge_human_agreement.py

Output:
  data/judge_human_agreement.json
"""

import json
import os
import sys
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_samples(results_dir: str, n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """Sample trajectory evaluation details from existing l2_debug.json files."""
    random.seed(seed)
    all_samples = []

    for run_dir in os.listdir(results_dir):
        debug_dir = os.path.join(results_dir, run_dir, "debug")
        if not os.path.isdir(debug_dir):
            continue
        for server in os.listdir(debug_dir):
            l2_path = os.path.join(debug_dir, server, "l2_debug.json")
            if not os.path.exists(l2_path):
                continue
            with open(l2_path) as f:
                l2 = json.load(f)
            traj = l2.get("trajectory", {})
            details = traj.get("details", {})
            if isinstance(details, dict):
                detail_list = details.get("details", [])
            elif isinstance(details, list):
                detail_list = details
            else:
                continue
            for det in detail_list:
                task = det.get("task", "")
                response = det.get("response", "")
                judge_output = det.get("judge_output", "")
                solved = det.get("solved", False)
                if task and response and judge_output:
                    all_samples.append({
                        "model": run_dir,
                        "server": server,
                        "task": task,
                        "response": response[:2000],  # truncate for API
                        "original_judge_output": judge_output[:1000],
                        "original_solved": solved,
                    })

    random.shuffle(all_samples)
    samples = all_samples[:n_samples]
    print(f"Loaded {len(all_samples)} total samples, selected {len(samples)}")
    return samples


def judge_with_gpt54(task: str, response: str) -> Dict[str, Any]:
    """Use GPT-5.4 to independently judge a task completion."""
    from src.utils.llm import call_llm

    prompt = f"""You are evaluating whether an AI assistant successfully completed a user's task using tool-generated responses.

## Task
{task}

## Assistant's Response
{response}

## Evaluation Instructions
Rate the task completion on a scale of 1-5:
1 = Very Incomplete: The response fails to address the core request
2 = Incomplete: Major parts of the request are missing
3 = Partially Complete: Some key elements addressed but significant gaps remain
4 = Mostly Complete: Most elements addressed with minor gaps
5 = Fully Complete: All aspects of the task are thoroughly addressed

Respond in this exact format:
<response>
<reasoning>Your brief analysis here</reasoning>
<rating>NUMBER</rating>
</response>"""

    try:
        result = call_llm(
            text=prompt,
            system_prompt="You are a strict but fair evaluator. Judge based solely on whether the task was actually completed.",
            model="gpt-5.4",
            max_tokens=500,
            temperature=0,
            platform="openai",
        )
        return {"raw": result, "error": None}
    except Exception as e:
        return {"raw": None, "error": str(e)}


def parse_rating(raw_text: str) -> Optional[int]:
    """Extract rating from GPT-5.4 response."""
    import re
    if not raw_text:
        return None
    # Try XML format
    match = re.search(r"<rating>\s*(\d)\s*</rating>", raw_text)
    if match:
        return int(match.group(1))
    # Try plain number
    match = re.search(r"[Rr]ating[:\s]+(\d)", raw_text)
    if match:
        return int(match.group(1))
    # Last resort
    match = re.search(r"(\d)/5", raw_text)
    if match:
        return int(match.group(1))
    return None


def compute_agreement(samples: List[Dict]) -> Dict:
    """Compute agreement metrics between original judge and GPT-5.4."""
    # Binary agreement (solved or not, threshold=3)
    original_binary = []
    gpt_binary = []
    original_scores = []
    gpt_scores = []

    valid = 0
    for s in samples:
        gpt_score = s.get("gpt_score")
        if gpt_score is None:
            continue
        valid += 1
        orig_solved = s["original_solved"]
        gpt_solved = gpt_score >= 3

        original_binary.append(1 if orig_solved else 0)
        gpt_binary.append(1 if gpt_solved else 0)
        gpt_scores.append(gpt_score)

    if valid < 10:
        return {"error": f"Only {valid} valid samples"}

    # Accuracy
    agree = sum(a == b for a, b in zip(original_binary, gpt_binary))
    accuracy = agree / valid

    # Cohen's Kappa
    p_o = accuracy
    p_yes_orig = sum(original_binary) / valid
    p_yes_gpt = sum(gpt_binary) / valid
    p_e = p_yes_orig * p_yes_gpt + (1 - p_yes_orig) * (1 - p_yes_gpt)
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0

    # Confusion matrix
    tp = sum(a == 1 and b == 1 for a, b in zip(original_binary, gpt_binary))
    tn = sum(a == 0 and b == 0 for a, b in zip(original_binary, gpt_binary))
    fp = sum(a == 0 and b == 1 for a, b in zip(original_binary, gpt_binary))
    fn = sum(a == 1 and b == 0 for a, b in zip(original_binary, gpt_binary))

    return {
        "n_valid": valid,
        "accuracy": round(accuracy, 4),
        "cohens_kappa": round(kappa, 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "original_positive_rate": round(p_yes_orig, 4),
        "gpt_positive_rate": round(p_yes_gpt, 4),
        "gpt_score_distribution": {
            str(i): sum(1 for s in gpt_scores if s == i)
            for i in range(1, 6)
        },
    }


def main():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    results_dir = str(PROJECT_ROOT / "temp" / "eval_results_v3")

    print("=== LLM Judge vs GPT-5.4 Agreement ===")
    print()

    # Step 1: Load samples
    samples = load_samples(results_dir, n_samples=100)

    # Step 2: Re-judge with GPT-5.4
    print(f"\nJudging {len(samples)} samples with GPT-5.4...")
    for i, s in enumerate(samples):
        result = judge_with_gpt54(s["task"], s["response"])
        s["gpt_raw"] = result["raw"]
        s["gpt_error"] = result["error"]
        s["gpt_score"] = parse_rating(result["raw"]) if result["raw"] else None
        s["gpt_solved"] = s["gpt_score"] >= 3 if s["gpt_score"] else None

        if (i + 1) % 10 == 0:
            done = sum(1 for x in samples[:i+1] if x.get("gpt_score") is not None)
            print(f"  [{i+1}/{len(samples)}] {done} successfully judged")

        time.sleep(0.2)  # rate limit

    # Step 3: Compute agreement
    print("\nComputing agreement metrics...")
    agreement = compute_agreement(samples)

    print(f"\n=== Results ===")
    print(f"Valid samples: {agreement.get('n_valid', 0)}")
    print(f"Accuracy: {agreement.get('accuracy', 0):.1%}")
    print(f"Cohen's Kappa: {agreement.get('cohens_kappa', 0):.3f}")
    conf = agreement.get("confusion", {})
    print(f"Confusion: TP={conf.get('tp')}, TN={conf.get('tn')}, FP={conf.get('fp')}, FN={conf.get('fn')}")
    print(f"Original positive rate: {agreement.get('original_positive_rate', 0):.1%}")
    print(f"GPT-5.4 positive rate: {agreement.get('gpt_positive_rate', 0):.1%}")
    print(f"GPT-5.4 score distribution: {agreement.get('gpt_score_distribution', {})}")

    # Save
    output = {
        "method": "GPT-5.4 as simulated human evaluator",
        "original_judge": "Qwen3-14B",
        "n_samples": len(samples),
        **agreement,
        "samples": [
            {
                "model": s["model"],
                "server": s["server"],
                "task": s["task"][:200],
                "original_solved": s["original_solved"],
                "gpt_score": s.get("gpt_score"),
                "gpt_solved": s.get("gpt_solved"),
            }
            for s in samples
        ],
    }
    out_path = PROJECT_ROOT / "data" / "judge_human_agreement.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
