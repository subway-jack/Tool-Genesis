from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import importlib.util
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_agentic = _load_module(
    "env_evaluation_agentic_framework",
    Path(__file__).resolve().parents[2] / "src" / "env_evaluation" / "agentic_framework.py",
)
initialize_agent = _agentic.initialize_agent

logger = logging.getLogger(__name__)


def build_solve_task_prompt(task: str) -> str:
    return f"""
    {task}
    """


def build_judge_task_prompt(task: str, conversation_history: str) -> str:
    template_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "env_evaluation"
        / "prompt"
        / "response_quality_check.md"
    )
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("{QUESTION_CONTENT}", task).replace(
            "{CONVERSATION_HISTORY}", conversation_history
        )
    )


class JudgeResult:
    def __init__(self, score: int, analysis: str) -> None:
        self.score = score
        self.analysis = analysis

    def __str__(self) -> str:
        return f"Score: {self.score}, Analysis: {self.analysis}"


class ExecuteTask:
    def __init__(
        self,
        tasks: List[Any],
        server_name: str,
        toolkit: Any,
        schema_path: Optional[str] = None,
        agent_result_path: Optional[Union[str, Path]] = "temp/agentic",
        include_history: bool = False,
    ):
        self.tasks = tasks
        self.server_name = server_name
        self.simulation_toolkit = toolkit
        self.schema_path = schema_path
        base_dir = Path(agent_result_path) if agent_result_path is not None else Path("temp/agentic")
        self.agent_result_root = base_dir
        os.makedirs(self.agent_result_root, exist_ok=True)
        self.include_history = include_history
        self.agents = initialize_agent(
            self.server_name, self.simulation_toolkit, str(self.agent_result_root)
        )

    def _build_conversation_history(self, agent) -> str:
        try:
            openai_messages, _ = agent.memory.get_context_all()
        except Exception:
            return ""
        try:
            return json.dumps(openai_messages, ensure_ascii=False, indent=2)
        except Exception:
            return "\n".join(str(m) for m in openai_messages)

    def _parse_judge_response(self, xml_text: str) -> "JudgeResult":
        text = xml_text.strip()
        start_idx = text.find("<response")
        end_idx = text.rfind("</response>")
        if start_idx != -1 and end_idx != -1:
            end_idx += len("</response>")
            text = text[start_idx:end_idx]
        if not text:
            return JudgeResult(score=0, analysis="")
        try:
            root = ET.fromstring(text)
        except Exception:
            return JudgeResult(score=0, analysis=text)
        reasoning_elem = root.find(".//reasoning")
        rating_elem = root.find(".//rating")
        if reasoning_elem is not None:
            reasoning = "".join(reasoning_elem.itertext()).strip()
        else:
            reasoning = text
        rating_raw = ""
        if rating_elem is not None and rating_elem.text:
            rating_raw = rating_elem.text.strip()
        rating_lower = rating_raw.lower()
        score_map = {
            "very incomplete": 1,
            "incomplete": 2,
            "partially complete": 3,
            "mostly complete": 4,
            "fully complete": 5,
        }
        score = 0
        if rating_lower.isdigit():
            try:
                score = int(rating_lower)
            except Exception:
                score = 0
        else:
            score = score_map.get(rating_lower, 0)
        if score < 0 or score > 5:
            score = 0
        return JudgeResult(score=score, analysis=reasoning)

    def _judge_task_solved_with_debug(
        self, agents, task: str, conversation_history: str
    ) -> JudgeResult:
        last_judge: Optional[JudgeResult] = None
        for _ in range(3):
            prompt = build_judge_task_prompt(task, conversation_history)
            try:
                raw_response = agents["judge"].step(prompt)
                content = getattr(raw_response, "content", str(raw_response))
                task_judge = self._parse_judge_response(content)
                last_judge = task_judge
                # score > 0 means the judge returned a meaningful rating (1-5);
                # score == 0 means parse failure, so retry.
                if task_judge.score > 0:
                    return task_judge
            except Exception:
                continue
        if last_judge is not None:
            return last_judge
        return JudgeResult(score=0, analysis="Failed to obtain valid score from judge after 3 attempts.")

    def _extract_task_text(self, task: Any) -> str:
        if isinstance(task, str):
            return task
        if isinstance(task, dict):
            question = task.get("question")
            if isinstance(question, str) and question.strip():
                return question.strip()
            messages = task.get("messages")
            if isinstance(messages, list):
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content")
                        if isinstance(content, str) and content.strip():
                            return content.strip()
        return ""

    def _extract_messages(self, task: Any) -> List[Dict[str, Any]]:
        if isinstance(task, dict):
            messages = task.get("messages")
            if isinstance(messages, list):
                return [m for m in messages if isinstance(m, dict)]
        return []

    def _simulate_execute_task(self) -> Tuple[List[JudgeResult], List[Dict[str, Any]]]:
        if not self.tasks:
            return [], []

        num_tasks = len(self.tasks)
        tasks_judge: List[Optional[JudgeResult]] = [None] * num_tasks
        details: List[Optional[Dict[str, Any]]] = [None] * num_tasks

        def run_single_task(idx: int, task: Any):
            task_dir = self.agent_result_root / f"task_{idx+1}"
            os.makedirs(task_dir, exist_ok=True)

            try:
                agents = initialize_agent(
                    self.server_name, self.simulation_toolkit, str(task_dir)
                )

                task_text = self._extract_task_text(task)
                task_messages = self._extract_messages(task)
                prompt = build_solve_task_prompt(task_text)
                raw_response = agents["simulate_solve"].step(prompt)
                response = getattr(raw_response, "content", str(raw_response))
                conversation_history = self._build_conversation_history(agents["simulate_solve"])
                if (not conversation_history or conversation_history.strip() == "") and task_messages:
                    history = list(task_messages)
                    history.append({"role": "assistant", "content": response})
                    conversation_history = json.dumps(history, ensure_ascii=False, indent=2)
                task_judge = self._judge_task_solved_with_debug(
                    agents, task_text, conversation_history
                )
                detail = {
                    "task": task_text,
                    "response": response,
                    "judge_output": task_judge.analysis,
                    "solved": task_judge.score >= 3,
                }
                if self.include_history:
                    detail["conversation_history"] = conversation_history
                return idx, task_judge, detail
            except Exception as exc:
                logger.exception("Task execution failed for task_%s", idx + 1)
                task_judge = JudgeResult(score=0, analysis=f"execution_failed: {exc}")
                detail = {
                    "task": task_text if "task_text" in locals() else "",
                    "response": "",
                    "judge_output": task_judge.analysis,
                    "solved": False,
                }
                if self.include_history:
                    detail["conversation_history"] = ""
                return idx, task_judge, detail

        max_workers = 8
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            future_to_idx: Dict[Any, int] = {}
            for idx, task in enumerate(self.tasks):
                future = executor.submit(run_single_task, idx, task)
                futures.append(future)
                future_to_idx[future] = idx
            for future in as_completed(futures):
                try:
                    idx, task_judge, detail = future.result()
                    tasks_judge[idx] = task_judge
                    details[idx] = detail
                except Exception as exc:
                    logger.exception("Task future failed")
                    idx = future_to_idx.get(future, 0)
                    task_judge = JudgeResult(score=0, analysis=f"future_failed: {exc}")
                    detail = {
                        "task": self.tasks[idx] if 0 <= idx < len(self.tasks) else "",
                        "response": "",
                        "judge_output": task_judge.analysis,
                        "solved": False,
                    }
                    if self.include_history:
                        detail["conversation_history"] = ""
                    tasks_judge[idx] = task_judge
                    details[idx] = detail

        return [t for t in tasks_judge if t is not None], [
            d for d in details if d is not None
        ]

    def get_execute_task_result(self) -> Tuple[float, Dict[str, Any]]:
        if not self.tasks:
            return 0.0, {"details": []}
        simulation_tasks_judge, details = self._simulate_execute_task()
        number_of_tasks = len(self.tasks)
        if number_of_tasks == 0:
            return 0.0, {"details": details}
        simulation_tasks_judge_score = (
            sum(int(t.score) for t in simulation_tasks_judge) / number_of_tasks
        )
        # Normalize: map average score (0-5) to [0, 1]
        score = max(0.0, min(1.0, simulation_tasks_judge_score / 5.0))

        return score, {"details": details}


def _load_registry_items(pred_path: str) -> List[Dict[str, Any]]:
    p = Path(pred_path)
    reg_path = p / "registry.json" if p.is_dir() else p
    if not reg_path.exists():
        return []
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            items: List[Dict[str, Any]] = []
            for slug, payload in data.items():
                if not isinstance(payload, dict):
                    continue
                schema_path = payload.get("json_schema_path")
                code_path = payload.get("env_code_path")
                server_id = payload.get("server_id")
                server_name = payload.get("server_name")
                if isinstance(schema_path, str) or isinstance(code_path, str):
                    items.append(
                        {
                            "server_id": server_id,
                            "server_slug": str(slug),
                            "server_name": server_name,
                            "schema_path": schema_path if isinstance(schema_path, str) else None,
                            "env_code_path": code_path if isinstance(code_path, str) else None,
                        }
                    )
            return items
    except Exception:
        return []
    return []


class _EmptyToolkit:
    def get_tools(self) -> List[Any]:
        return []

def _normalize_key(value: str) -> str:
    return str(value or "").strip().replace(" ", "_").replace("-", "_").lower()


def _pick_server_name(item: Dict[str, Any]) -> str:
    for key in ("server_name", "server_slug"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "unknown"


def _candidate_task_keys(item: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for key in ("server_slug", "server_name"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            norm = _normalize_key(val)
            if norm and norm not in keys:
                keys.append(norm)
    return keys


def _load_tasks_by_server(tasks_root: str) -> Dict[str, List[Dict[str, Any]]]:
    root = Path(tasks_root)
    out: Dict[str, List[Dict[str, Any]]] = {}
    all_tasks_path = root / "all_tasks.json"
    if all_tasks_path.exists():
        try:
            data = json.loads(all_tasks_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for rec in data:
                    if not isinstance(rec, dict):
                        continue
                    slug = rec.get("server_slug")
                    tasks = rec.get("tasks")
                    if isinstance(slug, str) and isinstance(tasks, list):
                        out[_normalize_key(slug)] = [t for t in tasks if isinstance(t, dict)]
        except Exception:
            pass
    if out:
        return out
    for sample in root.glob("*/sample_prepared.json"):
        try:
            data = json.loads(sample.read_text(encoding="utf-8"))
            if isinstance(data, list):
                out[_normalize_key(sample.parent.name)] = [
                    t for t in data if isinstance(t, dict)
                ]
        except Exception:
            continue
    return out


def _select_tasks(tasks: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return tasks
    return tasks[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", type=str, required=True)
    parser.add_argument("--final-task-root", type=str, default="data/final_task")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--trajectory-root", type=str, default="temp/agentic/no_tools")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tasks-limit", type=int, default=20)
    parser.add_argument("--registry-path", type=str, default=None)
    parser.add_argument("--use-tools", action="store_true")
    parser.add_argument("--include-history", action="store_true", default=True)
    args = parser.parse_args()

    items = _load_registry_items(args.pred_path)
    if args.limit is not None and args.limit > 0:
        items = items[: args.limit]

    tasks_by_server = _load_tasks_by_server(args.final_task_root)
    if tasks_by_server:
        items = [
            item
            for item in items
            if any(key in tasks_by_server for key in _candidate_task_keys(item))
        ]

    results: List[Dict[str, Any]] = []
    base_root = Path(args.trajectory_root)
    for item in items:
        server_name = _pick_server_name(item)
        server_slug = item.get("server_slug") or server_name
        raw_tasks: List[Dict[str, Any]] = []
        for key in _candidate_task_keys({"server_slug": server_slug, "server_name": server_name}):
            if key in tasks_by_server:
                raw_tasks = tasks_by_server.get(key, [])
                break
        tasks = _select_tasks(raw_tasks, args.tasks_limit)
        toolkit: Any
        if args.use_tools:
            try:
                from src.core.toolkits.mcp_sandbox_bridge_toolkit import (
                    MCPServerToolsToolkit,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to import MCPServerToolsToolkit: {exc}"
                ) from exc
            toolkit = MCPServerToolsToolkit(
                server_names=server_name,
                registry_path=args.registry_path,
            )
            toolkit.initialize_servers()
        else:
            toolkit = _EmptyToolkit()
        task_root = base_root / server_name
        exec_task = ExecuteTask(
            tasks=tasks,
            server_name=server_name,
            toolkit=toolkit,
            schema_path=item.get("schema_path"),
            agent_result_path=task_root,
            include_history=True if args.include_history else False,
        )
        score, detail = exec_task.get_execute_task_result()
        result = {
            "server": server_name,
            "score_no_tools": float(score),
            "score_no_tools_div4": float(score) / 4.0,
            "task_count": len(tasks),
        }
        if args.include_history:
            result["details"] = detail.get("details", [])
        results.append(result)
        if args.use_tools and hasattr(toolkit, "cleanup"):
            toolkit.cleanup()

    payload = {"count": len(results), "results": results}
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
