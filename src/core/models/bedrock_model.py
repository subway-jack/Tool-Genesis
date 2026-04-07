"""AWS Bedrock Converse API backend via bearer-token proxy.

Uses the same proxy endpoint as src/utils/llm._call_bedrock but wrapped
in the BaseModelBackend interface so that ModelFactory / ChatAgent can
use Claude models hosted on Bedrock.
"""
from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

import httpx
from pydantic import BaseModel

from src.core.messages import OpenAIMessage
from src.core.models.base import BaseModelBackend, SimpleHeuristicTokenCounter
from src.core.types import ModelType, UnifiedModelType


# ---------------------------------------------------------------------------
# Lightweight response wrappers that mimic the OpenAI SDK objects just enough
# for ChatAgent._handle_batch_response to work.
# ---------------------------------------------------------------------------

@dataclass
class _Function:
    name: str
    arguments: str  # JSON string


@dataclass
class _ToolCall:
    id: str
    type: str
    function: _Function


@dataclass
class _Message:
    role: str
    content: Optional[str]
    tool_calls: Optional[List[_ToolCall]] = None
    parsed: Optional[Any] = None


@dataclass
class _Choice:
    index: int
    message: _Message
    finish_reason: str
    logprobs: Optional[Any] = None


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def model_dump(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class _ChatCompletion:
    id: str = ""
    choices: List[_Choice] = field(default_factory=list)
    usage: Optional[_Usage] = None


# ---------------------------------------------------------------------------
# Format conversion helpers
# ---------------------------------------------------------------------------

def _openai_messages_to_bedrock(messages: List[OpenAIMessage]):
    """Convert OpenAI-style messages to Bedrock Converse format.

    Returns (system_prompt_list, bedrock_messages).
    """
    system_parts: List[Dict] = []
    bedrock_msgs: List[Dict] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # --- system / developer → Bedrock system block ---
        if role in ("system", "developer"):
            if isinstance(content, str) and content.strip():
                system_parts.append({"text": content})
            continue

        # --- tool result → Bedrock user message with toolResult ---
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            text = content if isinstance(content, str) else json.dumps(content)
            bedrock_msgs.append({
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_call_id,
                            "content": [{"text": text}],
                        }
                    }
                ],
            })
            continue

        # --- assistant with tool_calls ---
        if role == "assistant" and msg.get("tool_calls"):
            parts: List[Dict] = []
            if isinstance(content, str) and content.strip():
                parts.append({"text": content})
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                parts.append({
                    "toolUse": {
                        "toolUseId": tc.get("id", str(uuid.uuid4())),
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                })
            bedrock_msgs.append({"role": "assistant", "content": parts})
            continue

        # --- normal user / assistant text ---
        if isinstance(content, str):
            bedrock_msgs.append({
                "role": role if role in ("user", "assistant") else "user",
                "content": [{"text": content or " "}],
            })
        elif isinstance(content, list):
            # Multi-part content (e.g. images) — pass text parts only
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append({"text": block.get("text", "")})
            if not parts:
                parts = [{"text": " "}]
            bedrock_msgs.append({
                "role": role if role in ("user", "assistant") else "user",
                "content": parts,
            })

    # Bedrock requires alternating user/assistant; merge consecutive same-role
    merged: List[Dict] = []
    for m in bedrock_msgs:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"].extend(m["content"])
        else:
            merged.append(m)

    # Bedrock requires first message to be user
    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": [{"text": " "}]})

    return system_parts, merged


def _openai_tools_to_bedrock(tools: List[Dict]) -> Dict:
    """Convert OpenAI tool schemas to Bedrock toolConfig."""
    specs = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool["function"]
        spec: Dict[str, Any] = {
            "name": fn["name"],
            "description": fn.get("description", ""),
        }
        params = fn.get("parameters")
        if params:
            spec["inputSchema"] = {"json": params}
        specs.append({"toolSpec": spec})
    if not specs:
        return {}
    return {"tools": specs}


def _bedrock_response_to_openai(data: Dict) -> _ChatCompletion:
    """Convert Bedrock Converse response to OpenAI-like ChatCompletion."""
    output = data.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])
    stop_reason = data.get("stopReason", "end_turn")

    text_parts = []
    tool_calls = []

    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(_ToolCall(
                id=tu.get("toolUseId", str(uuid.uuid4())),
                type="function",
                function=_Function(
                    name=tu.get("name", ""),
                    arguments=json.dumps(tu.get("input", {})),
                ),
            ))

    finish = "tool_calls" if tool_calls else "stop"
    if stop_reason == "max_tokens":
        finish = "length"

    usage_data = data.get("usage", {})

    return _ChatCompletion(
        id=f"bedrock-{uuid.uuid4().hex[:12]}",
        choices=[
            _Choice(
                index=0,
                message=_Message(
                    role="assistant",
                    content="\n".join(text_parts) if text_parts else None,
                    tool_calls=tool_calls if tool_calls else None,
                ),
                finish_reason=finish,
            )
        ],
        usage=_Usage(
            prompt_tokens=usage_data.get("inputTokens", 0),
            completion_tokens=usage_data.get("outputTokens", 0),
            total_tokens=usage_data.get("inputTokens", 0)
            + usage_data.get("outputTokens", 0),
        ),
    )


# ---------------------------------------------------------------------------
# Model backend
# ---------------------------------------------------------------------------

class BedrockModel(BaseModelBackend):
    """Bedrock Converse API backend using bearer-token proxy."""

    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter=None,
        timeout: Optional[float] = None,
    ) -> None:
        if model_config_dict is None:
            model_config_dict = {}
        api_key = api_key or os.environ.get("BEDROCK_API_KEY", "")
        url = url or os.environ.get("BEDROCK_BASE_URL",
                                     "https://bedrock-runtime.us-west-2.amazonaws.com")
        timeout = timeout or float(os.environ.get("MODEL_TIMEOUT", 300))
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter, timeout
        )
        self._proxy = os.environ.get("BEDROCK_PROXY", "")
        self._max_retries = 5

    @property
    def token_counter(self):
        if not self._token_counter:
            self._token_counter = SimpleHeuristicTokenCounter()
        return self._token_counter

    # ---- internal plumbing ------------------------------------------------

    def _post(self, model_id: str, payload: Dict) -> Dict:
        """POST to Bedrock Converse endpoint with retries."""
        url = f"{self._url}/model/{model_id}/converse"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        client_kwargs: Dict[str, Any] = {"timeout": self._timeout}
        if self._proxy:
            client_kwargs["proxy"] = self._proxy

        for attempt in range(self._max_retries):
            try:
                with httpx.Client(**client_kwargs) as client:
                    resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 529):
                    wait = min(2 ** attempt + random.random(), 30)
                    time.sleep(wait)
                    continue
                if resp.status_code == 503:
                    wait = min(2 ** attempt + random.random(), 15)
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Bedrock API Error {resp.status_code}: {resp.text[:500]}"
                )
            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

        raise RuntimeError("Bedrock: max retries exceeded")

    # ---- BaseModelBackend interface ---------------------------------------

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        model_id = str(self.model_type)
        system_parts, bedrock_messages = _openai_messages_to_bedrock(messages)

        payload: Dict[str, Any] = {"messages": bedrock_messages}
        if system_parts:
            payload["system"] = system_parts

        # Inference config
        inf_config: Dict[str, Any] = {}
        max_tokens = self.model_config_dict.get("max_tokens", 8192)
        if max_tokens:
            inf_config["maxTokens"] = max_tokens
        temperature = self.model_config_dict.get("temperature")
        if temperature is not None:
            inf_config["temperature"] = temperature
        if inf_config:
            payload["inferenceConfig"] = inf_config

        # Tool config — Bedrock requires toolConfig whenever the conversation
        # history contains toolUse/toolResult blocks, even if the current turn
        # doesn't request tool use.
        if tools:
            tool_config = _openai_tools_to_bedrock(tools)
            if tool_config:
                payload["toolConfig"] = tool_config
                self._last_tool_config = tool_config
        elif not tools and self._history_has_tool_blocks(bedrock_messages):
            # Reuse the last known tool config so Bedrock doesn't reject
            if hasattr(self, "_last_tool_config") and self._last_tool_config:
                payload["toolConfig"] = self._last_tool_config
            else:
                # Build a minimal toolConfig from toolUse blocks in history
                minimal = self._extract_tool_config_from_history(bedrock_messages)
                if minimal:
                    payload["toolConfig"] = minimal

        data = self._post(model_id, payload)
        return _bedrock_response_to_openai(data)

    @staticmethod
    def _history_has_tool_blocks(bedrock_messages: List[Dict]) -> bool:
        for msg in bedrock_messages:
            for block in msg.get("content", []):
                if "toolUse" in block or "toolResult" in block:
                    return True
        return False

    @staticmethod
    def _extract_tool_config_from_history(bedrock_messages: List[Dict]) -> Dict:
        """Build minimal toolConfig from toolUse blocks found in history."""
        seen = set()
        specs = []
        for msg in bedrock_messages:
            for block in msg.get("content", []):
                tu = block.get("toolUse")
                if tu and tu.get("name") not in seen:
                    seen.add(tu["name"])
                    specs.append({
                        "toolSpec": {
                            "name": tu["name"],
                            "description": f"Tool {tu['name']}",
                            "inputSchema": {"json": {"type": "object"}},
                        }
                    })
        return {"tools": specs} if specs else {}

    def _sanitize_config(self, config_dict):
        return config_dict

    def _adapt_messages_for_o1_models(self, messages):
        return messages

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        import asyncio
        return await asyncio.to_thread(self._run, messages, response_format, tools)

    def check_model_config(self):
        pass
