# src/langgraph_agents_system_running_logs.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List
from datetime import datetime
import uuid
import json
import asyncio
import aiofiles
from datetime import datetime
from langchain.messages import HumanMessage, AIMessage


def rebuild_input_messages(query: str | list):
    if isinstance(query, list):
        lc_messages = []
        for msg in query:
            if msg.get("role") == "user":
                lc_messages.append(HumanMessage(
                    content=msg.get("content")))
            elif msg.get("role") == "assistant":
                lc_messages.append(AIMessage(content=msg.get("content")))
        messages = lc_messages
    else:
        messages = HumanMessage(content=query)
    return messages


def get_final_answer(response: dict) -> str:
    messages = response.get("messages", [])
    for msg in reversed(messages):
        if (msg.__class__.__name__ == "AIMessage"
                and not getattr(msg, "tool_calls", [])):
            return msg.content
    return ""


def extract_langgraph_log(
    response: Dict[str, Any],
    user_id: str = "unknown",
) -> Dict[str, Any]:

    trace_id = str(uuid.uuid4())

    parsed_log = {
        "trace_id": trace_id,
        "timestamp": datetime.now().isoformat(timespec="microseconds"),
        "user_id": user_id,
        "user_query": "",
        "steps": [],
    }

    for agent_name, agent_data in response.items():

        messages: List[Any] = agent_data.get("messages", [])

        if not messages:
            continue

        for msg in messages:
            if msg.__class__.__name__ == "HumanMessage":
                if not parsed_log["user_query"]:
                    content = msg.content

                    if isinstance(content, list):
                        # 多模态内容：只提取 type == "text" 的项
                        texts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                texts.append(item.get("text", ""))
                        # 忽略所有 image_url 等其他类型
                        parsed_log["user_query"] = "\n".join(texts).strip()
                    else:
                        # 纯文本字符串直接使用
                        parsed_log["user_query"] = content

        ai_messages = [
            m for m in messages
            if m.__class__.__name__ == "AIMessage"
        ]

        if not ai_messages:
            continue

        final_msg = None

        for m in reversed(ai_messages):
            if getattr(m, "content", None) not in [None, ""]:
                final_msg = m
                break

        if final_msg is None:
            continue

        usage = getattr(final_msg, "usage_metadata", {}) or {}
        response_metadata = getattr(final_msg, "response_metadata", {}) or {}

        tool_calls_raw = []

        for m in ai_messages:
            tc_list = getattr(m, "tool_calls", []) or []
            if tc_list:
                tool_calls_raw.extend(tc_list)

        tool_calls = [
            {
                "name": tc.get("name"),
                "arguments": tc.get("args", {}),
                "call_id": tc.get("id"),
            }
            for tc in tool_calls_raw
        ]

        reasoning_tokens = (
            usage.get("output_token_details", {})
            .get("reasoning", 0)
        )

        step_data = {
            "response_id": getattr(final_msg, "id", "__fake_id__"),

            "model": response_metadata.get("model_name", "unknown"),

            "agent": getattr(final_msg, "name", None) or agent_name,

            "messages": [
                {
                    "role": "assistant",
                    "content": final_msg.content,
                    "type": "final_answer",
                }
            ],

            "tool_calls": tool_calls,

            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "reasoning_tokens": reasoning_tokens,
            },

            "step": len(parsed_log["steps"]) + 1,
        }

        parsed_log["steps"].append(step_data)

    return parsed_log


_log_write_lock = asyncio.Lock()


async def save_log(user_id, response, saving_path):

    parsed = extract_langgraph_log(
        response=response,
        user_id=user_id,
    )

    async with _log_write_lock:
        async with aiofiles.open(saving_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
