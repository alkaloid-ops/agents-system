# src/openai_agents_system_running_log.py
# -*- coding: utf-8 -*-

import json
import uuid
import asyncio
import aiofiles
from datetime import datetime


def parse_single_response(model_response, last_agent):
    result = {
        "response_id": getattr(model_response, "response_id", None),
        "model": None,
        "messages": [],
        "reasoning": [],
        "tool_calls": [],
        "usage": {},
        "agent": last_agent if last_agent else None,
    }

    # ===== usage =====
    if hasattr(model_response, "usage") and model_response.usage:
        usage = model_response.usage
        result["usage"] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "reasoning_tokens": getattr(
                usage.output_tokens_details, "reasoning_tokens", 0
            )
        }

    # ===== output parsing =====
    for item in model_response.output:

        if item.type == "reasoning":
            summaries = getattr(item, "summary", [])
            for s in summaries:
                result["reasoning"].append({
                    "text": s.text,
                    "type": s.type
                })

        elif item.type == "message":
            result["model"] = item.provider_data.get("model")

            for c in item.content:
                if c.type == "output_text":
                    result["messages"].append({
                        "role": item.role,
                        "content": c.text,
                        "type": "model_answer"
                    })

        elif item.type == "function_call":
            try:
                arguments = json.loads(item.arguments)
            except Exception:
                arguments = item.arguments

            result["tool_calls"].append({
                "name": item.name,
                "arguments": arguments,
                "call_id": item.call_id
            })

    return result


def parse_model_response(user_id, user_query, raw_responses, last_agent):

    if not isinstance(raw_responses, list):
        raw_responses = [raw_responses]

    trace = {
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "user_query": user_query,
        "steps": [],
    }

    for i, resp in enumerate(raw_responses):
        step_data = parse_single_response(resp, last_agent)
        step_data["step"] = i + 1
        trace["steps"].append(step_data)

    return trace


_log_write_lock = asyncio.Lock()


async def save_log(user_id, user_query, response, saving_path):

    query = ""
    if isinstance(user_query, list):
        for item in user_query:
            if item["type"] == "input_text":
                query = item["text"]
    else:
        query = user_query

    parsed = parse_model_response(
        user_id=user_id,
        user_query=query,
        raw_responses=response.raw_responses,
        last_agent=response.last_agent.name,
    )

    async with _log_write_lock:
        async with aiofiles.open(saving_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
