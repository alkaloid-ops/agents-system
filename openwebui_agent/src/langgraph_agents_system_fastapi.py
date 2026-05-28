# src/langgraph_agents_system.py
# -*- coding: utf-8 -*-

import uuid
import json
import asyncio
import uvicorn
import requests
from pydantic import BaseModel
from typing import List, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.middleware.cors import CORSMiddleware

from langgraph_agents_system import MultiAgentsSystem

app = FastAPI(title="Agents_System", description="FastAPI接口", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MAS = MultiAgentsSystem()

# # 前端请求捕获
# @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"])
# async def catch_all(path: str, request: Request):
#     print(f"\n{'='*50}")
#     print(f"请求方法: {request.method}")
#     print(f"请求路径: /{path}")
#     print(f"完整URL: {request.url}")
#     print(f"请求头: {dict(request.headers)}")

#     # 尝试读取请求体（如果是POST）
#     try:
#         body = await request.json()
#         print(f"请求体: {body}")
#     except:
#         pass

#     print(f"{'='*50}\n")

#     # 返回一个友好提示
#     return JSONResponse(
#         status_code=200,
#         content={
#             "message": f"收到 {request.method} 请求到 /{path}",
#             "note": "这是一个模拟响应，你需要实现这个端点"
#         }
#     )


@app.get("/models")
def model():
    return {"data": [{"id": str(uuid.uuid4()), "name": "LangGraph_Agents_System"}]}

# 查看原始请求内容
# @app.post("/chat/completions")
# async def chat(request: Request):
#     print(f"method: {request.method}")
#     print(f"url: {request.url}")
#     print(f"headers: {request.headers}")
#     print(f"query: {request.query_params}")
#     print(f"path: {request.path_params}")
#     print(f"host: {request.client.host}")
#     print(f"port: {request.client.port}")
#     print(f"body: {await request.body()}")
#     print(f"cookies: {request.cookies}")


@app.post("/chat/completions")
async def chat(request: Request):

    headers = request.headers
    body = json.loads(await request.body())

    print(f"headers: {headers}")
    print(f"body: {body}")

    if body["stream"] == True:
        messages = body["messages"][-20:]

        print(messages[-3:])

        async def generator():
            try:
                async for token in MAS.stream_generator(query=messages, user_id=headers.get("x-openwebui-user-id"), collection_name=["disney"]):
                    if token:
                        event = {
                            "id": str(uuid.uuid4()),
                            "object": "chat.completion.chunk",
                            "model": body["model"],
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(event)}\n\n"
                        # print(f"data: {json.dumps(event)}\n\n")
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"智能体响应异常:{e}"
        return StreamingResponse(generator(), media_type="text/event-stream")
    return {"error": "stream must be true"}


# @app.post("/chat/completions", response_class=EventSourceResponse)
# async def chat(request: Request_content):

#     print(request)

#     if request.stream:
#         messages = request.messages[-10:]
#         print(messages)

#         async def generator():
#             try:
#                 async for token in MAS.stream_generator(query=messages, thread_id="null", collection_name=["disney"]):
#                     if token:
#                         yield ServerSentEvent(data=token)
#                 yield ServerSentEvent(raw_data="[DONE]")
#             except Exception as e:
#                 yield f"智能体响应异常:{e}"

#         return EventSourceResponse(generator(), media_type="text/event-stream")
#     return {"error": "stream must be true"}


# 访问 http://localhost:8000/openapi.json 获取 OpenAPI JSON
# 访问 http://localhost:8000/docs 查看交互式文档
# 导出 curl http://localhost:8000/openapi.json > openapi.json


if __name__ == "__main__":
    uvicorn.run(app="__main__:app", host="0.0.0.0", port=8000, reload=True)
