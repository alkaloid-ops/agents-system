# src/openai_agents_system.py
# -*- coding: utf-8 -*-

import base64
import asyncio
from typing import List, Dict
from agents import Agent, Runner, RunConfig, SQLiteSession, SessionSettings

from openai.types.responses import ResponseReasoningSummaryTextDeltaEvent, ResponseFunctionCallArgumentsDeltaEvent, ResponseTextDeltaEvent

from openai_agents_config import instant_agent_config, expert_agent_config, retrieve_agent_config, router_agent_config
from openai_agents_tools import db_info

from openai_agents_system_running_log import save_log


class MultiAgentsSystem:

    def __init__(self):

        try:

            self.instant_agent = Agent(**instant_agent_config)

            self.expert_agent = Agent(**expert_agent_config)

            self.retrieve_agent = Agent[db_info](**retrieve_agent_config)

            self.router_agent = Agent(**router_agent_config,
                                      handoffs=[self.instant_agent, self.expert_agent, self.retrieve_agent,])

            self.sessions: Dict[str, SQLiteSession] = {}

            self.session_lock = asyncio.Lock()

        except Exception as e:
            print(f"智能体实例化异常: {e}")

    async def create_session(self, session_id: str,) -> SQLiteSession:

        async with self.session_lock:

            if session_id not in self.sessions:

                self.sessions[session_id] = SQLiteSession(
                    session_id=session_id,
                    # db_path="/app/database/sql/conversations_history.db",
                )

            return self.sessions[session_id]

    async def running(self, query, session_id: str, collection_name: List[str]):

        session = await self.create_session(session_id)

        context = db_info(collection_name=collection_name)

        response = Runner.run_streamed(
            starting_agent=self.router_agent,
            input=query,
            session=session,
            context=context,
            run_config=RunConfig(
                session_settings=SessionSettings(limit=100),
                tracing_disabled=True,
            ),
        )

        async for event in response.stream_events():

            if event.type == "raw_response_event":

                if isinstance(event.data,  ResponseReasoningSummaryTextDeltaEvent):
                    print(event.data.delta, flush=True, end='')
                elif isinstance(event.data,  ResponseFunctionCallArgumentsDeltaEvent):
                    print(event.data.delta, flush=True, end='')
                elif isinstance(event.data,  ResponseTextDeltaEvent):
                    print(event.data.delta, flush=True, end='')

        asyncio.shield(asyncio.create_task(save_log(user_id=session_id, user_query=query[-1].get("content") if isinstance(query, list) else query, response=response,
                                                    saving_path="/app/logs/openai_agents_system_logs.jsonl")))

    async def stream_generator(self, query, session_id: str, collection_name: List[str]):

        session = await self.create_session(session_id)

        context = db_info(collection_name=collection_name)

        response = Runner.run_streamed(
            starting_agent=self.router_agent,
            input=query,
            session=session,
            context=context,
            run_config=RunConfig(
                session_settings=SessionSettings(limit=100),
                tracing_disabled=True,
            ),
        )

        async for event in response.stream_events():
            token = None
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseReasoningSummaryTextDeltaEvent):
                    token = event.data.delta
                # elif isinstance(event.data, ResponseFunctionCallArgumentsDeltaEvent):
                #     token = event.data.delta
                elif isinstance(event.data, ResponseTextDeltaEvent):
                    token = event.data.delta

            if token:
                yield token

        asyncio.shield(asyncio.create_task(save_log(user_id=session_id, user_query=query[-1].get("content") if isinstance(query, list) else query, response=response,
                                                    saving_path="/app/logs/openai_agents_system_logs.jsonl")))


# 图片OCR任务测试
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")

if __name__ == "__main__":
    MAS = MultiAgentsSystem()
    queries = [
        #  [{
        #     "role": "user",
        #     "content": [
        #         { "type": "input_text", "text": "请提取图片中的文字" },
        #         # {
        #         #     "type": "input_image",
        #         #     "image_url": f"data:image/jpeg;base64,{encode_image(image_path="./IMG_0110.PNG")}",
        #         # },
        #     ],
        # }],
        "你好, 我叫alkaloid(阿卡洛伊德), 我喜欢日本料理, 请介绍一下你自己.",
        # "请收集最新的伊朗局势并简要告诉我.",
        # "请帮我查询明天东京的天气",
        # "请根据我的喜好推荐一些.",
        # "你还记得我叫什么名字吗?",
        # "如何在上海迪士尼度假区查看和安排与迪士尼朋友的见面时间及地点？",
        # "AirPods Pro 3 的评价如何?",
    ]
    for q in queries:
        asyncio.run(MAS.running(query=q, session_id="admin",
                    collection_name=["disney"]))
