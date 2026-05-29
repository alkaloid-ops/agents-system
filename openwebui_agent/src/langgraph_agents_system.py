# src/langgraph_agents_system.py
# -*- coding: utf-8 -*-

import asyncio
import base64
from typing import List

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_agents_config import AgentState
from langgraph_agents_tools import db_info
from langgraph_agents_config import router_model_config, instant_model_config, expert_model_config, retrieve_model_config
from langgraph_agents_config import router_agent_config, instant_agent_config, expert_agent_config, retrieve_agent_config
from langgraph_agents_system_running_logs import rebuild_input_messages, get_final_answer, save_log


class MultiAgentsSystem:

    def __init__(self):

        self.memory = InMemorySaver()

        self.router_agent = create_agent(
            **router_agent_config,
            model=init_chat_model(**router_model_config),
        )

        self.instant_agent = create_agent(
            **instant_agent_config,
            model=init_chat_model(**instant_model_config),
        )

        self.expert_agent = create_agent(
            **expert_agent_config,
            model=init_chat_model(**expert_model_config),
        )

        self.retrieve_agent = create_agent(
            **retrieve_agent_config,
            model=init_chat_model(**retrieve_model_config),
            context_schema=db_info,
        )

        self.agents_system = self._build_graph()

    async def _router_agent_node(self, state: AgentState):

        latest_message = state["messages"][-1] if isinstance(
            state["messages"], list) else state["messages"]

        response = await self.router_agent.ainvoke(
            input={
                "messages": [latest_message]
                if isinstance(latest_message, HumanMessage)
                else []
            },
        )

        parsed = response.get("structured_response")

        return {
            "query": parsed.query,
            "next_executor": parsed.next_executor,
            "router_output": response,
        }

    async def _instant_agent_node(self, state: AgentState):

        response = await self.instant_agent.ainvoke(
            input={
                "messages": state["messages"]
            },
        )

        asyncio.shield(asyncio.create_task(save_log(response={"router": state.get("router_output"), "executor": response}, user_id=state.get("user_id"),
                                                    saving_path="/app/logs/langgraph_agents_system_logs.jsonl")))

        return {
            "messages": [AIMessage(content=get_final_answer(response))],
            "next_executor": None,
        }

    async def _expert_agent_node(self, state: AgentState):

        response = await self.expert_agent.ainvoke(
            input={
                "messages": state["messages"]
            },
        )

        asyncio.shield(asyncio.create_task(save_log(response={"router": state.get("router_output"), "executor": response}, user_id=state.get("user_id"),
                                                    saving_path="/app/logs/langgraph_agents_system_logs.jsonl")))

        return {
            "messages": [AIMessage(content=get_final_answer(response))],
            "next_executor": None,
        }

    async def _retrieve_agent_node(self, state: AgentState):

        response = await self.retrieve_agent.ainvoke(
            input={
                "messages": state["messages"]
            },
            context=db_info(collection_name=state.get("collection_name"))
        )

        asyncio.shield(asyncio.create_task(save_log(response={"router": state.get("router_output"), "executor": response}, user_id=state.get("user_id"),
                                                    saving_path="/app/logs/langgraph_agents_system_logs.jsonl")))

        return {
            "messages": [AIMessage(content=get_final_answer(response))],
            "next_executor": None,
        }

    def _error_handler_node(self, state: AgentState):

        failure = state.get("failure", "unknown error")

        if "No relevant context" in failure:
            message = "未找到相关信息，建议补充更具体的问题。"

        elif "hallucination" in failure.lower():
            message = "回答可能不准确，已中止，请重新提问。"

        else:
            message = f"系统检测到问题：{failure}，请重试。"

        return {
            "messages": [AIMessage(content=message)],
            "next_executor": None,
            "failure": failure,
        }

    def _router(self, state: AgentState):

        next_executor = state.get("next_executor")

        if next_executor is None:
            return END

        if next_executor == "instant_agent":
            return "instant_agent_node"

        if next_executor == "expert_agent":
            return "expert_agent_node"

        if next_executor == "retrieve_agent":
            return "retrieve_agent_node"

        print(f"[Router Warning] Unknown operator: {next_executor}")

        return "error_handler_node"

    def _build_graph(self):

        graph = StateGraph(AgentState)

        graph.add_node("router_agent_node", self._router_agent_node)

        graph.add_node("instant_agent_node", self._instant_agent_node)

        graph.add_node("expert_agent_node", self._expert_agent_node)

        graph.add_node("retrieve_agent_node", self._retrieve_agent_node)

        graph.add_node("error_handler_node", self._error_handler_node)

        graph.set_entry_point("router_agent_node")

        graph.add_conditional_edges(
            "router_agent_node",
            self._router,
            {
                "instant_agent_node": "instant_agent_node",
                "expert_agent_node": "expert_agent_node",
                "retrieve_agent_node": "retrieve_agent_node",
                "error_handler_node": "error_handler_node",
                END: END,
            }
        )

        graph.add_edge("instant_agent_node", END)

        graph.add_edge("expert_agent_node", END)

        graph.add_edge("retrieve_agent_node", END)

        graph.add_edge("error_handler_node", END)

        return graph.compile(
            checkpointer=self.memory
        )

    async def running(self, query: str | list, user_id: str, collection_name: List[str]):

        messages = rebuild_input_messages(query)

        async for event in self.agents_system.astream_events(
            input={
                "messages": messages,
                "user_id": user_id,
                "collection_name": collection_name,
            },
            config={"configurable": {"thread_id": user_id}},
            stream_mode="messages",
            version="v2",
        ):

            if event["event"] == "on_chat_model_stream":
                if event["data"]["chunk"].content:
                    print(event["data"]["chunk"].content, flush=True, end='')
                # if "__interrupt__" in event["data"]:
                #     print(f"\n\nInterrupt: {event['data']['__interrupt__']}")

    async def stream_generator(self, query: str | list, user_id: str, collection_name: List[str]):

        messages = rebuild_input_messages(query)

        async for event in self.agents_system.astream_events(
            input={
                "messages": messages,
                "user_id": user_id,
                "collection_name": collection_name,
            },
            config={"configurable": {"thread_id": user_id}},
            stream_mode="messages",
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                if event["data"]["chunk"].content:
                    yield event["data"]["chunk"].content


async def main():

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    queries = [
        # "你好呀, 我的名字是alkaloid, 请介绍一下你自己.",
        # "你还记得我的名字吗?",
        # "请帮我查询上海迪士尼的营业时间",
        "截止目前最新的2026WWDC的5条rumors新闻",
        # [{"role":"user","content":[{"type": "text", "text": "请提取图片中的文字"},{"type": "image_url", "image_url": f'data:image/jpeg;base64,{encode_image(image_path="./IMG_0110.PNG")}'}]}],
    ]
    MAS = MultiAgentsSystem()

    for q in queries:
        await MAS.running(
            query=q,
            user_id="admin",
            collection_name=["disney"]
        )

if __name__ == "__main__":
    asyncio.run(main())
