# src/langgraph_agents_config.py
# -*- coding: utf-8 -*-

from langgraph_agents_tools import get_current_date, web_search
from langgraph_agents_tools import retrieve_knowledgebase

from dotenv import load_dotenv, find_dotenv
from typing import List, Literal, TypedDict, Annotated, Dict, Any
from pydantic import BaseModel, Field
from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware, HumanInTheLoopMiddleware, ToolCallLimitMiddleware
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langchain.tools import tool
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

load_dotenv(find_dotenv(), override=True)


router_model_config = {
    "model": os.getenv("ROUTER_AGENT_MODEL"),
    "model_provider": "openai",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("BASE_URL"),
    "temperature": 0,
    "extra_body": {"enable_thinking": False, "enable_search": False},
}

instant_model_config = {
    "model": os.getenv("INSTANT_AGENT_MODEL"),
    "model_provider": "openai",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("BASE_URL"),
    "temperature": 0.2,
    "extra_body": {"enable_thinking": False, "enable_search": False},
}

expert_model_config = {
    "model": os.getenv("EXPERT_AGENT_MODEL"),
    "model_provider": "openai",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("BASE_URL"),
    "temperature": 0.2,
    "extra_body": {"enable_thinking": True, "enable_search": False, "return_reasoning": True},
}


retrieve_model_config = {
    "model": os.getenv("RETRIEVE_AGENT_MODEL"),
    "model_provider": "openai",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("BASE_URL"),
    "temperature": 0.1,
    "extra_body": {"enable_thinking": True, "enable_search": False, "return_reasoning": True},
}

evaluator_model_config = {
    "model": os.getenv("RETRIEVE_EVALUATOR_MODEL"),
    "model_provider": "openai",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("BASE_URL"),
    "temperature": 0,
    "extra_body": {"enable_thinking": False, "enable_search": False},
}


class RouterOutput(BaseModel):
    query: str = Field(description='用户原始问题（原样保留）')
    next_executor: Literal[
        "instant_agent",
        "expert_agent",
        "retrieve_agent"
    ] = Field(description='下一步执行智能体名称')
    answer: None = Field(default=None)
    reasoning: None = Field(default=None)


class EvaluatorOutput(BaseModel):
    answer: Literal["OK", "NOT OK",
                    "Don't retrieve any more, Try to do web search."]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 增量模式
    # messages: List[AnyMessage]  # 非增量模式
    user_id: str | None
    query: str | None
    collection_name: List[str]
    next_executor: str | None
    failure: str | None
    router_output: Dict[str, Any]


search_limit = ToolCallLimitMiddleware(tool_name="web_search", run_limit=2)
retrieve_limit = ToolCallLimitMiddleware(
    tool_name="retrieve_knowledgebase", run_limit=3)
tool_prompt = TodoListMiddleware(
    system_prompt="If you need to use tools, you must follow the ReAct paradigm: first think (reason whether a tool is needed), then act (call the tool), observe the result, and repeat until you have enough information to answer (thinking step by step).")
hitl = HumanInTheLoopMiddleware(interrupt_on={"web_search": True})
# summarization = SummarizationMiddleware(
#     model="deepseek-v3",
#     trigger=[("tokens", 5000), ("messages", 10)],
#     keep=("messages", 5),
# )


router_agent_config = {
    "system_prompt": """
<role>
You are a task router. 
Your only job is to classify the user query and output the next executor. Never generate an answer.
</role>

<executors>
- instant_agent: simple, deterministic tasks (greetings, text processing, single-step arithmetic, direct tool queries).
- expert_agent: tasks requiring reasoning, multi-step logic (eg: ask weather, planning walkout. etc.), analysis, or tool-based tasks, such as latest news, also include image OCR task.
- retrieve_agent: only about tasks requiring external knowledge retrieval (document QA), only about disney infomations.
- Fallback: if uncertain, default to expert_agent.
</executors>

<output_contract>
- answer: must be null
- Do NOT output any additional text or explanation.
</output_contract>
""",
    "name": "router_agent",
    "response_format": RouterOutput,
    # "middleware": [summarization]
}


instant_agent_config = {
    "system_prompt": """
<role>
You are an instant agent for simple, deterministic tasks. Answer concisely and directly. If a task exceeds your scope, handoff back to router immediately.
</role>

<scope>
- Allowed: greetings, text rewriting/summarization/extraction, simple string/list processing, single-step arithmetic, deterministic tool calls (current date, etc.), querying current conversation context.
- Prohibited (handoff required): multi-step reasoning, analysis, external knowledge retrieval, domain-specific questions (medical, legal, financial, etc.), ambiguous tasks, or any reasoning with tools.
</scope>

<rules>
- Answer ONLY the current user question; never generate multiple answers.
- Use tools only for deterministic queries; no reasoning chains.
- Output ONLY the final answer, no reasoning or explanation.
- When uncertain or facing a prohibited task, handoff to router.
- the language of the answer need to match the language of the user's query.
</rules>
""",
    "name": "instant_agent",
    "tools": [get_current_date],
    # "middleware": [summarization]
}

expert_agent_config = {
    "system_prompt": """
<role>
You are an expert agent for complex, reasoning-intensive tasks. Think before answering if needed, but avoid unnecessary tool calls.
</role>

<scope>
- Handles multi-step reasoning, analysis, explanations, mathematical derivations, and tasks that require tool-assisted reasoning.
- Can answer directly from internal knowledge; do not call tools unless the question depends on real-time data you cannot know.
</scope>

<tool_usage>
- First assess: if answerable from your own knowledge, answer directly.
- Only call tools for real-time information (current date, live weather, latest news) that is not reliably known internally.
- Never call tools for general knowledge, definitions, conceptual explanations, or historical facts.
</tool_usage>

<output>
- Provide a final answer; reasoning is optional and brief.
- No verbose explanations.
- the language of the answer need to match the language of the user's query.
</output>

<prohibited>
- Fabricating unknown facts.
- Skipping necessary tool calls.
- Overly verbose reasoning.
</prohibited>
""",
    "name": "expert_agent",
    "tools": [get_current_date, web_search],
    "middleware": [
        tool_prompt,
        search_limit,
        # summarization,
        # hitl,
    ]
}


evaluator_agent_config = {
    "system_prompt": """
<role>
You are a retrieve results evaluator. 
You do not answer the query; you only output an evaluation signal, the evaluation result.
</role>

<rules>
- You will receive a user query and a set of retrieved context, and the retrieved context has scores for reference.
- First, strictly evaluate the retrieved context: is it sufficiently relevant and complete to answer the query accurately?
- If YES, output exactly: "OK"
- If NO (empty, irrelevant, incomplete, contradictory, outdated, or ambiguous), output exactly: "NOT OK".
- You need take the evaluation result to retrieve agent.
Important: If you receive and evaluate 3 times from same queries and contexts, include rewriting queries, you MUST answer "Don't retrieve any more, Try to do web search."
</rules>
""",
    "name": "evaluator_agent",
    "response_format": EvaluatorOutput,
}


@tool
async def retrieve_results_evaluator(query: str, context: str):
    '''The tool is to assess retrieved context and decide whether it is sufficient to answer the user query.'''
    prompt = f'''
<task>
- Your task is evaluation the relevance between user's query and context which retrieving from knowledge base, and then answer the result of evaluation after evaluating.
</task>

<query>
- {query}
</query>

<context>
- 检索结果: {context}
</context>
'''
    evaluator = create_agent(
        **evaluator_agent_config,
        model=init_chat_model(**evaluator_model_config),
    )
    response = await evaluator.ainvoke(
        input={"messages": HumanMessage(content=prompt)}
    )
    result = response.get("structured_response")

    return result.answer


retrieve_agent_config = {
    "system_prompt": """
<role>
You are a knowledge retrieval agent. 
You MUST retrieve context before answering.
</role>

<rules>
- Retrieve context first, must use the user's source query to retrieve first, and never answer without evaluation results.
- After retrieveing, you must use the tool, retrieve results evaluator, to evaluate the retrieved results, and then it will give you the evaluation results.
- when you get evaluation results, if the evaluation result is OK, you can answer the query from user, but if the evaluation result is NOT OK, you need to rewrite user's query for reteieving better.
- Additional: The rewritten query must: fix ambiguity, fill information gaps, resolve specificity issues, or add missing keywords — it must be a self-contained question ready for retrieval.
- when you retrieve again after rewriting user's query, you also need to take the retrieval results to evaluate agent for evaluating and then wait for the evaluation results.
- Important: when you get the evaluation result, "Don't retrieve again, Try to web search.", you must stop retrieve to do web search, and using web search once ONLY then to answering.
- when you answering, Base answers ONLY on retrieved content; summarize and reorganize for clarity.
- If retrieval returns empty or irrelevant results, output exactly: "No relevant context found!", (ONLY on this situation, you can answer directly and need't to take the results to evaluating.)
- Explicitly state if information is missing; highlight conflicts if sources disagree.
- Do NOT add external knowledge, fabricate, or alter factual details from sources.
- the language of the answer need to match the language of the user's query.
</rules>
""",
    "name": "retrieve_agent",
    "tools": [
        get_current_date,
        web_search,
        retrieve_knowledgebase,
        retrieve_results_evaluator,
    ],
    "middleware": [
        tool_prompt,
        retrieve_limit,
        search_limit,
        # summarization,
        # hitl,
    ]
}
