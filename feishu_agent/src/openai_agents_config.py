# src/openai_agents_config.py
# -*- coding: utf-8 -*-

from openai_agents_tools import get_current_date, get_weather, retrieve_knowledgebase
from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel, ModelSettings, Agent
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'


# load_dotenv("../../.env")


class EvaluatorOutput(BaseModel):
    answer: Literal["OK", "NOT OK",
                    "Don't retrieve any more, Try to do web search."]


router_agent_config = {
    "name": "router_agent",
    "model": OpenAIChatCompletionsModel(
        model=os.getenv("ROUTER_AGENT_MODEL"),
        openai_client=AsyncOpenAI(api_key=os.getenv(
            "OPENAI_API_KEY"), base_url=os.getenv("BASE_URL")),
    ),
    "model_settings": ModelSettings(temperature=0, include_usage=True, extra_body={"enable_thinking": False}),
    "instructions": '''
<role>
You are a task router. 
Your only job is to classify the user query and output the next executor. Never generate an answer.
</role>

<executors>
- instant_agent: simple, deterministic tasks (greetings, text processing, single-step arithmetic, direct tool queries).
- expert_agent: tasks requiring reasoning, multi-step logic, analysis, or tool-based tasks (date, weather, flight etc.), also include image OCR task.
- retrieve_agent: only about tasks requiring external knowledge retrieval (document QA), especial about disney.
- Fallback: if uncertain, default to expert_agent.
</executors>

<output_contract>
- answer: must be null
- Do NOT output any additional text or explanation.
</output_contract>
''',
}


instant_agent_config = {
    "name": "instant_agent",
    "model": OpenAIChatCompletionsModel(
        model=os.getenv("INSTANT_AGENT_MODEL"),
        openai_client=AsyncOpenAI(api_key=os.getenv(
            "OPENAI_API_KEY"), base_url=os.getenv("BASE_URL")),
    ),
    "model_settings": ModelSettings(temperature=0.5, tool_choice="auto", parallel_tool_calls=True, include_usage=True, extra_body={"enable_thinking": False}),
    "tools": [get_current_date],
    "handoff_description": "A simple task processing agent who is able to quickly answer questions that do not require complex reasoning or external knowledge retrieval.",
    "instructions": '''
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
'''
}


expert_agent_config = {
    "name": "expert_agent",
    "model": OpenAIChatCompletionsModel(
        model=os.getenv("EXPERT_AGENT_MODEL"),
        openai_client=AsyncOpenAI(api_key=os.getenv(
            "OPENAI_API_KEY"), base_url=os.getenv("BASE_URL")),
    ),
    "model_settings": ModelSettings(
        temperature=0.5,
        tool_choice="auto",
        parallel_tool_calls=True,
        include_usage=True,
        extra_body={"enable_thinking": True, "enable_search": True}
    ),
    "tools": [get_current_date, get_weather],
    "handoff_description": "An expert agent who is able to handle complex task which need single-step or muti-steps thinking before generating final answer.",
    "instructions": '''
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
'''
}


retrieve_results_evaluator = {
    "name": "retrieve_results_evaluator",
    "model": OpenAIChatCompletionsModel(
        model=os.getenv("RETRIEVE_EVALUATOR_MODEL"),
        openai_client=AsyncOpenAI(api_key=os.getenv(
            "OPENAI_API_KEY"), base_url=os.getenv("BASE_URL")),
    ),
    "model_settings": ModelSettings(temperature=0, include_usage=True),
    "output_type": EvaluatorOutput,
    "instructions": '''
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
'''
}


retrieve_agent_config = {
    "name": "retrieve_agent",
    "model": OpenAIChatCompletionsModel(
        model=os.getenv("RETRIEVE_AGENT_MODEL"),
        openai_client=AsyncOpenAI(api_key=os.getenv(
            "OPENAI_API_KEY"), base_url=os.getenv("BASE_URL")),
    ),
    "model_settings": ModelSettings(
        temperature=0.1,
        tool_choice="required",
        parallel_tool_calls=False,
        include_usage=True,
        extra_body={"enable_thinking": True, "enable_search": True}
    ),
    "tools": [
        # web_search,
        retrieve_knowledgebase,
        Agent(**retrieve_results_evaluator).as_tool(
            tool_name="retrieve_results_evaluator",
            tool_description="The tool is to assess retrieved context and decide whether it is sufficient to answer the user query.",
        ),
    ],
    "handoff_description": "An knowledge retrieval agent who is able to handle especial task, which about the Disney resort.",
    "instructions": '''
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
'''
}
