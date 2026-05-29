# src/openai_agents_tools.py
# -*- coding: utf-8 -*-

from retriever import main as run_retrieve
from agents import function_tool, RunContextWrapper
from typing import List
from dataclasses import dataclass
from datetime import datetime
from tavily import TavilyClient
from langchain_community.utilities import GoogleSerperAPIWrapper
import asyncio
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'


@function_tool
def get_current_date() -> str:
    '''获取当前日期'''
    return datetime.now().strftime("%Y-%m-%d")


@function_tool()
def web_search(query: str):
    """通过输入用户的询问来进行联网搜索以获取相关信息"""
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        search_depth="fast",
        time_range="year",
    )
    return f"搜索结果如下:\n\n{response["results"]}"


@dataclass
class db_info:
    collection_name: List[str]


@function_tool
async def retrieve_knowledgebase(wrapper: RunContextWrapper[db_info], query: str) -> str:
    """通过检索知识库来获取迪士尼相关信息"""

    if not wrapper.context.collection_name:
        return "错误：未配置知识库集合"

    try:
        all_chunks = []
        tasks = [run_retrieve(collection_name=collection, query=query)
                 for collection in wrapper.context.collection_name]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for collection, result in zip(wrapper.context.collection_name, results):
            if isinstance(result, Exception):
                print(f"检索集合 {collection} 失败: {result}")
                continue

            for text, score, metadata in zip(result["text"], result["score"], result["metadata"]):
                file_name = metadata.get("file_name", "未知来源")
                all_chunks.append((score, text, file_name))
    except Exception as e:
        print(f"检索异常: {e}")

    if not all_chunks:
        return f"未找到与「{query}」相关的信息"

    all_chunks.sort(key=lambda x: x[0], reverse=True)

    rebuild_context = []
    for i, (score, text, file_name) in enumerate(all_chunks, 1):
        chunk_text = f"片段{i}:(相关性:{score:.4f}, 来源:{file_name}, 内容:{text.strip()})"
        rebuild_context.append(chunk_text)

    title = f"以下是根据用户问题「{query}」检索到的相关信息片段（已按相关性从高到低排序）\n"
    return title + "\n".join(rebuild_context)
