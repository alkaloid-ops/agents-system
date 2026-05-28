# src/langgraph_agents_tools.py
# -*- coding: utf-8 -*-


from retriever import main as run_retrieve
from typing import List
from dataclasses import dataclass
from datetime import datetime
from langchain.tools import tool, ToolRuntime
from langchain_community.utilities import GoogleSerperAPIWrapper
import asyncio
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'


@tool
def get_current_date():
    """获取当前日期"""
    return datetime.now().strftime("%Y-%m-%d")


@tool
def get_weather(date: str, city: str):
    """获取某天某个城市的天气(日期格式:YYYY-MM-DD)"""
    try:
        parsed_date = datetime.strptime(date, "%Y-%m-%d")
        standardized_date = parsed_date.strftime("%Y-%m-%d")
    except ValueError:
        return {"error": f"日期格式错误，请使用 YYYY-MM-DD 格式，例如: 2026-04-22。当前输入: {date}"}

    typical = {
        "纽约": "雪/雨",
        "洛杉矶": "晴",
        "柏林": "多云",
        "巴黎": "小雨",
        "伦敦": "雾",
        "莫斯科": "雪",
        "东京": "雨",
        "上海": "雨"
    }

    if city in typical:
        return {
            "date": standardized_date,
            "city": city,
            "weather": typical[city]
        }
    else:
        return {"error": f"不支持的城市: {city}，支持的城市: {list(typical.keys())}"}


# @tool
# def web_search(query: str):
#     """通过联网搜索获取信息"""
    # with open("/AirPods Pro 3 Comment.txt", "r", encoding="utf-8") as f:
    #     search_result = f.read()
    # return f"关于<{query}>的相关搜索结果如下\n\n<{search_result}>"
    # search = GoogleSerperAPIWrapper()
    # return f"搜索结果:{search.run(query)}"


@dataclass
class db_info:
    collection_name: List[str]


@tool
async def retrieve_knowledgebase(runtime: ToolRuntime[db_info], query: str) -> str:
    """通过检索知识库来获取迪士尼相关信息"""

    if not runtime.context.collection_name:
        return "错误：未配置知识库集合"

    try:
        all_chunks = []
        tasks = [run_retrieve(collection_name=collection, query=query)
                 for collection in runtime.context.collection_name]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for collection, result in zip(runtime.context.collection_name, results):
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
