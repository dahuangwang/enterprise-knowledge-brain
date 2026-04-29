"""
agents/web_researcher_agent.py
────────────────────────────────────────────────────────────────────────────
【新增文件】基于 LangGraph 的 Web Researcher Agent

功能：
  专职负责互联网信息的情报收集，支持多步搜索与推理。
  工作流：
  1. plan_search_node: 分析用户问题，生成 1-3 个针对性的搜索关键词。
  2. execute_search_node: 调用 DuckDuckGo 执行搜索。
  3. select_urls_node: 根据搜索结果的摘要，筛选出最值得深度抓取的 URL。
  4. fetch_content_node: 调用 Playwright 抓取页面并进行 LLM 压缩。
  5. synthesize_node: 综合搜索摘要和深研内容，生成最终情报报告。
"""
from typing import Any, Dict, List, Optional
import json
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from core.logger import get_logger
from core.config import settings
from tools.web_tools import search_web, fetch_and_summarize_webpage

logger = get_logger("web_researcher_agent")

# ============================================================================
# § 1  LangGraph 状态与 Schema 定义
# ============================================================================

class WebAgentState(TypedDict, total=False):
    question: str
    search_queries: List[str]
    search_results: str
    selected_urls: List[str]
    web_contents: str
    answer: str
    error: Optional[str]

class PlanSearchSchema(BaseModel):
    queries: List[str] = Field(..., description="为回答用户问题生成的 1 到 3 个搜索引擎查询关键词")

class SelectUrlSchema(BaseModel):
    urls: List[str] = Field(..., description="从搜索结果中挑选的，最需要深度抓取正文的 URL 列表（最多选 3 个）。如果摘要已足够回答问题，可返回空列表")

# ============================================================================
# § 2  LLM JSON 辅助调用
# ============================================================================

async def _llm_json_call_async(
    client: AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
) -> Optional[dict]:
    try:
        response = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            timeout=30,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"WebAgent LLM JSON 调用失败: {e}")
        return None

# ============================================================================
# § 3  LangGraph 节点
# ============================================================================

async def plan_search_node(state: WebAgentState, llm_client: AsyncOpenAI) -> dict:
    question = state.get("question", "")
    logger.info(f"[WebAgent] 正在规划搜索: {question}")
    
    sys_prompt = "你是一个情报搜索专家。请根据用户的问题，生成 1 到 3 个最精准的搜索引擎查询词（Query）。以 JSON 格式输出，包含 `queries` 字段（字符串数组）。"
    user_prompt = f"用户问题: {question}\n请输出 JSON:"
    
    data = await _llm_json_call_async(llm_client, sys_prompt, user_prompt)
    if not data or "queries" not in data:
        # Fallback
        return {"search_queries": [question]}
    
    queries = data["queries"][:3] # 限制最多 3 个
    logger.info(f"[WebAgent] 规划出的搜索词: {queries}")
    return {"search_queries": queries, "error": None}


async def execute_search_node(state: WebAgentState) -> dict:
    queries = state.get("search_queries", [])
    if not queries:
        return {"search_results": "无搜索词。"}
        
    logger.info(f"[WebAgent] 开始执行搜索，词数: {len(queries)}")
    # 并发执行所有搜索
    tasks = [search_web(q, max_results=4) for q in queries]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    
    combined_results = []
    for q, res in zip(queries, results_list):
        if isinstance(res, Exception):
            combined_results.append(f"【搜索词: {q} 失败: {res}】")
        else:
            combined_results.append(str(res))
            
    final_res = "\n\n".join(combined_results)
    return {"search_results": final_res}


async def select_urls_node(state: WebAgentState, llm_client: AsyncOpenAI) -> dict:
    search_results = state.get("search_results", "")
    question = state.get("question", "")
    
    sys_prompt = (
        "你是一个情报筛选助手。你的任务是从搜索结果列表中挑选出最有价值的 1 到 3 个 URL，"
        "以便系统进一步抓取其全文。只选择必须阅读全文才能回答问题的 URL（如新闻详情、财报原文件等）。"
        "如果搜索结果的摘要已经足够回答问题，请返回空列表。\n"
        "以 JSON 格式输出，必须包含 `urls` 字段（字符串数组）。"
    )
    user_prompt = f"用户问题: {question}\n\n搜索结果摘要:\n{search_results}\n\n请输出 JSON:"
    
    data = await _llm_json_call_async(llm_client, sys_prompt, user_prompt)
    if not data or "urls" not in data:
        return {"selected_urls": []}
        
    urls = data["urls"][:3]
    logger.info(f"[WebAgent] 决定深度抓取 {len(urls)} 个 URL")
    return {"selected_urls": urls}


async def fetch_content_node(state: WebAgentState) -> dict:
    urls = state.get("selected_urls", [])
    question = state.get("question", "")
    
    if not urls:
        return {"web_contents": "未选择深度抓取的 URL。"}
        
    logger.info(f"[WebAgent] 开始并行抓取 {len(urls)} 个网页...")
    tasks = [fetch_and_summarize_webpage(u, question) for u in urls]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    
    contents = []
    for u, res in zip(urls, results_list):
        if isinstance(res, Exception):
            contents.append(f"【URL抓取失败: {u}】\n异常: {res}")
        else:
            contents.append(str(res))
            
    return {"web_contents": "\n\n=====\n\n".join(contents)}


async def synthesize_node(state: WebAgentState, llm_client: AsyncOpenAI) -> dict:
    question = state.get("question", "")
    search_results = state.get("search_results", "")
    web_contents = state.get("web_contents", "")
    
    logger.info("[WebAgent] 正在合成最终调研报告...")
    sys_prompt = (
        "你是一个专业的情报分析专家。请根据提供的“搜索结果摘要”和“深度抓取内容”，"
        "为用户的问题生成一份专业、详实且条理清晰的调查报告。\n"
        "要求：\n"
        "1. 结论明确，数据准确。\n"
        "2. 尽量在回答中引用来源信息的 URL。\n"
        "3. 如果提供的信息不足以完美回答，请基于已有信息给出最佳推测，并明确指出现有情报的局限性。"
    )
    user_prompt = (
        f"用户问题: {question}\n\n"
        f"【基础搜索摘要】:\n{search_results}\n\n"
        f"【深度调研内容】:\n{web_contents}"
    )
    
    try:
        response = await llm_client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        answer = response.choices[0].message.content.strip()
        logger.info("[WebAgent] 报告合成完毕。")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"报告合成失败: {e}")
        return {"answer": f"报告生成失败，内部系统异常: {e}"}

# ============================================================================
# § 4  LangGraph 图构建与执行
# ============================================================================

def build_web_agent(llm_client: AsyncOpenAI):
    import functools
    
    _plan = functools.partial(plan_search_node, llm_client=llm_client)
    _select = functools.partial(select_urls_node, llm_client=llm_client)
    _synth = functools.partial(synthesize_node, llm_client=llm_client)
    
    graph = StateGraph(WebAgentState)
    graph.add_node("plan_search", _plan)
    graph.add_node("execute_search", execute_search_node)
    graph.add_node("select_urls", _select)
    graph.add_node("fetch_content", fetch_content_node)
    graph.add_node("synthesize", _synth)
    
    graph.add_edge(START, "plan_search")
    graph.add_edge("plan_search", "execute_search")
    graph.add_edge("execute_search", "select_urls")
    graph.add_edge("select_urls", "fetch_content")
    graph.add_edge("fetch_content", "synthesize")
    graph.add_edge("synthesize", END)
    
    return graph.compile()

async def run_web_researcher_agent(question: str) -> str:
    """
    外部调用的主入口
    """
    llm_client = AsyncOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
    
    agent = build_web_agent(llm_client)
    initial_state: WebAgentState = {
        "question": question,
        "search_queries": [],
        "search_results": "",
        "selected_urls": [],
        "web_contents": "",
        "answer": "",
        "error": None
    }
    
    try:
        # LangGraph Python SDK 支持 ainvoke
        final_state = await agent.ainvoke(initial_state)
        return final_state.get("answer", "未能生成有效回答。")
    except Exception as e:
        logger.error(f"WebAgent 执行异常: {e}")
        return f"互联网检索 Agent 执行失败: {e}"
