"""
tools/web_tools.py
────────────────────────────────────────────────────────────────────────────
外部情报与联网搜索工具
包含：
1. DuckDuckGo 搜索引擎查询
2. Playwright 动态网页抓取 + LLM 智能内容摘要提取
"""
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from playwright.async_api import async_playwright
from openai import AsyncOpenAI
import json

from core.logger import get_logger
from core.config import settings

logger = get_logger("web_tools")

# 懒加载的 OpenAI 客户端实例
_llm_client = None

def get_llm_client() -> AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    return _llm_client

async def search_web(query: str, max_results: int = 5) -> str:
    """
    使用 DuckDuckGo 搜索网络，返回包含标题、链接和摘要的结果。
    """
    logger.info(f"正在通过 DDGS 搜索网络: {query}")
    try:
        results = []
        with DDGS() as ddgs:
            # text() 默认返回生成器
            for r in ddgs.text(query, max_results=max_results):
                results.append(r)
        
        if not results:
            return f"搜索 '{query}' 未找到相关结果。"
        
        # 格式化输出
        output = [f"【搜索结果：{query}】"]
        for idx, res in enumerate(results, 1):
            title = res.get("title", "无标题")
            href = res.get("href", "")
            body = res.get("body", "")
            output.append(f"{idx}. 标题: {title}\n   链接: {href}\n   摘要: {body}\n")
            
        return "\n".join(output)
    except Exception as e:
        logger.error(f"DDGS 搜索失败: {e}")
        return f"搜索执行失败: {e}"

async def fetch_and_summarize_webpage(url: str, query: str = "") -> str:
    """
    使用 Playwright 无头模式抓取动态网页，提取文本，
    并使用 LLM 根据 query (如果提供) 进行摘要压缩，避免 Context Window Overflow。
    """
    logger.info(f"准备抓取网页: {url}")
    
    html_content = ""
    try:
        # 1. 抓取网页内容
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            
            # 设置较短超时避免卡死
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            
            # 简单等待以应对部分前端动态渲染，然后提取 body 的 HTML
            try:
                await page.wait_for_timeout(2000)
            except Exception:
                pass
                
            html_content = await page.content()
            await browser.close()
            
    except Exception as e:
        logger.error(f"Playwright 抓取网页 {url} 失败: {e}")
        return f"网页抓取失败: {e}"

    # 2. BeautifulSoup 清理 HTML (去除 script/style)
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "img"]):
            tag.decompose()
        
        # 提取纯文本并合并空白字符
        raw_text = soup.get_text(separator="\n")
        cleaned_text = "\n".join(line.strip() for line in raw_text.splitlines() if line.strip())
    except Exception as e:
        logger.error(f"BeautifulSoup 清理 HTML 失败: {e}")
        return f"HTML 解析失败: {e}"

    if not cleaned_text:
        return "抓取到的网页不包含可见文本。"

    # 截断过长文本（防止单次请求直接超出大模型最大 token）
    # 假设约 15000 字符是单次摘要处理的上限
    MAX_CHARS = 15000
    if len(cleaned_text) > MAX_CHARS:
        cleaned_text = cleaned_text[:MAX_CHARS] + "\n...(内容已截断)..."

    logger.info(f"网页清理完毕，正文字符数: {len(cleaned_text)}，开始 LLM 摘要...")

    # 3. 调用 LLM 进行压缩摘要
    client = get_llm_client()
    system_prompt = (
        "你是一个专业的情报分析员。你的任务是对给定的网页原始文本进行高度压缩和总结。"
        "提取其中的核心观点、关键数据和重要事实。剔除广告、导航栏文本等无用信息。"
        "输出应保持客观真实，不编造内容。"
    )
    user_prompt = f"网页来源: {url}\n"
    if query:
        user_prompt += f"用户关注的问题/主题: {query}\n请重点提取与该主题相关的信息。\n"
    user_prompt += f"\n网页原始内容:\n{cleaned_text}"

    try:
        response = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        summary = response.choices[0].message.content.strip()
        logger.info(f"LLM 摘要完成，长度: {len(summary)} 字符。")
        return f"来源: {url}\n摘要总结:\n{summary}"
    except Exception as e:
        logger.error(f"LLM 摘要过程失败: {e}")
        # 如果大模型摘要失败，降级返回截断后的纯文本
        return f"来源: {url}\n(摘要生成失败，返回部分原文)\n" + cleaned_text[:2000]
