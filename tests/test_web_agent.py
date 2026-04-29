import asyncio
import os

import pytest

from agents.web_researcher_agent import run_web_researcher_agent
from tools.web_tools import search_web, fetch_and_summarize_webpage

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_WEB_TESTS") != "1",
    reason="Web researcher tests require external network and LLM access; set RUN_WEB_TESTS=1 to enable.",
)

@pytest.mark.asyncio
async def test_search_web():
    """测试 DuckDuckGo 搜索功能"""
    query = "微软 2024 Q1 财报"
    result = await search_web(query, max_results=2)
    print("\n[搜索结果]")
    print(result)
    assert len(result) > 0
    assert "搜索执行失败" not in result

@pytest.mark.asyncio
async def test_fetch_and_summarize():
    """测试 Playwright 抓取和 LLM 摘要提取功能"""
    url = "https://cn.bing.com/search?q=test"
    result = await fetch_and_summarize_webpage(url, "有什么新闻")
    print("\n[网页抓取结果]")
    print(result)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_web_researcher_agent_full():
    """测试整个 Agent 多步工作流"""
    query = "请总结一下 OpenAI 最新的 Sora 模型的核心亮点和发布时间。"
    result = await run_web_researcher_agent(query)
    print("\n[Agent 综合调查报告]")
    print(result)
    assert len(result) > 0
    assert "执行异常" not in result

if __name__ == "__main__":
    # 手动触发测试
    asyncio.run(test_web_researcher_agent_full())
