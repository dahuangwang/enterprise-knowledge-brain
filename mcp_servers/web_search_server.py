"""
mcp_servers/web_search_server.py
────────────────────────────────────────────────────────────────────────────
MCP 架构：外部情报与联网搜索 MCP Server (基于 SSE)
提供基于 Web Researcher Agent 的多步智能检索服务。
"""
import sys
import os
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from core.registry import AgentCard, RegistryClient
from core.config import settings
from core.logger import get_logger
from agents.web_researcher_agent import run_web_researcher_agent

logger = get_logger("mcp_web_search_server")

# ============================================================================
# § 1  显式定义 Schema
# ============================================================================
class WebResearchSchema(BaseModel):
    query: str = Field(..., description="需要通过互联网进行多步情报检索的自然语言问题。例如：'搜索并总结微软最近发布的财报亮点'")

# ============================================================================
# § 2  初始化 MCP Server 并注册 Tools
# ============================================================================
mcp_server = Server("web-search-server")

@mcp_server.tool()
async def conduct_web_research(query: str) -> str:
    """使用智能搜索助手在互联网上进行多步深度调研，获取实时资讯、外部情报或百科知识，并返回综合研判报告。"""
    try:
        logger.info(f"接收到联网检索请求: {query}")
        report = await run_web_researcher_agent(query)
        return report
    except Exception as e:
        logger.error(f"Web Research Execution Error: {e}")
        return f"互联网检索工具执行故障: {e}"


# ============================================================================
# § 3  Streamable HTTP 传输层与 Starlette 路由绑定
# ============================================================================
session_manager = StreamableHTTPSessionManager(mcp_server)

async def handle_mcp(request: Any):
    """处理 MCP Client 的流式请求"""
    await session_manager.handle_request(request.scope, request.receive, request._send)

async def health_check(request: Any):
    return JSONResponse({"status": "ok", "service": "web-search-mcp"})

# ============================================================================
# § 4  生命周期与 AgentCard 注册
# ============================================================================
websearch_agent_card = AgentCard(
    agent_id="websearch_worker_01",
    agent_name="WebSearch_Worker",
    description="负责利用互联网进行多步搜索与情报收集，获取最新的外部资讯与百科知识。",
    endpoint=settings.mcp_web_search_server_url,
    capabilities=["web_search", "research", "news", "external_info"],
    mcp_tools_summary=[
        {"name": "conduct_web_research", "description": "使用智能搜索助手在互联网上进行多步深度调研"}
    ]
)

@asynccontextmanager
async def lifespan(app: Starlette):
    await RegistryClient.register(websearch_agent_card, ttl=30)
    
    async def heartbeat_loop():
        while True:
            await asyncio.sleep(15)
            await RegistryClient.heartbeat("websearch_worker_01", ttl=30)
            
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    
    yield
    
    heartbeat_task.cancel()
    await RegistryClient.unregister("websearch_worker_01")

# 挂载路由
app = Starlette(routes=[
    Route("/health", endpoint=health_check, methods=["GET"]),
    Route("/mcp", endpoint=handle_mcp, methods=["GET", "POST"])
], lifespan=lifespan)

if __name__ == "__main__":
    logger.info("启动 Web Search MCP Server (Streamable HTTP) 端口: 8005")
    uvicorn.run(app, host="0.0.0.0", port=8005, workers=1)
