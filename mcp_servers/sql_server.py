"""
mcp_servers/sql_server.py
────────────────────────────────────────────────────────────────────────────
MCP 架构第一期：金融沙箱数据库 MCP Server (基于 SSE)
提供跨平台的企业报表工具，包含沙箱安全隔离。
"""
import sys
import os
from typing import Any

# 将根目录添加到环境变量，以使包导入正常工作
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
from tools.sql_executor import execute_sandbox_sql

logger = get_logger("mcp_sql_server")

# ============================================================================
# § 1  显式定义 Schema (满足代码约束 5)
# ============================================================================
class ExecuteSQLSchema(BaseModel):
    query: str = Field(..., description="需要在财务沙箱数据库中执行的 SELECT SQL 语句")

# ============================================================================
# § 2  初始化 MCP Server 并注册 Tools (满足代码约束 2, 6)
# ============================================================================
mcp_server = Server("financial-sql-server")

@mcp_server.tool()
async def execute_financial_sql(query: str) -> str:
    """执行 SQL 查询来获取公司财务与营收报表 (仅限 SELECT)。表名为 financial_reports(project_name, investment, revenue, start_date)"""
    try:
        return execute_sandbox_sql(query)
    except Exception as e:
        logger.error(f"SQL Execution Error: {e}")
        return f"数据库执行故障: {e}"


# ============================================================================
# § 3  Streamable HTTP 传输层与 Starlette 路由绑定
# ============================================================================
session_manager = StreamableHTTPSessionManager(mcp_server)

async def handle_mcp(request: Any):
    """处理 MCP Client 的流式请求"""
    await session_manager.handle_request(request.scope, request.receive, request._send)

async def health_check(request: Any):
    return JSONResponse({"status": "ok", "service": "sql-mcp"})

# ============================================================================
# § 4  生命周期与 AgentCard 注册
# ============================================================================
sql_agent_card = AgentCard(
    agent_id="sql_worker_01",
    agent_name="SQL_Worker",
    description="负责查询结构化的数据库、财务报表、经营数据。适合回答涉及具体数值、统计、汇总等精确数据问题。",
    endpoint=settings.mcp_sql_server_url,
    capabilities=["database", "finance", "metrics", "sql"],
    mcp_tools_summary=[
        {"name": "execute_financial_sql", "description": "执行 SQL 查询来获取公司财务与营收报表 (仅限 SELECT)"}
    ]
)

@asynccontextmanager
async def lifespan(app: Starlette):
    # 注册 AgentCard
    await RegistryClient.register(sql_agent_card, ttl=30)
    
    # 开启心跳任务
    async def heartbeat_loop():
        while True:
            await asyncio.sleep(15)
            await RegistryClient.heartbeat("sql_worker_01", ttl=30)
            
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    
    yield
    
    # 清理任务与注销
    heartbeat_task.cancel()
    await RegistryClient.unregister("sql_worker_01")

# 挂载路由
app = Starlette(routes=[
    Route("/health", endpoint=health_check, methods=["GET"]),
    Route("/mcp", endpoint=handle_mcp, methods=["GET", "POST"])
], lifespan=lifespan)

if __name__ == "__main__":
    logger.info("启动 SQL MCP Server (Streamable HTTP) 端口: 8002")
    uvicorn.run(app, host="0.0.0.0", port=8002, workers=1)
