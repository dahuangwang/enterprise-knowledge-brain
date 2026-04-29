"""
mcp_servers/api_server.py
────────────────────────────────────────────────────────────────────────────
MCP 架构第一期：内部业务网关 MCP Server (基于 SSE)
提供跨平台的企业网关工具，使大模型可以直接作为 Client 连接。
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
from tools.internal_apis import (
    _get_project_approval_status, 
    _get_department_budget, 
    _get_employee_tickets
)

logger = get_logger("mcp_api_server")

# ============================================================================
# § 1  显式定义 Schema (满足代码约束 5)
# ============================================================================
class ProjectApprovalSchema(BaseModel):
    project_name: str = Field(..., description="需要查询审批状态的项目名称，例如 '极速开户系统'")

class DepartmentBudgetSchema(BaseModel):
    department: str = Field(..., description="需要查询可用经费的部门名称，例如 '金融科技部'")

class EmployeeTicketsSchema(BaseModel):
    employee_name: str = Field(..., description="需要查询待办工单的员工姓名，例如 '李四'")


# ============================================================================
# § 2  初始化 MCP Server 并注册 Tools (满足代码约束 2, 6)
# ============================================================================
mcp_server = Server("enterprise-api-server")

@mcp_server.tool()
async def get_approval_status(project_name: str) -> str:
    """查询企业内部项目的审批流状态"""
    try:
        return _get_project_approval_status(project_name)
    except Exception as e:
        logger.error(f"API Error: {e}")
        return f"接口调用异常: {e}"

@mcp_server.tool()
async def get_department_budget(department: str) -> str:
    """查询企业内部部门的可用财务预算"""
    try:
        return _get_department_budget(department)
    except Exception as e:
        logger.error(f"API Error: {e}")
        return f"接口调用异常: {e}"

@mcp_server.tool()
async def get_employee_tickets(employee_name: str) -> str:
    """查询指定员工的内部未处理紧急工单数量"""
    try:
        return _get_employee_tickets(employee_name)
    except Exception as e:
        logger.error(f"API Error: {e}")
        return f"接口调用异常: {e}"


# ============================================================================
# § 3  Streamable HTTP 传输层与 Starlette 路由绑定
# ============================================================================
session_manager = StreamableHTTPSessionManager(mcp_server)

async def handle_mcp(request: Any):
    """处理 MCP Client 的流式请求"""
    await session_manager.handle_request(request.scope, request.receive, request._send)

async def health_check(request: Any):
    return JSONResponse({"status": "ok", "service": "api-mcp"})

# ============================================================================
# § 4  生命周期与 AgentCard 注册
# ============================================================================
api_agent_card = AgentCard(
    agent_id="api_worker_01",
    agent_name="API_Worker",
    description="负责调用内部系统集成接口（如查预算、查审批、查工单）。",
    endpoint=settings.mcp_api_server_url,
    capabilities=["api", "internal", "budget", "approval", "ticket"],
    mcp_tools_summary=[
        {"name": "get_approval_status", "description": "查询项目审批流状态"},
        {"name": "get_department_budget", "description": "查询部门可用预算"},
        {"name": "get_employee_tickets", "description": "查询员工紧急工单"}
    ]
)

@asynccontextmanager
async def lifespan(app: Starlette):
    # 注册 AgentCard
    await RegistryClient.register(api_agent_card, ttl=30)
    
    # 开启心跳任务
    async def heartbeat_loop():
        while True:
            await asyncio.sleep(15)
            await RegistryClient.heartbeat("api_worker_01", ttl=30)
            
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    
    yield
    
    # 清理任务与注销
    heartbeat_task.cancel()
    await RegistryClient.unregister("api_worker_01")

# 挂载路由
app = Starlette(routes=[
    Route("/health", endpoint=health_check, methods=["GET"]),
    Route("/mcp", endpoint=handle_mcp, methods=["GET", "POST"])
], lifespan=lifespan)

if __name__ == "__main__":
    logger.info("启动 API MCP Server (Streamable HTTP) 端口: 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)
