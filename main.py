"""
main.py — FastAPI 应用入口
────────────────────────────────────────────────────────────────────────────
启动命令:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

交互式文档:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc   (ReDoc)
"""
from contextlib import asynccontextmanager, AsyncExitStack

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from core.config import settings
from core.logger import get_logger
from api.routes import router

logger = get_logger(__name__)

MCP_ENDPOINTS = [
    ("API_Server", settings.mcp_api_server_url),
    ("SQL_Server", settings.mcp_sql_server_url),
    ("GraphRAG_Server", settings.mcp_graphrag_server_url),
    ("DataAnalysis_Server", settings.mcp_data_analysis_server_url),
    ("WebSearch_Server", settings.mcp_web_search_server_url),
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 启动时建立连接池
    app.state.mcp_sessions = {}
    app.state.mcp_tools = []
    app.state.mcp_stack = AsyncExitStack()
    
    logger.info("🚀 Enterprise Knowledge Brain 开始启动...")
    logger.info("正在建立全局 MCP Streamable HTTP 长连接池...")
    
    try:
        for srv_name, url in MCP_ENDPOINTS:
            try:
                # 建立底层传输流
                streams = await app.state.mcp_stack.enter_async_context(streamable_http_client(url))
                # 基于双向流建立会话
                session = await app.state.mcp_stack.enter_async_context(ClientSession(streams[0], streams[1]))
                # 初始化握手
                await session.initialize()
                
                # 获取该节点暴露的工具
                tools_resp = await session.list_tools()
                for t in tools_resp.tools:
                    app.state.mcp_sessions[t.name] = session
                    app.state.mcp_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.inputSchema
                        }
                    })
                logger.info(f"[MCP] 成功挂载节点 {srv_name}，拉取工具数: {len(tools_resp.tools)}")
            except Exception as e:
                logger.error(f"[MCP] 连接节点 {srv_name} ({url}) 失败，已降级跳过: {e}")
                
        logger.info("✅ 全局长连接池建立完毕")
        logger.info("   文档地址: http://localhost:8000/docs")
        yield  # 让 FastAPI 开始处理请求
    finally:
        # 2. 停机时优雅释放资源
        logger.info("🛑 正在关闭应用，断开所有 MCP 长连接...")
        await app.state.mcp_stack.aclose()
        app.state.mcp_sessions.clear()
        logger.info("✅ 资源清理完毕")

app = FastAPI(
    title="Enterprise Knowledge Brain",
    description=(
        "GraphRAG + 多 Agent 协同的企业知识决策系统。\n\n"
        "核心能力:\n"
        "- **POST /index** — 离线建库（PDF → Neo4j 图谱 + Milvus 向量）\n"
        "- **POST /query** — GraphRAG Agent 问答（查询重写 → 路由 → 双路检索 → 合成）\n"
        "- **GET  /graph/entities** — 查询实体节点\n"
        "- **GET  /graph/relations** — 查询关系三元组\n"
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
