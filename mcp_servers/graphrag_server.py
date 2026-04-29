"""
mcp_servers/graphrag_server.py
────────────────────────────────────────────────────────────────────────────
MCP 架构第二期：GraphRAG 核心检索 MCP Server (基于 SSE)
提供跨平台的高级知识智库工具。利用 asyncio.to_thread 包装底层复杂的双路混合检索引擎。
"""
import sys
import os
import asyncio
from typing import Any, Optional
from contextlib import asynccontextmanager

# 将根目录添加到环境变量，以使包导入正常工作
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from openai import OpenAI
from neo4j import GraphDatabase, Driver
from pymilvus import MilvusClient

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from langsmith.wrappers import wrap_openai

from core.logger import get_logger
from core.config import settings
from core.registry import AgentCard, RegistryClient
from agents.graphrag_agent import run_agent

logger = get_logger("mcp_graphrag_server")

# ============================================================================
# § 0  全局单例客户端缓存池（防止反复启动引发连接耗尽）
# ============================================================================
_llm_client: Optional[OpenAI] = None
_neo4j_driver: Optional[Driver] = None
_milvus_client: Optional[MilvusClient] = None

@asynccontextmanager
async def lifespan(app: Starlette):
    """Starlette 生命周期管理器：启动时建连，结束时释放资源"""
    global _llm_client, _neo4j_driver, _milvus_client
    logger.info("正在初始化 GraphRAG 依赖的各大数据库连接池...")
    
    # 1. 初始化 LLM
    _llm_client = OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
    if settings.langchain_tracing_v2:
        _llm_client = wrap_openai(_llm_client)
        
    # 2. 初始化 Neo4j
    _neo4j_driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    
    # 3. 初始化 Milvus
    _milvus_client = MilvusClient(
        uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
    )
    
    logger.info("GraphRAG 数据库连接池初始化完成！")
    
    # --- AgentCard 注册与心跳 ---
    graphrag_agent_card = AgentCard(
        agent_id="graphrag_worker_01",
        agent_name="GraphRAG_Worker",
        description="负责查询复杂的企业知识图谱、政策法规文档、非结构化文本。",
        endpoint=settings.mcp_graphrag_server_url,
        capabilities=["graphrag", "knowledge_base", "unstructured", "policy"],
        mcp_tools_summary=[{"name": "ask_enterprise_knowledge_base", "description": "查询企业内部知识智库"}]
    )
    await RegistryClient.register(graphrag_agent_card, ttl=30)
    
    async def heartbeat_loop():
        while True:
            await asyncio.sleep(15)
            await RegistryClient.heartbeat("graphrag_worker_01", ttl=30)
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    
    yield
    
    # 清理阶段
    heartbeat_task.cancel()
    await RegistryClient.unregister("graphrag_worker_01")
    logger.info("正在关闭 Neo4j 数据库连接池...")
    if _neo4j_driver:
        _neo4j_driver.close()
    if _milvus_client:
        _milvus_client.close()

# ============================================================================
# § 1  显式定义 Schema (满足代码约束 5)
# ============================================================================
class GraphRAGQuerySchema(BaseModel):
    question: str = Field(..., description="需要向企业图谱智库查询的自然语言问题")

# ============================================================================
# § 2  初始化 MCP Server 并注册 Tools (满足代码约束 2, 6)
# ============================================================================
mcp_server = Server("graphrag-knowledge-server")

@mcp_server.tool()
async def ask_enterprise_knowledge_base(question: str) -> str:
    """查询企业内部知识智库。背后会使用图谱和向量双引擎进行检索，融合出最忠实可靠的回答。"""
    logger.info(f"MCP Tool 被触发: 提问 -> {question}")
    
    try:
        # 使用 asyncio.to_thread 开启线程管控，将原本的同步运行逻辑推到后台执行
        # 传入之前生命周期生成的单例句柄，防止底层触发 run_agent 的重连泄漏 Bug
        result = await asyncio.to_thread(
            run_agent,
            question=question,
            llm_client=_llm_client,
            neo4j_driver=_neo4j_driver,
            milvus_client=_milvus_client
        )
        
        # 结果包装（只暴露回答本体和错误，对上游屏蔽内部复杂的跳数和相似度匹配逻辑）
        if result.get("error"):
            return f"检索遭遇异常阻断: {result['error']}"
        
        return result.get("answer", "内部智库未返回任何可用信息。")
        
    except Exception as e:
        logger.error(f"GraphRAG Execution Error: {e}")
        return f"企业智库发生崩溃错误: {e}"


# ============================================================================
# § 3  Streamable HTTP 传输层与 Starlette 路由绑定
# ============================================================================
session_manager = StreamableHTTPSessionManager(mcp_server)

async def handle_mcp(request: Any):
    """处理 MCP Client 的流式请求"""
    await session_manager.handle_request(request.scope, request.receive, request._send)

async def health_check(request: Any):
    return JSONResponse({"status": "ok", "service": "graphrag-mcp"})

# 挂载路由，植入 lifespan 管理器
app = Starlette(
    routes=[
        Route("/health", endpoint=health_check, methods=["GET"]),
        Route("/mcp", endpoint=handle_mcp, methods=["GET", "POST"])
    ],
    lifespan=lifespan
)

if __name__ == "__main__":
    logger.info("启动 GraphRAG MCP Server (Streamable HTTP) 端口: 8003")
    uvicorn.run(app, host="0.0.0.0", port=8003, workers=1)
