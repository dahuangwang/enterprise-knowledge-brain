"""
api/routes.py
────────────────────────────────────────────────────────────────────────────
【新增文件】FastAPI 路由定义

端点清单：
  POST /index               — 触发离线建库（后台任务，立即返回 202）
  POST /query               — 调用 GraphRAG Agent 回答用户问题
  GET  /graph/entities      — 查询 Neo4j 实体节点列表
  GET  /graph/relations     — 查询 Neo4j 关系列表（支持按类型过滤）
  GET  /health              — 健康检查

原则：
  - 路由层只做参数校验和依赖组装，业务逻辑不写在此
  - Prompt 全部在 agents/prompts.py 和 graphrag/prompts.py，不硬编码在路由里
  - Agent / 数据库客户端通过 FastAPI Depends 注入（可替换/可 mock）
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from pydantic import BaseModel, Field

from core.config import settings
from core.logger import get_logger
from utils.event_publisher import publish_query_audit_event

logger = get_logger(__name__)

router = APIRouter()


# ============================================================================
# § 1  请求 / 响应 Schema（显式定义，类型安全）
# ============================================================================

class IndexRequest(BaseModel):
    """建库请求"""
    pdf_path: str = Field(..., description="PDF 文件路径（服务器端路径）")
    dry_run: bool = Field(False, description="True 时只打印抽取结果，不写入数据库")


class IndexResponse(BaseModel):
    """建库响应（异步后台任务，立即返回）"""
    message: str
    pdf_path: str
    dry_run: bool


class QueryRequest(BaseModel):
    """Agent 查询请求"""
    question: str = Field(
        ...,
        description="用户自然语言问题",
        min_length=2,
        max_length=1000,
    )


class QueryResponse(BaseModel):
    """Agent 查询响应"""
    question: str
    normalized: str          # 标准化查询
    entities: List[str]      # 识别出的实体
    route: str               # 选用的检索策略
    route_reason: str        # 路由理由
    answer: str              # 最终回答
    vector_hits: int         # 向量检索命中数
    graph_stats: str         # 图谱检索统计
    elapsed_ms: int          # 总耗时（毫秒）
    error: Optional[str]     # 如有错误


class EntityItem(BaseModel):
    """Neo4j 实体节点"""
    name: str
    entity_type: str
    description: Optional[str] = None
    source_doc: Optional[str] = None


class RelationItem(BaseModel):
    """Neo4j 关系"""
    source: str
    target: str
    relation_type: str
    evidence: Optional[str] = None
    amount: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    neo4j: str
    milvus: str

class ReportGenerateRequest(BaseModel):
    topic: str
    requirements: Optional[str] = None

class ReportGenerateResponse(BaseModel):
    task_id: str
    status: str
    message: str

class OutlineApproveRequest(BaseModel):
    approved_outline: List[Dict[str, Any]]

# ============================================================================
# § 2  依赖项工厂（可替换为其他实现）
# ============================================================================

def _get_neo4j_driver():
    """获取 Neo4j driver（FastAPI Depends 可注入 mock）"""
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def _get_milvus_client():
    """获取 Milvus client（FastAPI Depends 可注入 mock）"""
    from pymilvus import MilvusClient
    return MilvusClient(
        uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
    )


def _get_llm_client():
    """获取 DeepSeek V3 LLM client"""
    from openai import OpenAI
    return OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )


# ============================================================================
# § 3  后台任务函数
# ============================================================================

def _run_indexing_task(pdf_path: str, dry_run: bool) -> None:
    """在 FastAPI BackgroundTask 中执行建库流水线"""
    try:
        from graphrag.indexer import run_indexing_pipeline
        result = run_indexing_pipeline(pdf_path, dry_run=dry_run)
        logger.info(f"建库后台任务完成: {result}")
    except Exception as e:
        logger.error(f"建库后台任务失败: {type(e).__name__}: {e}")


# ============================================================================
# § 4  路由定义
# ============================================================================

@router.get("/health", response_model=HealthResponse, tags=["系统"])
def health_check() -> HealthResponse:
    """健康检查：验证 Neo4j 和 Milvus 连接状态"""
    neo4j_status = "ok"
    milvus_status = "ok"

    try:
        driver = _get_neo4j_driver()
        driver.verify_connectivity()
        driver.close()
    except Exception as e:
        neo4j_status = f"error: {e}"

    try:
        client = _get_milvus_client()
        _ = client.list_collections()
    except Exception as e:
        milvus_status = f"error: {e}"

    overall = "ok" if neo4j_status == "ok" and milvus_status == "ok" else "degraded"
    return HealthResponse(status=overall, neo4j=neo4j_status, milvus=milvus_status)


@router.post("/index", response_model=IndexResponse, status_code=202, tags=["建库"])
def trigger_index(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
) -> IndexResponse:
    """
    触发离线建库（异步后台任务）。
    立即返回 202 Accepted，建库在后台执行，通过日志观察进度。

    风险点:
      - pdf_path 是服务器端路径，需确保文件存在且可读
      - 大型 PDF 建库耗时较长（每 chunk 约 1s LLM 调用）
    """
    import os
    if not os.path.exists(request.pdf_path):
        raise HTTPException(
            status_code=404,
            detail=f"PDF 文件不存在: {request.pdf_path}",
        )

    background_tasks.add_task(_run_indexing_task, request.pdf_path, request.dry_run)
    logger.info(f"建库任务已加入后台队列: {request.pdf_path} (dry_run={request.dry_run})")

    return IndexResponse(
        message="建库任务已启动，请通过日志查看进度",
        pdf_path=request.pdf_path,
        dry_run=request.dry_run,
    )


@router.post("/query", response_model=QueryResponse, tags=["Agent 查询"])
async def query(request_data: QueryRequest, request: Request) -> QueryResponse:
    """
    企业中枢 Agent (Supervisor) 问答接口。

    内部执行链：
      Supervisor (LLM) 路由决策 → 并发调用底层 Worker (RAG/SQL/API) → 回答合成 (LLM)

    风险点:
      - 底层依赖全局建立的 MCP SSE 长连接池
    """
    from agents.planner_agent import _run_planner_async

    # 1. 模拟 OIDC 鉴权：从 Header 中提取已由网关解析好的用户信息
    user_id = request.headers.get("X-OIDC-User", "anonymous_user")

    # 2. 获取全局池化的 MCP 会话字典
    mcp_sessions = getattr(request.app.state, "mcp_sessions", {})
    mcp_tools = getattr(request.app.state, "mcp_tools", [])
    if not mcp_sessions:
        logger.warning("未检测到有效的全局 MCP 长连接池，可能影响工具调用")

    t0 = time.monotonic()
    status = "success"
    try:
        # 改为直接 await 异步调度核心，传入连接池与身份信息
        result: Dict[str, Any] = await _run_planner_async(
            question=request_data.question,
            mcp_sessions=mcp_sessions,
            mcp_tools=mcp_tools,
            user_id=user_id
        )
    except Exception as e:
        status = "failed"
        logger.error(f"/query 执行失败: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 执行异常: {e}")
    finally:
        # 3. 查询审计事件：demo 记录日志，production 可对接审计网关/Kafka Bridge
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        await publish_query_audit_event(
            user_id=user_id,
            elapsed_ms=elapsed_ms,
            question=request_data.question,
            status=status,
        )

    return QueryResponse(
        question=result.get("question", request_data.question),
        normalized=result.get("normalized", ""),
        entities=result.get("entities", []),
        route=result.get("route", "unknown"),
        route_reason=result.get("route_reason", ""),
        answer=result.get("answer", ""),
        vector_hits=result.get("vector_hits", 0),
        graph_stats=result.get("graph_stats", "无"),
        elapsed_ms=elapsed_ms,
        error=result.get("error"),
    )


@router.get("/graph/entities", response_model=List[EntityItem], tags=["图谱查询"])
def get_entities(
    entity_type: Optional[str] = Query(None, description="按实体类型过滤，如 '公司'"),
    limit: int = Query(50, ge=1, le=500, description="返回最大条数"),
) -> List[EntityItem]:
    """查询 Neo4j 实体节点列表，支持按类型过滤"""
    driver = _get_neo4j_driver()
    try:
        with driver.session() as session:
            if entity_type:
                records = session.run(
                    """
                    MATCH (e:Entity {entity_type: $entity_type})
                    RETURN e.name AS name, e.entity_type AS entity_type,
                           e.description AS description, e.source_doc AS source_doc
                    LIMIT $limit
                    """,
                    entity_type=entity_type,
                    limit=limit,
                )
            else:
                records = session.run(
                    """
                    MATCH (e:Entity)
                    RETURN e.name AS name, e.entity_type AS entity_type,
                           e.description AS description, e.source_doc AS source_doc
                    LIMIT $limit
                    """,
                    limit=limit,
                )
            return [
                EntityItem(
                    name=r["name"],
                    entity_type=r["entity_type"],
                    description=r["description"],
                    source_doc=r["source_doc"],
                )
                for r in records
            ]
    except Exception as e:
        logger.error(f"/graph/entities 查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        driver.close()


@router.get("/graph/relations", response_model=List[RelationItem], tags=["图谱查询"])
def get_relations(
    relation_type: Optional[str] = Query(None, description="按关系类型过滤，如 '投资'"),
    entity_name: Optional[str] = Query(None, description="按实体名称过滤（模糊匹配）"),
    limit: int = Query(50, ge=1, le=500, description="返回最大条数"),
) -> List[RelationItem]:
    """查询 Neo4j 关系列表，支持按关系类型或实体名称过滤"""
    driver = _get_neo4j_driver()
    try:
        with driver.session() as session:
            where_clauses = []
            params: Dict[str, Any] = {"limit": limit}

            if relation_type:
                where_clauses.append("r.relation_type = $relation_type")
                params["relation_type"] = relation_type
            if entity_name:
                where_clauses.append(
                    "(s.name CONTAINS $entity_name OR t.name CONTAINS $entity_name)"
                )
                params["entity_name"] = entity_name

            where_str = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            records = session.run(
                f"""
                MATCH (s:Entity)-[r:RELATION]->(t:Entity)
                {where_str}
                RETURN s.name AS source, t.name AS target,
                       r.relation_type AS relation_type,
                       r.evidence AS evidence, r.amount AS amount
                LIMIT $limit
                """,
                **params,
            )
            return [
                RelationItem(
                    source=r["source"],
                    target=r["target"],
                    relation_type=r["relation_type"],
                    evidence=r["evidence"],
                    amount=r["amount"],
                )
                for r in records
            ]
    except Exception as e:
        logger.error(f"/graph/relations 查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        driver.close()

# ============================================================================
# § 5  长报告生成 API (异步 & 人机协同)
# ============================================================================

@router.post("/report/generate", response_model=ReportGenerateResponse, tags=["报告生成"])
async def trigger_report_generate(
    request_data: ReportGenerateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> ReportGenerateResponse:
    import uuid
    from agents.report_agent import start_report_task
    
    task_id = f"rep-{uuid.uuid4().hex[:8]}"
    mcp_sessions = getattr(request.app.state, "mcp_sessions", {})
    mcp_tools = getattr(request.app.state, "mcp_tools", [])
    
    # 异步触发 LangGraph 报告生成任务
    background_tasks.add_task(
        start_report_task, 
        task_id, 
        request_data.topic, 
        request_data.requirements or "",
        mcp_sessions,
        mcp_tools
    )
    
    return ReportGenerateResponse(
        task_id=task_id,
        status="pending",
        message="长报告生成任务已启动，请轮询大纲状态"
    )

@router.get("/report/{task_id}/status", tags=["报告生成"])
async def get_report_status(task_id: str):
    from agents.report_agent import get_report_graph
    graph = await get_report_graph()
    config = {"configurable": {"thread_id": task_id}}
    
    try:
        state_snap = await graph.aget_state(config)
        if not state_snap or not state_snap.values:
            return {"task_id": task_id, "status": "not_found"}
            
        values = state_snap.values
        # 判断状态
        if not values.get("outline"):
            status = "planning_outline"
        elif not values.get("outline_approved"):
            status = "waiting_approval"
        elif not values.get("final_report"):
            status = "drafting_sections"
        else:
            status = "completed"
            
        return {
            "task_id": task_id,
            "status": status,
            "outline": values.get("outline", []),
            "next": state_snap.next # LangGraph 挂起的下一个节点
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/report/{task_id}/outline/approve", tags=["报告生成"])
async def approve_report_outline(task_id: str, request_data: OutlineApproveRequest, background_tasks: BackgroundTasks):
    from agents.report_agent import get_report_graph
    from langgraph.types import Command
    
    graph = await get_report_graph()
    config = {"configurable": {"thread_id": task_id}}
    
    try:
        state_snap = await graph.aget_state(config)
        if not state_snap or not state_snap.values:
            raise HTTPException(status_code=404, detail="Task not found")
            
        if state_snap.values.get("outline_approved"):
            return {"message": "Outline already approved"}
            
        # 恢复图执行
        async def _resume():
            await graph.ainvoke(
                Command(resume=True, update={"outline": request_data.approved_outline, "outline_approved": True}),
                config=config
            )
            
        background_tasks.add_task(_resume)
        
        return {"message": "大纲已确认，恢复报告起草任务"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report/{task_id}/download", tags=["报告生成"])
async def download_report(task_id: str, format: str = "pdf"):
    import os
    from fastapi.responses import FileResponse
    from agents.report_agent import get_report_graph
    from utils.pdf_exporter import markdown_to_pdf
    
    graph = await get_report_graph()
    config = {"configurable": {"thread_id": task_id}}
    
    state_snap = await graph.aget_state(config)
    if not state_snap or not state_snap.values:
        raise HTTPException(status_code=404, detail="Task not found")
        
    final_report = state_snap.values.get("final_report")
    if not final_report:
        raise HTTPException(status_code=400, detail="Report is not completed yet")
        
    # 保存并导出
    output_dir = "output/reports"
    os.makedirs(output_dir, exist_ok=True)
    
    if format == "markdown" or format == "md":
        file_path = os.path.join(output_dir, f"{task_id}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        return FileResponse(file_path, filename=f"{state_snap.values.get('topic', task_id)}.md")
        
    elif format == "pdf":
        pdf_path = os.path.join(output_dir, f"{task_id}.pdf")
        if not os.path.exists(pdf_path):
            success = markdown_to_pdf(final_report, pdf_path)
            if not success:
                raise HTTPException(status_code=500, detail="PDF generation failed")
        return FileResponse(pdf_path, filename=f"{state_snap.values.get('topic', task_id)}.pdf")
        
    raise HTTPException(status_code=400, detail="Unsupported format")

# ============================================================================
# § 6  A2A 架构与 Registry (AgentCard) API
# ============================================================================
from core.registry import RegistryClient, AgentCard

@router.get("/agents", response_model=List[AgentCard], tags=["A2A 服务发现"])
async def get_active_agents():
    """获取当前所有通过心跳机制注册的活跃 AgentCard 列表"""
    try:
        agents = await RegistryClient.discover()
        return agents
    except Exception as e:
        logger.error(f"/agents 查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
