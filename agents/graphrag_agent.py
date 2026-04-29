"""
agents/graphrag_agent.py
────────────────────────────────────────────────────────────────────────────
【新增文件】基于 LangGraph 的主控 GraphRAG Agent（第二阶段 MVP）

功能：
  当用户提问时，Agent 通过以下三步完成查询：
  ① 查询重写  — LLM 将口语化问题标准化，提取关键实体，描述意图
  ② 路由决策  — LLM 判断检索策略（向量/图谱/混合）
  ③ 检索执行  — 根据路由调用 Milvus / Neo4j / 混合检索
  ④ 回答合成  — LLM 基于检索结果生成忠实、可溯源的回答

LangGraph 状态机：
  START
    └─► rewrite_query
          └─► route_query
                └─► [条件分支]
                      ├─► vector_search_node ──┐
                      ├─► graph_search_node  ──┤─► synthesize_node ─► END
                      └─► hybrid_search_node ──┘

设计原则：
  - 所有外部依赖（LLM client / Neo4j driver / Milvus client）通过参数注入
  - 测试时可传入 mock，无需真实数据库
  - Prompt 全部来自 agents/prompts.py，此文件只含流程逻辑
  - 单次查询失败（LLM/DB 异常）不崩溃，error 字段记录原因
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Literal, Optional

from openai import OpenAI
from neo4j import Driver
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict
from langsmith import traceable
from langsmith.wrappers import wrap_openai

from core.config import settings
from core.logger import get_logger
from agents.prompts import (
    NormalizedQuery,
    RoutingDecision,
    SynthesisInput,
    RouteType,
    build_rewrite_prompt,
    build_route_prompt,
    build_synthesis_prompt,
    QUERY_REWRITE_SYSTEM,
    ROUTE_SYSTEM,
    SYNTHESIS_SYSTEM,
)
from graphrag.graph_search import run_graph_search, GraphSearchResult

logger = get_logger(__name__)


# ============================================================================
# § 1  LangGraph 状态定义（显式字段，类型安全）
# ============================================================================

# 继承TypedDict，total=False表示字段可以缺省
# 继承TypedDict的作用是把类变成状态机
class AgentState(TypedDict, total=False):
    """
    LangGraph 状态字典。
    每个节点只更新自己负责的字字段。
    total=False 允许字段缺省（初始段，不触碰其他化时不需要提供所有字段）。
    """

    # ── 输入 ────────────────────────────────────────────────────────────────
    question: str                          # 原始用户问题（必填）

    # ── 中间状态 ─────────────────────────────────────────────────────────────
    normalized: Optional[NormalizedQuery]  # 查询重写结果
    routing: Optional[RoutingDecision]     # 路由决策结果

    # ── 检索结果 ─────────────────────────────────────────────────────────────
    vector_results: List[Dict[str, Any]]   # Milvus 检索命中列表
    graph_result: Optional[GraphSearchResult]  # Neo4j 图谱检索结果

    # ── 输出 ────────────────────────────────────────────────────────────────
    answer: str                            # 最终回答
    error: Optional[str]                   # 错误信息（不为 None 表示某步骤失败）


# ============================================================================
# § 2  依赖项 —— Embedding 模型懒加载单例（线程安全）
# ============================================================================

_embedding_model: Optional[SentenceTransformer] = None
_embedding_model_lock = threading.Lock()   # 保护懒加载，防止并发重复加载


def _get_embedding_model() -> SentenceTransformer:
    """懒加载 bge-m3，进程生命周期内只加载一次（约 2.3GB）。

    使用 double-checked locking 保证线程安全：
    - 外层无锁判断：热路径（模型已加载）零竞争开销
    - 内层持锁再判断：防止多个线程同时通过外层 None 检查后重复加载
    """
    # global是全局变量，用于声明函数内部使用的变量是全局变量
    # 属性中有_embedding_model，在此设置全局变量就是为了修改属性，此方法就是给属性赋值
    global _embedding_model
    if _embedding_model is None:                          # 外层：热路径无锁
        with _embedding_model_lock:                       # 内层：持锁保护
            if _embedding_model is None:                  # 二次检查防竞争
                logger.info(f"首次加载 Embedding 模型: {settings.embedding_model}")
                _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


# ============================================================================
# § 3  LLM 调用辅助函数（复用 DeepSeek V3 JSON mode）
# ============================================================================

@traceable(name="LLM_JSON_Call")
def _llm_json_call(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_retries: int = 2,
) -> Optional[dict]:
    """
    向 DeepSeek V3 发起 JSON mode 调用，返回解析后的 dict。
    失败时指数退避重试，全部失败返回 None（不抛异常，由调用方处理）。
    """
    # 在此处指数退避重试的最大重试次数是3次
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=settings.deepseek_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                timeout=30,
            )
            raw: str = response.choices[0].message.content
            # 将 JSON 字符串解析为 Python 字典
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM JSON 解析失败 (尝试 {attempt + 1}): {e}")
        except Exception as e:
            logger.warning(
                f"LLM 调用失败 (尝试 {attempt + 1}/{max_retries + 1}): "
                f"{type(e).__name__}: {e}"
            )
        if attempt < max_retries:
            time.sleep(2 ** attempt)

    logger.error("LLM 调用最终失败，返回 None")
    return None


# ============================================================================
# § 4  Milvus 向量检索辅助函数
# ============================================================================

def _search_milvus(
    client: MilvusClient,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    对 Milvus 执行向量余弦相似度检索。

    流程: query_text → bge-m3 embed → COSINE search → 格式化结果

    Returns:
        List of {"score": float, "chunk_text": str, "page_num": int,
                 "doc_id": str, "entities_json": str}

    风险点:
        - MilvusClient 搜索要求 collection 已 load，ensure_milvus_collection()
          在 indexer 中已处理，但 Agent 启动时需确认 collection 存在
    """
    try:
        # 懒加载 bge-m3 模型
        model = _get_embedding_model()
        # normalize_embeddings参数是True，表示对向量进行归一化，与False的区别是
        # False：不归一化，直接计算余弦相似度
        
        query_vector: List[float] = (
            model.encode([query], normalize_embeddings=True)[0].tolist()
        )

        # 客户端搜索,没加搜索方式，默认是向量相似度搜索
        # 如果要加搜索方式格式如下：search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
        raw_results = client.search(
            collection_name=settings.milvus_collection,
            data=[query_vector],
            limit=top_k,
            output_fields=["chunk_text", "entities_json", "page_num", "doc_id"],
        )
        # raw_results输出格式为List[List[Dict[str, Any]]]
        hits: List[Dict[str, Any]] = []
        for result_group in raw_results:
            for hit in result_group:
                entity = hit.get("entity", {})
                hits.append(
                    {
                        "score":        round(float(hit.get("distance", 0)), 4),
                        "chunk_text":   entity.get("chunk_text", ""),
                        "page_num":     entity.get("page_num", 0),
                        "doc_id":       entity.get("doc_id", ""),
                        "entities_json": entity.get("entities_json", "[]"),
                    }
                )
        logger.info(f"Milvus 向量检索完成: {len(hits)} 条结果")
        return hits
    except Exception as e:
        logger.error(f"Milvus 检索失败: {type(e).__name__}: {e}")
        return []


def _format_vector_results(results: List[Dict[str, Any]]) -> str:
    """将 Milvus 检索结果格式化为 LLM 可读文本"""
    if not results:
        return "（未找到相关向量检索内容）"
    lines: List[str] = []
    for i, r in enumerate(results, 1):
        snippet = r["chunk_text"][:400].replace("\n", " ")
        lines.append(
            f"[{i}] 相似度={r['score']:.3f}，第 {r['page_num']} 页\n"
            f"    {snippet}..."
        )
    return "\n".join(lines)


# ============================================================================
# § 5  LangGraph 节点函数（每个节点返回对当前状态的局部更新）
# ============================================================================

def rewrite_query_node(
    state: AgentState,
    *,
    llm_client: OpenAI,
) -> dict:
    """
    节点①：查询重写节点
    输入:  state["question"]
    输出:  state["normalized"] | state["error"]
    """
    question = state.get("question", "")
    logger.info(f"[rewrite] 原始问题: {question!r}")
    # 查询改写节点，将原始问题进行改写，返回改写后的问题和实体
    # 返回： 
    # 1.简称补全为正式全称（语义统一） 
    # 2.实体消歧（明确指向图谱中的哪个实体） 
    # 3.查询意确图识别（明用户想做什么）
    # 4.查询改写（将模糊查询改写为精确查询）
    # 其他：幻觉抵制：防止模型乱问乱答，强制输出JSON约束
    data = _llm_json_call(
        client=llm_client,
        system_prompt=QUERY_REWRITE_SYSTEM,
        user_prompt=build_rewrite_prompt(question), 
        temperature=settings.agent_temperature,
    )
    # data{
    #   {
    #     "normalized": "标准化后的查询语句",
    #     "entities": ["实体名称1", "实体名称2"],
    #     "intent": "用户意图一句话描述"
    #   }
    # }
    if data is None:
        return {"error": "查询重写失败：LLM 调用异常"}

    try:
        # 将字典转换为 Pydantic 模型实例，**是指解包字典，将字典的键值对作为参数传递给模型

        # 原始问题：告知我当前和北京传智播客教育科技有限公司有合作的公司有哪些？
        # 标准化: 告知我当前和北京传智播客教育科技有限公司有合作的公司有哪些？
        # 实体： ['北京传智播客教育科技有限公司']
        normalized = NormalizedQuery(**data)
        logger.info(
            f"[rewrite] 标准化: {normalized.normalized!r} | "
            f"实体: {normalized.entities}"
        )
        return {"normalized": normalized, "error": None}
    except Exception as e:
        logger.error(f"[rewrite] Schema 解析失败: {e}")
        return {"error": f"查询重写 Schema 解析失败: {e}"}


def route_query_node(
    state: AgentState,
    *,
    llm_client: OpenAI,
) -> dict:
    """
    节点②：路由决策节点
    输入:  state["normalized"]
    输出:  state["routing"] | state["error"]
    """
    if state.get("error"):
        return {}   # 已有错误，跳过，让条件边路由到 synthesize

    # normalized格式：NormalizedQuery(
    #   normalized='告知我当前和北京传智播客教育科技有限公司有合作的公司有哪些？',
    #   entities=['北京传智播客教育科技有限公司'],
    #   intent='查询与北京传智播客教育科技有限公司有合作的公司')
    normalized: Optional[NormalizedQuery] = state.get("normalized")
    if normalized is None:
        return {"error": "路由决策失败：normalized 为空"}
    # 调用LLM进行路由决策
    # data格式：RoutingDecision(
    #   route="vector_search | graph_search | hybrid_search",
    #   reasoning="选择该策略的具体理由（不超过100字）"
    # )
    data = _llm_json_call(
        client=llm_client,
        system_prompt=ROUTE_SYSTEM,
        user_prompt=build_route_prompt(normalized),
        temperature=settings.agent_temperature,
    )
    if data is None:
        logger.warning("[route] LLM 失败，降级到 hybrid_search")
        return {
            "routing": RoutingDecision(
                route="hybrid_search", reasoning="LLM 路由失败，使用默认混合检索"
            )
        }

    try:
        routing = RoutingDecision(**data)
        logger.info(f"[route] 决策: {routing.route}  原因: {routing.reasoning}")
        return {"routing": routing}
    except Exception as e:
        logger.error(f"[route] Schema 解析失败: {e}，降级到 hybrid_search")
        return {
            "routing": RoutingDecision(
                route="hybrid_search", reasoning=f"路由解析失败: {e}"
            )
        }


def vector_search_node(
    state: AgentState,
    *,
    milvus_client: MilvusClient,
) -> dict:
    """
    节点③-A：Milvus 向量检索节点
    输入:  state["normalized"]
    输出:  state["vector_results"]
    """
    normalized: Optional[NormalizedQuery] = state.get("normalized")
    # 如果normalized不为空，则使用normalized中的normalized字段，否则使用question字段
    # 这样做的目的是防止查询重写出错导致程序中断
    query = normalized.normalized if normalized else state.get("question", "")
    # 通过余弦相似度的方式检索milvus数据库
    results = _search_milvus(milvus_client, query, top_k=settings.retrieval_top_k)
    return {"vector_results": results, "graph_result": GraphSearchResult()}


def graph_search_node(
    state: AgentState,
    *,
    neo4j_driver: Driver,
) -> dict:
    """
    节点③-B：Neo4j 图谱检索节点
    输入:  state["normalized"]
    输出:  state["graph_result"]
    """
    normalized: Optional[NormalizedQuery] = state.get("normalized")
    keywords = (
        normalized.entities if normalized and normalized.entities
        else ([state.get("question", "")][:1])
    )
    # 运行图谱检索
    # 通过模糊匹配实体 → N-hop 邻域展开来检索图谱信息
    graph_result = run_graph_search(
        driver=neo4j_driver,
        keywords=keywords,
        max_depth=settings.graph_hop_depth,
        limit=50,
    )
    return {"vector_results": [], "graph_result": graph_result}


def hybrid_search_node(
    state: AgentState,
    *,
    milvus_client: MilvusClient,
    neo4j_driver: Driver,
) -> dict:
    """
    节点③-C：混合检索（向量 + 图谱）
    输入:  state["normalized"]
    输出:  state["vector_results"] + state["graph_result"]
    """
    normalized: Optional[NormalizedQuery] = state.get("normalized")
    query = normalized.normalized if normalized else state.get("question", "")
    keywords = (
        normalized.entities if normalized and normalized.entities
        else ([query])
    )

    # 并行执行（此处串行实现，后续可改为 asyncio.gather）
    vector_results = _search_milvus(
        milvus_client, query, top_k=settings.retrieval_top_k
    )
    graph_result = run_graph_search(
        driver=neo4j_driver,
        keywords=keywords,
        max_depth=settings.graph_hop_depth,
        limit=50,
    )
    return {"vector_results": vector_results, "graph_result": graph_result}


def synthesize_node(
    state: AgentState,
    *,
    llm_client: OpenAI,
) -> dict:
    """
    节点④：回答合成节点
    输入:  state["question"] + state["vector_results"] + state["graph_result"]
    输出:  state["answer"]

    若 state["error"] 不为空，直接返回错误提示，不调用 LLM。
    """
    # ── 错误短路 ──────────────────────────────────────────────────────────────
    if state.get("error"):
        error_msg = state["error"]
        logger.error(f"[synthesize] 因上游错误跳过 LLM 合成: {error_msg}")
        return {
            "answer": f"系统处理异常，无法完成回答。\n原因: {error_msg}"
        }

    question = state.get("question", "")
    vector_results: List[Dict] = state.get("vector_results", [])
    graph_result: Optional[GraphSearchResult] = state.get("graph_result")
    # 格式化向量检索结果,变为llm可以认识的格式
    vector_context = _format_vector_results(vector_results)
    # 格式化图谱检索结果,变为llm可以认识的格式
    graph_context = (
        graph_result.to_text()
        if graph_result and not graph_result.is_empty
        else "（未执行图谱检索或无结果）"
    )

    logger.info(
        f"[synthesize] 向量命中={len(vector_results)} 条, "
        f"图谱={graph_result.stats() if graph_result else '无'}"
    )
    # 格式化输入，将向量检索结果和图谱检索结果合并
    inp = SynthesisInput(
        question=question,
        vector_context=vector_context,
        graph_context=graph_context,
    )

    # ── LLM 生成回答 ─────────────────────────────────────────────────────────
    try:
        response = llm_client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {"role": "user",   "content": build_synthesis_prompt(inp)},
            ],
            temperature=settings.synthesis_temperature,
            timeout=60,
        )
        answer: str = response.choices[0].message.content.strip()
        logger.info(f"[synthesize] 回答生成完成（{len(answer)} 字符）")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"[synthesize] LLM 合成失败: {e}")
        return {
            "answer": (
                "回答合成失败，请稍后重试。\n"
                f"检索摘要 — 向量: {len(vector_results)} 条"
                f"，图谱: {graph_result.stats() if graph_result else '无'}"
            )
        }


# ============================================================================
# § 6  路由条件函数（决定 route_query 之后走哪个分支）
# ============================================================================

def _route_condition(state: AgentState) -> str:
    """
    条件边判断函数，返回值必须与 add_conditional_edges 中的映射 key 一致。

    - 如果上游有错误，直接跳到 synthesize（错误短路）
    - 否则按 RoutingDecision.route 分支
    """
    if state.get("error"):
        return "synthesize"

    routing: Optional[RoutingDecision] = state.get("routing")
    if routing is None:
        return "hybrid_search"   # 安全默认值

    return routing.route  # "vector_search" | "graph_search" | "hybrid_search"


# ============================================================================
# § 7  LangGraph 图构建（工厂函数，依赖通过闭包注入）
# ============================================================================

def build_graphrag_agent(
    llm_client: OpenAI,
    neo4j_driver: Driver,
    milvus_client: MilvusClient,
):
    """
    构建并编译 LangGraph GraphRAG Agent。

    所有外部依赖（LLM / Neo4j / Milvus）通过参数传入，
    通过 functools.partial 绑定到节点函数，实现依赖注入。

    Args:
        llm_client:    DeepSeek V3 OpenAI 兼容客户端
        neo4j_driver:  Neo4j driver（已验证连接）
        milvus_client: MilvusClient（已连接）

    Returns:
        编译后的 LangGraph（可直接调用 .invoke(state)）
    """
    import functools

    # ── 绑定依赖到各节点函数 ──────────────────────────────────────────────────

    _rewrite  = functools.partial(rewrite_query_node,  llm_client=llm_client)
    _route    = functools.partial(route_query_node,    llm_client=llm_client)
    _vector   = functools.partial(vector_search_node,  milvus_client=milvus_client)
    _graph    = functools.partial(graph_search_node,   neo4j_driver=neo4j_driver)
    _hybrid   = functools.partial(
        hybrid_search_node,
        milvus_client=milvus_client,
        neo4j_driver=neo4j_driver,
    )
    _synth    = functools.partial(synthesize_node,     llm_client=llm_client)

    # ── 构建状态图 ────────────────────────────────────────────────────────────
    # 1. 实例化 StateGraph，并传入我们之前定义的 AgentState 类型。这告诉 LangGraph：
    #    “这个图的运行状态（State）应该包含哪些字段。”
    graph = StateGraph(AgentState)

    graph.add_node("rewrite_query",      _rewrite)
    graph.add_node("route_query",        _route)
    graph.add_node("vector_search",      _vector)
    graph.add_node("graph_search",       _graph)
    graph.add_node("hybrid_search",      _hybrid)
    graph.add_node("synthesize",         _synth)

    # ── 固定边 ───────────────────────────────────────────────────────────────
    # 2. 添加边（Edges）：定义节点之间的流向。
    # add_edge定义确定的、必然发生的步骤走向。
    graph.add_edge(START,            "rewrite_query")
    graph.add_edge("rewrite_query",  "route_query")

    # ── 条件边（路由决策后分支）─────────────────────────────────────────────
    # 3. 添加条件边（Conditional Edges）：这是实现“智能路由”的关键。
    # 定义基于不同条件产生分岔的步骤走向，相当于代码中的 switch / if-else 分支。
    # 在 "route_query" 节点执行后，图引擎会调用 _route_condition 这个条件判断方法。
    # 引擎将查看 _route_condition 返回的字符串指引，并根据后面提供的大括号字典映射，匹配到具体的下一个节点。
    # 这赋予了智能体“思考并决定策略”的能力。系统会根据用户的初始输入，在这个地方动态决定是走单纯查向量库 (vector_search)、
    # 单纯查图谱 (graph_search)、两者都查 (hybrid_search)，或者是如果发生了错误甚至发现无需检索时短路直达回答器 (synthesize)。
    graph.add_conditional_edges(
        "route_query",
        _route_condition,
        {
            "vector_search": "vector_search",
            "graph_search":  "graph_search",
            "hybrid_search": "hybrid_search",
            "synthesize":    "synthesize",   # 错误短路
        },
    )

    # ── 各检索节点汇聚到 synthesize ──────────────────────────────────────────
    # 管上面第 4 步走入了哪一条检索分支途径，这里规定了它们执行完毕后，统统都固定汇聚走向最后的 "synthesize"（生成答案/合成）节点。
    # 这确保了检索到的任何数据最终都会被整合输送给 LLM 回答。
    graph.add_edge("vector_search",  "synthesize")
    graph.add_edge("graph_search",   "synthesize")
    graph.add_edge("hybrid_search",  "synthesize")
    graph.add_edge("synthesize",     END)
    # 将之前定义的所有节点（Nodes）、常规边（Edges）和条件边组合成一个立即可用的、高度优化的可执行状态机引擎应用。
    # 整个方法的返回值就是这个准备好接收外界请求（通过类似 .invoke() 触发）的智能体对象。
    return graph.compile()


# ============================================================================
# § 8  Public API —— 对外暴露的主入口
# ============================================================================

@traceable(name="GraphRAG_Run_Agent")
def run_agent(
    question: str,
    llm_client: Optional[OpenAI] = None,
    neo4j_driver: Optional[Driver] = None,
    milvus_client: Optional[MilvusClient] = None,
) -> Dict[str, Any]:
    """
    主入口：接受用户问题，返回完整 Agent 执行结果。

    Args:
        question:      用户原始问题（自然语言）
        llm_client:    可注入 mock（测试用），None 时自动创建
        neo4j_driver:  可注入 mock（测试用），None 时自动创建
        milvus_client: 可注入 mock（测试用），None 时自动创建

    Returns:
        {
          "question":    str,         # 原始问题
          "normalized":  str,         # 标准化查询
          "route":       str,         # 选用的检索策略
          "answer":      str,         # 最终回答
          "vector_hits": int,         # 向量命中数量
          "graph_stats": str,         # 图谱统计
          "error":       str | None,  # 错误信息
        }

    风险点:
      - 首次调用会加载 bge-m3 embedding 模型（约 2.3GB），耗时约 10~30s
      - 确保 Neo4j 已建库（run_indexing_pipeline 已运行）
      - 确保 Milvus collection 中有数据
    """
    from openai import OpenAI as _OpenAI
    from neo4j import GraphDatabase
    from pymilvus import MilvusClient as _MilvusClient

    # ── 构建依赖（未注入时创建真实连接）─────────────────────────────────────
    _llm = llm_client or _OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
    if settings.langchain_tracing_v2:
        _llm = wrap_openai(_llm)
        
    _neo4j = neo4j_driver or GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    _milvus = milvus_client or _MilvusClient(
        uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
    )

    # ── 构建并运行 Agent ──────────────────────────────────────────────────────
    # 构建通过自定义封装的 LangGraph 状态机引擎智能体
    agent = build_graphrag_agent(_llm, _neo4j, _milvus)

    initial_state: AgentState = {
        "question": question,
        "normalized": None,
        "routing": None,
        "vector_results": [],
        "graph_result": None,
        "answer": "",
        "error": None,
    }

    logger.info(f"Agent 启动: {question!r}")
    final_state: AgentState = agent.invoke(initial_state)
    logger.info(f"Agent 完成，回答长度: {len(final_state.get('answer', ''))} 字")

    # ── 格式化输出 ────────────────────────────────────────────────────────────
    normalized: Optional[NormalizedQuery] = final_state.get("normalized")
    routing: Optional[RoutingDecision] = final_state.get("routing")
    graph_result = final_state.get("graph_result")

    return {
        "question":    question,
        "normalized":  normalized.normalized if normalized else "",
        "entities":    normalized.entities if normalized else [],
        "route":       routing.route if routing else "unknown",
        "route_reason": routing.reasoning if routing else "",
        "answer":      final_state.get("answer", ""),
        "vector_hits": len(final_state.get("vector_results", [])),
        "graph_stats": graph_result.stats() if graph_result else "无",
        "error":       final_state.get("error"),
    }
