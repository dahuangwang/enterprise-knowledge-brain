"""
graphrag/graph_search.py
────────────────────────────────────────────────────────────────────────────
【新增文件】在线图谱检索：Neo4j 图拓扑遍历

提供三个层次的检索能力：
  1. search_entities_by_name()    — 关键词模糊匹配实体节点
  2. get_entity_neighborhood()    — N-hop 邻域展开（获取实体及其关系）
  3. run_graph_search()           — 主入口：关键词 → 匹配实体 → 展开邻域

设计原则：
  - driver 通过参数注入，测试时可 mock
  - 所有检索结果通过 GraphSearchResult Schema 返回，类型安全
  - to_text() 方法将结果格式化为 LLM 可读文本
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field
from neo4j import Driver

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# § 1  输出 Schema（显式定义，类型安全）
# ============================================================================

class GraphNode(BaseModel):
    """图谱节点 —— 对应 Neo4j 中的 Entity 节点"""

    name: str = Field(..., description="实体名称")
    entity_type: str = Field(..., description="实体类型（公司/项目/人物等）")
    description: Optional[str] = Field(None, description="实体描述")


class GraphEdge(BaseModel):
    """图谱有向边 —— 对应 Neo4j 中的 RELATION 关系"""

    source: str = Field(..., description="起点实体名称")
    target: str = Field(..., description="终点实体名称")
    relation_type: str = Field(..., description="关系类型（投资/持股/子公司等）")
    evidence: Optional[str] = Field(None, description="原文依据")
    amount: Optional[str] = Field(None, description="金额或比例")


class GraphSearchResult(BaseModel):
    """图谱检索的完整结果"""

    nodes: List[GraphNode] = Field(default_factory=list, description="图谱节点列表")
    edges: List[GraphEdge] = Field(default_factory=list, description="图谱关系列表")
    query_entities: List[str] = Field(
        default_factory=list, description="检索时使用的实体关键词"
    )

    @property
    def is_empty(self) -> bool:
        return not self.nodes and not self.edges

    def stats(self) -> str:
        return f"{len(self.nodes)} 节点, {len(self.edges)} 关系"

    def to_text(self) -> str:
        """
        将图谱结果格式化为 LLM 可读的纯文本。
        供 synthesize_node 注入 Prompt 使用。
        """
        if self.is_empty:
            return "（未找到相关图谱信息）"

        lines: List[str] = []

        if self.nodes:
            lines.append("【实体节点】")
            for n in self.nodes:
                desc = f"（{n.description}）" if n.description else ""
                lines.append(f"  · [{n.entity_type}] {n.name}{desc}")

        if self.edges:
            lines.append("【实体关系】")
            for e in self.edges:
                amount_str = f"，金额/比例: {e.amount}" if e.amount else ""
                ev_str = (
                    f"\n      原文依据: {e.evidence[:100]}..."
                    if e.evidence and len(e.evidence) > 100
                    else (f"\n      原文依据: {e.evidence}" if e.evidence else "")
                )
                lines.append(
                    f"  · {e.source} —[{e.relation_type}]→ {e.target}{amount_str}{ev_str}"
                )

        return "\n".join(lines)


# ============================================================================
# § 2  核心检索函数
# ============================================================================

def search_entities_by_name(
    driver: Driver,
    keywords: List[str],
    limit: int = 20,
) -> List[GraphNode]:
    """
    按关键词列表模糊搜索实体节点（Neo4j CONTAINS 匹配，大小写不敏感）。
    多关键词分别搜索后去重合并。

    Args:
        driver:   Neo4j driver（可注入 mock）
        keywords: 关键词列表，来自 NormalizedQuery.entities
        limit:    每个关键词最多返回的节点数

    Returns:
        去重后的 GraphNode 列表

    风险点:
        - CONTAINS 是全表扫描，数据量大时建议建 fulltext index:
          CREATE FULLTEXT INDEX entity_name_idx FOR (e:Entity) ON EACH [e.name]
    """
    if not keywords:
        return []

    seen: set[str] = set()
    results: List[GraphNode] = []

    with driver.session() as session:
        for kw in keywords:
            records = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $kw
                   OR toLower(e.name) CONTAINS toLower($kw)
                RETURN e.name       AS name,
                       e.entity_type AS entity_type,
                       e.description AS description
                LIMIT $limit
                """,
                kw=kw,
                limit=limit,
            )
            for r in records:
                name: str = r["name"]
                if name not in seen:
                    seen.add(name)
                    results.append(
                        GraphNode(
                            name=name,
                            entity_type=r["entity_type"] or "未知",
                            description=r["description"],
                        )
                    )

    logger.info(f"实体名称搜索: {keywords} → {len(results)} 个节点")
    return results


def get_entity_neighborhood(
    driver: Driver,
    entity_names: List[str],
    max_depth: int = 2,
    max_edges: int = 50,
) -> GraphSearchResult:
    """
    以给定实体为起点，展开 N-hop 邻域，返回节点和关系。

    Args:
        driver:       Neo4j driver
        entity_names: 起始实体名称列表（来自 search_entities_by_name 结果）
        max_depth:    最大遍历深度（1~3；depth=3 在密集图中可能超时）
        max_edges:    最多返回的关系条数

    Returns:
        GraphSearchResult

    风险点:
        - max_depth=3 且图较密时，Cypher 可能触发 Neo4j 60s 查询超时
        - 生产环境建议在 neo4j.conf 中配置 dbms.transaction.timeout
    """
    if not entity_names:
        return GraphSearchResult(query_entities=entity_names)

    nodes_seen: set[str] = set()
    edges_seen: set[tuple] = set()
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []

    with driver.session() as session:
        for entity_name in entity_names:
            records = session.run(
                f"""
                MATCH (start:Entity {{name: $name}})
                OPTIONAL MATCH path = (start)-[r:RELATION*1..{max_depth}]-(nbr:Entity)
                UNWIND (CASE WHEN path IS NULL THEN [null] ELSE relationships(path) END) AS rel
                UNWIND (CASE WHEN path IS NULL THEN [start] ELSE nodes(path) END) AS n
                RETURN DISTINCT
                    n.name             AS node_name,
                    n.entity_type      AS node_type,
                    n.description      AS node_desc,
                    rel
                LIMIT $limit
                """,
                name=entity_name,
                limit=max_edges,
            )

            for record in records:
                # ── 收集节点 ────────────────────────────────────────────────
                node_name = record.get("node_name")
                if node_name and node_name not in nodes_seen:
                    nodes_seen.add(node_name)
                    nodes.append(
                        GraphNode(
                            name=node_name,
                            entity_type=record.get("node_type") or "未知",
                            description=record.get("node_desc"),
                        )
                    )

                # ── 收集关系 ────────────────────────────────────────────────
                rel = record.get("rel")
                if rel is None:
                    continue
                try:
                    src_name = rel.start_node["name"]
                    tgt_name = rel.end_node["name"]
                except (KeyError, TypeError, AttributeError):
                    continue

                rel_type = rel.get("relation_type", "未知")
                edge_key = (src_name, tgt_name, rel_type)
                if edge_key not in edges_seen:
                    edges_seen.add(edge_key)
                    edges.append(
                        GraphEdge(
                            source=src_name,
                            target=tgt_name,
                            relation_type=rel_type,
                            evidence=rel.get("evidence"),
                            amount=rel.get("amount"),
                        )
                    )

    result = GraphSearchResult(
        nodes=nodes,
        edges=edges,
        query_entities=entity_names,
    )
    logger.info(
        f"邻域遍历完成: {entity_names} (depth={max_depth}) → {result.stats()}"
    )
    return result


# ============================================================================
# § 3  主入口
# ============================================================================

def run_graph_search(
    driver: Driver,
    keywords: List[str],
    max_depth: Optional[int] = None,
    limit: int = 50,
) -> GraphSearchResult:
    """
    图谱检索主入口：关键词 → 模糊匹配实体 → N-hop 邻域展开。

    Args:
        driver:    Neo4j driver（依赖注入，可 mock）
        keywords:  实体关键词列表（来自 NormalizedQuery.entities）
        max_depth: 邻域遍历深度，默认使用 settings.graph_hop_depth
        limit:     最大返回关系数

    Returns:
        GraphSearchResult（is_empty=True 表示未命中任何实体）
    """
    _depth = max_depth if max_depth is not None else settings.graph_hop_depth

    if not keywords:
        logger.warning("run_graph_search: keywords 为空，返回空结果")
        return GraphSearchResult()

    # Step 1: 按关键词模糊匹配实体
    matched_nodes = search_entities_by_name(driver, keywords, limit=10)
    if not matched_nodes:
        logger.info(f"图谱检索：未找到匹配实体，keywords={keywords}")
        return GraphSearchResult(query_entities=keywords)

    # Step 2: 以匹配实体为起点展开邻域（限制起点数量防止查询超时）
    entity_names = [n.name for n in matched_nodes[:5]]
    neighborhood = get_entity_neighborhood(
        driver,
        entity_names=entity_names,
        max_depth=_depth,
        max_edges=limit,
    )

    # Step 3: 补全匹配到但未出现在邻域中的孤立节点
    existing_names = {n.name for n in neighborhood.nodes}
    for node in matched_nodes:
        if node.name not in existing_names:
            neighborhood.nodes.append(node)

    neighborhood.query_entities = keywords
    logger.info(f"图谱检索完成 → {neighborhood.stats()}") # 图谱检索完成 → 2 节点, 0 关系
    return neighborhood
