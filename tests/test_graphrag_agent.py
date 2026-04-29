"""
tests/test_graphrag_agent.py
────────────────────────────────────────────────────────────────────────────
【新增文件】graphrag_agent + graph_search 最小单元测试

测试策略:
  - 全部通过 mock，无需真实 LLM / Neo4j / Milvus 连接
  - 覆盖 Prompt Schema 解析、图谱检索逻辑、Agent 节点函数、路由条件

运行:
  cd enterprise-knowledge-brain
  pytest tests/test_graphrag_agent.py -v
"""
from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, call

import pytest

from agents.prompts import (
    NormalizedQuery,
    RoutingDecision,
    SynthesisInput,
    build_rewrite_prompt,
    build_route_prompt,
    build_synthesis_prompt,
)
from graphrag.graph_search import (
    GraphNode,
    GraphEdge,
    GraphSearchResult,
)
from agents.graphrag_agent import (
    AgentState,
    _route_condition,
    rewrite_query_node,
    route_query_node,
    vector_search_node,
    graph_search_node,
    synthesize_node,
    _format_vector_results,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_normalized() -> NormalizedQuery:
    return NormalizedQuery(
        normalized="招商银行股份有限公司投资了哪些科技项目",
        entities=["招商银行股份有限公司"],
        intent="查询招商银行的科技项目投资情况",
    )


@pytest.fixture
def sample_routing_vector() -> RoutingDecision:
    return RoutingDecision(route="vector_search", reasoning="主题概念类问题")


@pytest.fixture
def sample_routing_graph() -> RoutingDecision:
    return RoutingDecision(route="graph_search", reasoning="精确实体关系问题")


@pytest.fixture
def sample_routing_hybrid() -> RoutingDecision:
    return RoutingDecision(route="hybrid_search", reasoning="综合复杂问题")


@pytest.fixture
def sample_graph_result() -> GraphSearchResult:
    return GraphSearchResult(
        nodes=[
            GraphNode(name="招商银行股份有限公司", entity_type="公司", description="全国性股份制商业银行"),
            GraphNode(name="科技创新产业园区项目", entity_type="项目", description="重点投资项目"),
        ],
        edges=[
            GraphEdge(
                source="招商银行股份有限公司",
                target="科技创新产业园区项目",
                relation_type="投资",
                evidence="本行于2023年向科技创新产业园区项目投入资金3.5亿元",
                amount="3.5亿元",
            )
        ],
        query_entities=["招商银行股份有限公司"],
    )


@pytest.fixture
def sample_vector_results() -> List[Dict[str, Any]]:
    return [
        {
            "score": 0.912,
            "chunk_text": "招商银行于2023年加大科技投入，向多个产业项目注资。",
            "page_num": 12,
            "doc_id": "abc123",
            "entities_json": "[]",
        }
    ]


def _make_llm_mock(response_dict: dict) -> MagicMock:
    """构造返回固定 JSON 的 mock LLM client"""
    mock = MagicMock()
    mock.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps(response_dict, ensure_ascii=False)
    )
    return mock


def _make_neo4j_mock(records: list) -> MagicMock:
    """构造 mock Neo4j driver，session.run 返回 records"""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_session.run.return_value = iter(records)
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return mock_driver


# ============================================================================
# § 1  agents/prompts.py 测试
# ============================================================================

class TestPromptSchemas:

    def test_normalized_query_valid(self):
        nq = NormalizedQuery(
            normalized="招商银行投资了哪些项目",
            entities=["招商银行股份有限公司"],
            intent="查询投资项目",
        )
        assert nq.normalized != ""
        assert len(nq.entities) == 1

    def test_routing_decision_valid_routes(self):
        for route in ("vector_search", "graph_search", "hybrid_search"):
            rd = RoutingDecision(route=route, reasoning="test")
            assert rd.route == route

    def test_routing_decision_invalid_route(self):
        """非法 route 值应被 Pydantic 拒绝"""
        with pytest.raises(Exception):
            RoutingDecision(route="sql_search", reasoning="test")

    def test_build_rewrite_prompt_contains_question(self):
        prompt = build_rewrite_prompt("招行投了哪些基金")
        assert "招行投了哪些基金" in prompt

    def test_build_route_prompt_contains_entities(self, sample_normalized):
        prompt = build_route_prompt(sample_normalized)
        assert "招商银行股份有限公司" in prompt
        assert sample_normalized.normalized in prompt

    def test_build_synthesis_prompt_contains_context(self):
        inp = SynthesisInput(
            question="招行投资情况",
            vector_context="向量结果A",
            graph_context="图谱结果B",
        )
        prompt = build_synthesis_prompt(inp)
        assert "向量结果A" in prompt
        assert "图谱结果B" in prompt


# ============================================================================
# § 2  graphrag/graph_search.py 测试
# ============================================================================

class TestGraphSearchResult:

    def test_is_empty_when_no_data(self):
        result = GraphSearchResult()
        assert result.is_empty

    def test_is_not_empty_with_nodes(self):
        result = GraphSearchResult(
            nodes=[GraphNode(name="A", entity_type="公司")]
        )
        assert not result.is_empty

    def test_stats_format(self, sample_graph_result):
        stats = sample_graph_result.stats()
        assert "节点" in stats
        assert "关系" in stats

    def test_to_text_contains_entities(self, sample_graph_result):
        text = sample_graph_result.to_text()
        assert "招商银行股份有限公司" in text
        assert "科技创新产业园区项目" in text
        assert "投资" in text

    def test_to_text_empty_returns_placeholder(self):
        text = GraphSearchResult().to_text()
        assert "未找到" in text


class TestSearchEntitiesByName:

    def test_returns_matched_nodes(self):
        """mock session.run 返回匹配记录"""
        from graphrag.graph_search import search_entities_by_name

        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "name": "招商银行股份有限公司",
            "entity_type": "公司",
            "description": "全国性股份制商业银行",
        }[key]

        mock_driver = _make_neo4j_mock([mock_record])
        results = search_entities_by_name(mock_driver, ["招商银行"])
        assert len(results) == 1
        assert results[0].name == "招商银行股份有限公司"

    def test_empty_keywords_returns_empty(self):
        from graphrag.graph_search import search_entities_by_name
        mock_driver = MagicMock()
        results = search_entities_by_name(mock_driver, keywords=[])
        assert results == []
        mock_driver.session.assert_not_called()  # 不应调用 session


# ============================================================================
# § 3  AgentState 和路由条件测试
# ============================================================================

class TestRouteCondition:

    def test_error_routes_to_synthesize(self):
        state: AgentState = {"question": "test", "error": "some error"}
        assert _route_condition(state) == "synthesize"

    def test_no_routing_defaults_to_hybrid(self):
        state: AgentState = {"question": "test", "error": None, "routing": None}
        assert _route_condition(state) == "hybrid_search"

    def test_routing_vector(self, sample_routing_vector):
        state: AgentState = {
            "question": "test", "error": None, "routing": sample_routing_vector
        }
        assert _route_condition(state) == "vector_search"

    def test_routing_graph(self, sample_routing_graph):
        state: AgentState = {
            "question": "test", "error": None, "routing": sample_routing_graph
        }
        assert _route_condition(state) == "graph_search"

    def test_routing_hybrid(self, sample_routing_hybrid):
        state: AgentState = {
            "question": "test", "error": None, "routing": sample_routing_hybrid
        }
        assert _route_condition(state) == "hybrid_search"


# ============================================================================
# § 4  各 Agent 节点函数测试（mock LLM/DB）
# ============================================================================

class TestRewriteQueryNode:

    def test_successful_rewrite(self):
        mock_llm = _make_llm_mock({
            "normalized": "招商银行股份有限公司投资了哪些科技项目",
            "entities": ["招商银行股份有限公司"],
            "intent": "查询招商银行科技类投资项目",
        })
        state: AgentState = {"question": "招行投了哪些科技项目"}
        result = rewrite_query_node(state, llm_client=mock_llm)
        assert result.get("error") is None
        assert isinstance(result["normalized"], NormalizedQuery)
        assert "招商银行" in result["normalized"].normalized

    def test_llm_failure_sets_error(self):
        mock_llm = MagicMock()
        mock_llm.chat.completions.create.side_effect = Exception("API down")
        state: AgentState = {"question": "测试问题"}
        result = rewrite_query_node(state, llm_client=mock_llm)
        assert result.get("error") is not None
        assert "normalized" not in result or result.get("normalized") is None


class TestRouteQueryNode:

    def test_successful_routing(self, sample_normalized):
        mock_llm = _make_llm_mock({
            "route": "graph_search",
            "reasoning": "涉及精确实体关系",
        })
        state: AgentState = {
            "question": "test",
            "normalized": sample_normalized,
            "error": None,
        }
        result = route_query_node(state, llm_client=mock_llm)
        assert isinstance(result["routing"], RoutingDecision)
        assert result["routing"].route == "graph_search"

    def test_skips_on_existing_error(self):
        """上游有 error 时，route 节点应直接返回空 dict"""
        mock_llm = MagicMock()
        state: AgentState = {"question": "test", "error": "upstream error"}
        result = route_query_node(state, llm_client=mock_llm)
        assert result == {}
        mock_llm.chat.completions.create.assert_not_called()

    def test_llm_failure_falls_back_to_hybrid(self, sample_normalized):
        mock_llm = MagicMock()
        mock_llm.chat.completions.create.side_effect = Exception("timeout")
        state: AgentState = {
            "question": "test",
            "normalized": sample_normalized,
            "error": None,
        }
        result = route_query_node(state, llm_client=mock_llm)
        assert result["routing"].route == "hybrid_search"


class TestVectorSearchNode:

    def test_returns_vector_results(self, sample_normalized, sample_vector_results):
        mock_milvus = MagicMock()
        # mock MilvusClient.search 返回格式
        mock_hit = MagicMock()
        mock_hit.get.side_effect = lambda k, d=None: {
            "distance": 0.912, "entity": sample_vector_results[0]
        }.get(k, d)
        mock_milvus.search.return_value = [[mock_hit]]

        state: AgentState = {
            "question": "test",
            "normalized": sample_normalized,
            "error": None,
        }

        with patch("agents.graphrag_agent._get_embedding_model") as mock_model:
            mock_model.return_value.encode.return_value = [[0.0] * 1024]
            result = vector_search_node(state, milvus_client=mock_milvus)

        assert "vector_results" in result


class TestSynthesizeNode:

    def test_error_state_returns_error_message(self):
        mock_llm = MagicMock()
        state: AgentState = {
            "question": "test",
            "error": "LLM 调用失败",
            "vector_results": [],
            "graph_result": None,
        }
        result = synthesize_node(state, llm_client=mock_llm)
        assert "error" in result["answer"].lower() or "异常" in result["answer"]
        mock_llm.chat.completions.create.assert_not_called()

    def test_generates_answer(self, sample_normalized, sample_graph_result):
        mock_llm = MagicMock()
        mock_llm.chat.completions.create.return_value.choices[0].message.content = (
            "招商银行于2023年向科技创新产业园区项目投资3.5亿元。"
        )
        state: AgentState = {
            "question": "招行投了哪些科技项目",
            "normalized": sample_normalized,
            "error": None,
            "vector_results": [],
            "graph_result": sample_graph_result,
        }
        result = synthesize_node(state, llm_client=mock_llm)
        assert len(result["answer"]) > 0
        assert "3.5亿元" in result["answer"]


# ============================================================================
# § 5  格式化辅助函数测试
# ============================================================================

class TestFormatVectorResults:

    def test_empty_returns_placeholder(self):
        text = _format_vector_results([])
        assert "未找到" in text

    def test_single_result_formatted(self, sample_vector_results):
        text = _format_vector_results(sample_vector_results)
        assert "0.912" in text
        assert "招商银行" in text

    def test_truncates_long_chunk_text(self):
        results = [
            {
                "score": 0.9,
                "chunk_text": "A" * 1000,  # 超长文本
                "page_num": 1,
                "doc_id": "x",
                "entities_json": "[]",
            }
        ]
        text = _format_vector_results(results)
        # 检查截断后不超过合理长度
        assert len(text) < 1000
