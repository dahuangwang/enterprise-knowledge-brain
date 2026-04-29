"""
tests/test_indexer.py
────────────────────────────────────────────────────────────────────────────
【新增文件】graphrag/indexer.py 的最小单元测试

测试策略:
  - Mock LLM client（不真实调用 API）
  - Mock Neo4j driver（不需要 DB 连接）
  - Mock Milvus client（不需要 DB 连接）
  - 只验证函数逻辑的正确性

运行:
  cd enterprise-knowledge-brain
  pytest tests/test_indexer.py -v
"""

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from graphrag.prompts import (
    Entity,
    ExtractionResult,
    Relation,
    build_extraction_prompt,
    ENTITY_TYPES,
    RELATION_TYPES,
)
from graphrag.indexer import (
    split_into_chunks,
    extract_from_chunk,
    write_entities_to_neo4j,
    write_relations_to_neo4j,
    write_to_milvus,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_pages() -> List[dict]:
    """模拟 2 页 PDF 内容"""
    return [
        {
            "page": 1,
            "text": (
                "招商银行股份有限公司（以下简称'本行'）于2023年向科技创新产业园区项目投入资金3.5亿元，"
                "该项目位于深圳市南山区，是本行重点支持的战略性新兴产业项目。"
                "本行持有平安银行股份有限公司约15%的股份，双方在零售金融领域开展深度合作。"
            ) * 3,  # 重复3次，确保产生多个 chunk
        },
        {
            "page": 2,
            "text": (
                "公司全资子公司招银国际金融控股有限公司负责集团境外投资业务，"
                "管理海外资产规模约500亿港元。"
                "报告期内，公司向粤港澳大湾区数字经济基金发行了总规模30亿元的优先级份额。"
            ) * 2,
        },
    ]


@pytest.fixture
def sample_extraction() -> ExtractionResult:
    """模拟一次 LLM 抽取结果"""
    return ExtractionResult(
        entities=[
            Entity(name="招商银行股份有限公司", entity_type="公司", description="全国性股份制商业银行"),
            Entity(name="科技创新产业园区项目", entity_type="项目", description="战略性新兴产业项目"),
            Entity(name="平安银行股份有限公司", entity_type="公司"),
        ],
        relations=[
            Relation(
                source="招商银行股份有限公司",
                target="科技创新产业园区项目",
                relation_type="投资",
                evidence="招商银行股份有限公司于2023年向科技创新产业园区项目投入资金3.5亿元",
                amount="3.5亿元",
            ),
            Relation(
                source="招商银行股份有限公司",
                target="平安银行股份有限公司",
                relation_type="持股",
                evidence="本行持有平安银行股份有限公司约15%的股份",
                amount="15%",
            ),
        ],
    )


@pytest.fixture
def mock_llm_client(sample_extraction: ExtractionResult) -> MagicMock:
    """返回固定 JSON 的 mock LLM client"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = sample_extraction.model_dump_json(
        exclude_none=False
    )
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_neo4j_driver() -> MagicMock:
    """Mock Neo4j driver，避免真实 DB 连接"""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return mock_driver


@pytest.fixture
def mock_milvus_client() -> MagicMock:
    """Mock MilvusClient，避免真实 DB 连接"""
    mock_client = MagicMock()
    mock_client.insert.return_value = {"insert_count": 5}
    mock_client.has_collection.return_value = True
    return mock_client


# ============================================================================
# § 1   文本切块测试
# ============================================================================

class TestSplitIntoChunks:

    def test_basic_chunking(self, sample_pages):
        """验证切块产生正确数量的 chunk"""
        chunks = split_into_chunks(sample_pages, chunk_size=200, overlap=20)
        assert len(chunks) > 0
        assert all("chunk_idx" in c for c in chunks)
        assert all("page" in c for c in chunks)
        assert all("text" in c for c in chunks)

    def test_chunk_idx_sequential(self, sample_pages):
        """chunk_idx 必须连续递增"""
        chunks = split_into_chunks(sample_pages, chunk_size=200, overlap=20)
        idxs = [c["chunk_idx"] for c in chunks]
        assert idxs == list(range(len(idxs)))

    def test_chunk_size_respected(self, sample_pages):
        """每个 chunk 不超过 chunk_size 字符"""
        chunk_size = 300
        chunks = split_into_chunks(sample_pages, chunk_size=chunk_size, overlap=50)
        for c in chunks:
            assert len(c["text"]) <= chunk_size, (
                f"Chunk {c['chunk_idx']} 超出大小限制: {len(c['text'])} > {chunk_size}"
            )

    def test_page_preserved(self, sample_pages):
        """chunk 保留来源页码"""
        chunks = split_into_chunks(sample_pages, chunk_size=200, overlap=20)
        pages_in_chunks = {c["page"] for c in chunks}
        assert 1 in pages_in_chunks
        assert 2 in pages_in_chunks

    def test_invalid_overlap_raises(self, sample_pages):
        """overlap >= chunk_size 时应抛出 ValueError"""
        with pytest.raises(ValueError):
            split_into_chunks(sample_pages, chunk_size=100, overlap=100)

    def test_empty_pages(self):
        """空页面列表返回空 chunks"""
        chunks = split_into_chunks([])
        assert chunks == []


# ============================================================================
# § 2  Prompt 构建测试
# ============================================================================

class TestBuildExtractionPrompt:

    def test_prompt_contains_text(self):
        prompt = build_extraction_prompt(page=1, chunk_idx=0, text="测试文本内容")
        assert "测试文本内容" in prompt

    def test_prompt_contains_page_info(self):
        prompt = build_extraction_prompt(page=5, chunk_idx=10, text="abc")
        assert "5" in prompt
        assert "10" in prompt

    def test_prompt_contains_entity_types(self):
        prompt = build_extraction_prompt(page=1, chunk_idx=0, text="test")
        for et in ENTITY_TYPES:
            assert et in prompt, f"实体类型 '{et}' 未出现在 prompt 中"

    def test_prompt_contains_relation_types(self):
        prompt = build_extraction_prompt(page=1, chunk_idx=0, text="test")
        for rt in RELATION_TYPES:
            assert rt in prompt, f"关系类型 '{rt}' 未出现在 prompt 中"


# ============================================================================
# § 3  Schema 验证测试
# ============================================================================

class TestExtractionResultSchema:

    def test_empty_result(self):
        result = ExtractionResult()
        assert result.is_empty
        assert result.stats() == "0 实体, 0 关系"

    def test_parse_from_dict(self):
        data = {
            "entities": [
                {"name": "招商银行", "entity_type": "公司", "description": None}
            ],
            "relations": [],
        }
        result = ExtractionResult(**data)
        assert len(result.entities) == 1
        assert result.entities[0].name == "招商银行"

    def test_entity_without_description(self):
        entity = Entity(name="平安银行", entity_type="公司")
        assert entity.description is None

    def test_relation_without_amount(self):
        rel = Relation(
            source="A",
            target="B",
            relation_type="投资",
            evidence="A向B投资",
        )
        assert rel.amount is None


# ============================================================================
# § 4  LLM 抽取测试（mock）
# ============================================================================

class TestExtractFromChunk:

    def test_successful_extraction(self, mock_llm_client, sample_extraction):
        """mock LLM 返回正确 JSON，验证解析结果"""
        chunk = {"chunk_idx": 0, "page": 1, "text": "任意文本"}
        result = extract_from_chunk(chunk, mock_llm_client)
        assert len(result.entities) == len(sample_extraction.entities)
        assert len(result.relations) == len(sample_extraction.relations)

    def test_llm_failure_returns_empty(self):
        """LLM 调用失败时返回空 ExtractionResult，不抛出异常"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        chunk = {"chunk_idx": 0, "page": 1, "text": "任意文本"}
        result = extract_from_chunk(chunk, mock_client, max_retries=0)
        assert result.is_empty

    def test_invalid_json_returns_empty(self):
        """LLM 返回非法 JSON 时返回空结果"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "这不是 JSON"
        mock_client.chat.completions.create.return_value = mock_response
        chunk = {"chunk_idx": 0, "page": 1, "text": "任意文本"}
        result = extract_from_chunk(chunk, mock_client, max_retries=0)
        assert result.is_empty


# ============================================================================
# § 5  Neo4j 写入测试（mock）
# ============================================================================

class TestWriteToNeo4j:

    def test_write_entities(self, mock_neo4j_driver, sample_extraction):
        count = write_entities_to_neo4j(
            sample_extraction.entities, mock_neo4j_driver, doc_id="test_doc"
        )
        assert count == len(sample_extraction.entities)
        # 验证 session.run 被调用了正确次数
        session = mock_neo4j_driver.session.return_value.__enter__.return_value
        assert session.run.call_count == len(sample_extraction.entities)

    def test_write_relations(self, mock_neo4j_driver, sample_extraction):
        count = write_relations_to_neo4j(
            sample_extraction.relations, mock_neo4j_driver, doc_id="test_doc"
        )
        assert count == len(sample_extraction.relations)

    def test_write_empty_entities(self, mock_neo4j_driver):
        count = write_entities_to_neo4j([], mock_neo4j_driver, doc_id="test_doc")
        assert count == 0
        mock_neo4j_driver.session.assert_not_called()


# ============================================================================
# § 6  Milvus 写入测试（mock）
# ============================================================================

class TestWriteToMilvus:

    def test_write_chunks(
        self, mock_milvus_client, sample_extraction, sample_pages
    ):
        chunks = split_into_chunks(sample_pages, chunk_size=200, overlap=20)
        extractions = [sample_extraction] * len(chunks)
        embeddings = [[0.0] * 1024] * len(chunks)  # 假 embedding

        with patch("graphrag.indexer.settings") as mock_settings:
            mock_settings.milvus_collection = "test_collection"
            mock_settings.embedding_dim = 1024

            write_to_milvus(
                chunks, extractions, embeddings, mock_milvus_client, doc_id="test_doc"
            )

        mock_milvus_client.insert.assert_called_once()
        insert_call_args = mock_milvus_client.insert.call_args
        inserted_data = insert_call_args.kwargs.get("data") or insert_call_args.args[1] if insert_call_args.args else insert_call_args.kwargs["data"]
        assert len(inserted_data) == len(chunks)


# ============================================================================
# § 7  集成测试（dry-run，不需要任何 DB）
# ============================================================================

class TestRunIndexingPipelineDryRun:

    def test_dry_run_with_mock_pdf(self, tmp_path, mock_llm_client):
        """
        创建一个真实的（简单）PDF 测试 dry-run 流程。
        需要 reportlab 或直接 mock load_pdf。
        此处 mock load_pdf 避免依赖 PDF 文件。
        """
        from graphrag.indexer import run_indexing_pipeline

        fake_pages = [{"page": 1, "text": "招商银行向科技园区项目投资3.5亿元" * 20}]

        # 创建一个假 PDF 文件（只需存在，load_pdf 被 mock 掉）
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        with patch("graphrag.indexer.load_pdf", return_value=fake_pages):
            with patch("graphrag.indexer.build_embeddings", return_value=[[0.0] * 1024]):
                result = run_indexing_pipeline(
                    str(fake_pdf),
                    dry_run=True,
                    llm_client=mock_llm_client,
                )

        assert result["pdf_name"] == "test.pdf"
        assert result["total_chunks"] > 0
        assert result["total_entities"] == 0   # dry_run 不写入
        assert result["milvus_inserted"] == 0  # dry_run 不写入
        assert "doc_id" in result
