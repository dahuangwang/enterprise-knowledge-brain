"""
graphrag/indexer.py
────────────────────────────────────────────────────────────────────────────
【新增文件】GraphRAG 离线建库流水线

流程:
    PDF
     └─► pdfplumber 页面提取
          └─► 滑动窗口切块
               └─► DeepSeek V3 (JSON mode) 实体关系抽取  ──► Neo4j (MERGE)
                    └─► BAAI/bge-m3 本地 Embedding
                         └─► Milvus (IVF_FLAT + COSINE)

设计原则:
    - LLM client 通过参数注入，测试可 mock
    - Neo4j driver 通过参数注入，可替换
    - Milvus collection 通过参数注入，可替换
    - 所有 Prompt / Schema 来自 graphrag.prompts，不硬编码在此

CLI:
    python -m graphrag.indexer --pdf path/to/annual_report.pdf
    python -m graphrag.indexer --pdf path/to/annual_report.pdf --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
from openai import OpenAI
from neo4j import GraphDatabase, Driver
from pymilvus import (
    MilvusClient,
    DataType,
)
from sentence_transformers import SentenceTransformer

from core.config import settings
from core.logger import get_logger
from graphrag.prompts import (
    ExtractionResult,
    Entity,
    Relation,
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
)

logger = get_logger(__name__)


# ============================================================================
# § 1  PDF 解析与文本切块
# ============================================================================

def load_pdf(pdf_path: str) -> List[dict]:
    """
    使用 pdfplumber 逐页提取文本。

    Args:
        pdf_path: PDF 文件路径（绝对或相对）

    Returns:
        [{"page": int, "text": str}, ...] — 只包含有效文本页
    """
    pages: List[dict] = []
    total_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page": i, "text": text})

    logger.info(
        f"PDF 解析完成: [bold]{len(pages)}[/bold] 有效页 / 共 {total_pages} 页"
        f"  ({pdf_path})"
    )
    return pages


def split_into_chunks(
    pages: List[dict],
    chunk_size: int = 800,
    overlap: int = 100,
) -> List[dict]:
    """
    滑动窗口切块。每个 chunk 保留来源页码，方便溯源。

    Args:
        pages:      load_pdf() 返回的页面列表
        chunk_size: 每块最大字符数
        overlap:    相邻块共享的字符数（防止边界截断实体）

    Returns:
        [{"chunk_idx": int, "page": int, "text": str}, ...]
    """
    if chunk_size <= overlap:
        raise ValueError(f"chunk_size({chunk_size}) 必须大于 overlap({overlap})")

    chunks: List[dict] = []
    chunk_idx = 0

    for page_info in pages:
        text: str = page_info["text"]
        page: int = page_info["page"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {"chunk_idx": chunk_idx, "page": page, "text": chunk_text}
                )
                chunk_idx += 1
            if end == len(text):
                break
            start += chunk_size - overlap

    logger.info(
        f"文本切块完成: [bold]{len(chunks)}[/bold] 个 chunks "
        f"(size={chunk_size}, overlap={overlap})"
    )
    return chunks


# ============================================================================
# § 2  LLM 实体关系抽取（DeepSeek V3，JSON mode）
# ============================================================================

def build_llm_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """
    构建 OpenAI 兼容客户端（默认指向 DeepSeek V3）。
    通过参数注入，测试时可传入 mock client。

    Args:
        api_key:  覆盖 settings.deepseek_api_key
        base_url: 覆盖 settings.deepseek_base_url（可切换到其他兼容 API）
    """
    client = OpenAI(
        api_key=api_key or settings.deepseek_api_key,
        base_url=base_url or settings.deepseek_base_url,
    )
    logger.info(
        f"LLM 客户端就绪: [cyan]{base_url or settings.deepseek_base_url}[/cyan]"
        f" / {settings.deepseek_model}"
    )
    return client


def extract_from_chunk(
    chunk: dict,
    client: OpenAI,
    model: Optional[str] = None,
    max_retries: int = 2,
) -> ExtractionResult:
    """
    对单个 chunk 调用 LLM，返回结构化 ExtractionResult。
    使用指数退避重试，全部失败返回空结果（不中断流水线）。

    Args:
        chunk:       {"chunk_idx": int, "page": int, "text": str}
        client:      OpenAI 兼容客户端
        model:       模型名，默认使用 settings.deepseek_model
        max_retries: 最大重试次数

    Returns:
        ExtractionResult —— 失败时为空结果
        ExtractionResult成功时格式：
        {
            "entities": [
                {
                    "name": "招商银行股份有限公司",
                    "entity_type": "公司",
                    "description": "全国性股份制商业银行"
                },
                {
                    "name": "科技创新产业园区项目",
                    "entity_type": "项目",
                    "description": "战略性新兴产业项目"
                },
                {
                    "name": "平安银行股份有限公司",
                    "entity_type": "公司"
                }
            ],
            "relations": [
                {
                    "source": "招商银行股份有限公司",
                    "target": "科技创新产业园区项目",
                    "relation_type": "投资",
                    "evidence": "招商银行股份有限公司于2023年向科技创新产业园区项目投入资金3.5亿元",
                    "amount": "3.5亿元"
                },
                {
                    "source": "招商银行股份有限公司",
                    "target": "平安银行股份有限公司",
                    "relation_type": "持股",
                    "evidence": "本行持有平安银行股份有限公司约15%的股份",
                    "amount": "15%"
                }
            ]
        }
    """
    prompt = build_extraction_prompt(
        page=chunk["page"],
        chunk_idx=chunk["chunk_idx"],
        text=chunk["text"],
    )
    _model = model or settings.deepseek_model
    # 指数退避重试
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=settings.llm_temperature,
                timeout=60,
            )
            raw_json: str = response.choices[0].message.content
            data: dict = json.loads(raw_json)
            # result是ExtractionResult类的实例，包含entities和relations两个列表
            result = ExtractionResult(**data)
            if not result.is_empty:
                logger.debug(
                    f"[Chunk {chunk['chunk_idx']:>4}|p{chunk['page']}] "
                    f"✓ {result.stats()}"
                )
            return result

        except json.JSONDecodeError as e:
            logger.warning(
                f"[Chunk {chunk['chunk_idx']}] JSON 解析失败 (尝试 {attempt + 1}): {e}"
            )
        except Exception as e:
            logger.warning(
                f"[Chunk {chunk['chunk_idx']}] LLM 调用失败 "
                f"(尝试 {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {e}"
            )

        if attempt < max_retries:
            wait = 2**attempt  # 1s, 2s, ...
            time.sleep(wait)

    logger.error(f"[Chunk {chunk['chunk_idx']}] 抽取最终失败，跳过")
    return ExtractionResult()


# ============================================================================
# § 3  Neo4j 图谱写入
# ============================================================================

def get_neo4j_driver(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Driver:
    """
    创建并验证 Neo4j 连接。参数可覆盖 settings，方便测试注入。

    风险点:
      - 若 Neo4j 未启动，此处抛出 ServiceUnavailable，应先 docker-compose up neo4j -d
    """
    driver = GraphDatabase.driver(
        uri or settings.neo4j_uri,
        auth=(user or settings.neo4j_user, password or settings.neo4j_password),
    )
    driver.verify_connectivity()  # 提前验证，快速失败，这样做的目的是防止后续代码报错
    logger.info(f"Neo4j 连接成功: [green]{uri or settings.neo4j_uri}[/green]")
    return driver

# 确保 Neo4j 约束
def ensure_neo4j_constraints(driver: Driver) -> None:
    """
    创建实体名为唯一约束（幂等操作）。
    防止并发 MERGE 产生重复节点。
    """
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
    logger.info("Neo4j 唯一约束已就绪")

# 向neo4j写入实体节点
def write_entities_to_neo4j(
    entities: List[Entity],
    driver: Driver,
    doc_id: str,
) -> int:
    """
    批量 MERGE 实体节点。
    ON CREATE: 写入所有属性
    ON MATCH:  更新 entity_type 和 description（更新比首次写入更完整时）

    Returns:
        实际处理的实体数量
    """
    if not entities:
        return 0

    with driver.session() as session:
        for entity in entities:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET
                    e.entity_type = $entity_type,
                    e.description = $description,
                    e.source_doc  = $doc_id,
                    e.created_at  = timestamp()
                ON MATCH SET
                    e.entity_type = $entity_type,
                    e.description = coalesce(nullif($description, ''), e.description),
                    e.updated_at  = timestamp()
                """,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description or "",
                doc_id=doc_id,
            )
    return len(entities)

# 向neo4j写入关系边（关系）
def write_relations_to_neo4j(
    relations: List[Relation],
    driver: Driver,
    doc_id: str,
) -> int:
    """
    批量 MERGE 关系边。
    source/target 节点若不存在会自动创建（保证图的完整性）。

    风险点:
      - relation_type 存为属性而非关系类型标签，是为了支持动态关系类型；
        如需按关系类型索引，可改为 APOC 动态关系创建。

    Returns:
        实际处理的关系数量
    """
    if not relations:
        return 0

    with driver.session() as session:
        for rel in relations:
            session.run(
                """
                MERGE (s:Entity {name: $source})
                MERGE (t:Entity {name: $target})
                MERGE (s)-[r:RELATION {relation_type: $relation_type}]->(t)
                ON CREATE SET
                    r.evidence   = $evidence,
                    r.amount     = $amount,
                    r.source_doc = $doc_id,
                    r.created_at = timestamp()
                ON MATCH SET
                    r.evidence   = $evidence,
                    r.amount     = coalesce(nullif($amount, ''), r.amount),
                    r.updated_at = timestamp()
                """,
                source=rel.source,
                target=rel.target,
                relation_type=rel.relation_type,
                evidence=rel.evidence,
                amount=rel.amount or "",
                doc_id=doc_id,
            )
    return len(relations)


def write_extraction_to_neo4j(
    extraction: ExtractionResult,
    driver: Driver,
    doc_id: str,
) -> Tuple[int, int]:
    """
    写入单次抽取结果，返回 (entity_count, relation_count)。
    """
    ec = write_entities_to_neo4j(extraction.entities, driver, doc_id)
    rc = write_relations_to_neo4j(extraction.relations, driver, doc_id)
    return ec, rc


# ============================================================================
# § 4  Embedding + Milvus 向量库写入
# ============================================================================


def build_embeddings(
    texts: List[str],
    model_name: Optional[str] = None,
) -> List[List[float]]:
    """
    使用本地 sentence-transformers 模型批量生成 embedding。
    normalize_embeddings=True: L2 归一化，使 COSINE 相似度等价于点积，加速检索。

    Args:
        texts:      待 embed 的文本列表
        model_name: 覆盖 settings.embedding_model（可替换为其他模型）

    风险点:
      - 首次运行会从 HuggingFace 下载模型（约 2.3GB），需要网络访问
      - 如 HuggingFace 访问受限，设置环境变量:
          HF_ENDPOINT=https://hf-mirror.com

    Returns:
        List[List[float]]，长度与 texts 相同，每个向量维度为 embedding_dim
    """
    _model_name = model_name or settings.embedding_model
    logger.info(f"加载 Embedding 模型: [cyan]{_model_name}[/cyan]")
    model = SentenceTransformer(_model_name)

    logger.info(f"生成 Embedding，共 [bold]{len(texts)}[/bold] 条文本 ...")
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True, # 显示进度条
        normalize_embeddings=True, # 归一化，使COSINE相似度等价于点积，加速检索
    )
    return vectors.tolist()


def get_milvus_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> MilvusClient:
    """
    创建 Milvus 客户端连接（pymilvus MilvusClient API，2.4+）。

    风险点:
      - Milvus standalone 需要 docker-compose up milvus-standalone etcd minio -d
      - 启动后等待约 30s 才能接受连接，可用 healthcheck 确认
    """
    _host = host or settings.milvus_host
    _port = port or settings.milvus_port
    uri = f"http://{_host}:{_port}"

    try:
        client = MilvusClient(uri=uri)
        logger.info(f"Milvus 连接成功: [green]{uri}[/green]")
        return client
    except Exception as e:
        raise ConnectionError(
            f"无法连接 Milvus ({uri}): {e}\n"
            "请先运行: docker-compose up milvus-standalone etcd minio -d\n"
            "并等待约 30s 后再试。"
        ) from e


def ensure_milvus_collection(client: MilvusClient) -> None:
    """
    确保 Milvus collection 存在（幂等）。
    Schema:
        id            INT64   PK auto_id
        doc_id        VARCHAR(64)
        chunk_idx     INT64
        page_num      INT64
        chunk_text    VARCHAR(4096)
        entities_json VARCHAR(8192)   — 抽取实体的 JSON 快照，方便检索后溯源
        embedding     FLOAT_VECTOR(dim)

    风险点:
      - VARCHAR max_length 超出会导致插入失败，chunk_text 写入前截断到 4000 字符
    """
    col_name = settings.milvus_collection

    if client.has_collection(col_name):
        logger.info(f"Milvus collection 已存在: [cyan]{col_name}[/cyan]")
        return
    # 创建schema对象
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False) 
    schema.add_field("id",            DataType.INT64,        is_primary=True)
    schema.add_field("doc_id",        DataType.VARCHAR,      max_length=64)
    schema.add_field("chunk_idx",     DataType.INT64)
    schema.add_field("page_num",      DataType.INT64)
    schema.add_field("chunk_text",    DataType.VARCHAR,      max_length=4096)
    schema.add_field("entities_json", DataType.VARCHAR,      max_length=8192)
    schema.add_field("embedding",     DataType.FLOAT_VECTOR, dim=settings.embedding_dim)

    index_params = client.prepare_index_params()
    # 给field添加索引
    index_params.add_index(
        field_name="embedding",# 稠密向量
        metric_type="COSINE", # 余弦相似度
        index_type="IVF_FLAT", # 倒排索引
        params={"nlist": 128},
    )
    # 创建collection对象
    client.create_collection(
        collection_name=col_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info(
        f"Milvus collection 创建完成: [cyan]{col_name}[/cyan] "
        f"(dim={settings.embedding_dim}, IVF_FLAT + COSINE)"
    )


def write_to_milvus(
    chunks: List[dict],
    extractions: List[ExtractionResult],
    embeddings: List[List[float]],
    client: MilvusClient,
    doc_id: str,
) -> int:
    """
    批量插入 chunks + embeddings 到 Milvus。

    Args:
        chunks:      split_into_chunks() 结果
        extractions: 与 chunks 一一对应的 LLM 抽取结果（包含实体和关系）
        embeddings:  与 chunks 一一对应的向量列表 
        client:      MilvusClient
        doc_id:      文档唯一标识

    Returns:
        成功插入的行数

    风险点:
      - chunk_text 超过 4096 字符会被截断（pdfplumber 偶有超长段落）
      - entities_json 超过 8192 字符会被截断（极端情况下实体过多）
    """
    col_name = settings.milvus_collection
    rows: List[dict] = []

    for chunk, extraction, emb in zip(chunks, extractions, embeddings):
        # dump是python对象转成json字符串
        entities_json = json.dumps(
            [e.model_dump() for e in extraction.entities],
            ensure_ascii=False,
        )
        rows.append(
            {
                "doc_id":        doc_id,
                "chunk_idx":     chunk["chunk_idx"],
                "page_num":      chunk["page"],
                "chunk_text":    chunk["text"][:4000],       # 截断保护
                "entities_json": entities_json[:8000],       # 截断保护
                "embedding":     emb,
            }
        )

    result = client.insert(collection_name=col_name, data=rows)
    inserted = result.get("insert_count", len(rows))
    logger.info(f"Milvus 写入完成: [bold]{inserted}[/bold] 条向量")
    return inserted


# ============================================================================
# § 5  主流水线
# ============================================================================

def run_indexing_pipeline(
    pdf_path: str,
    dry_run: bool = False,
    llm_client: Optional[OpenAI] = None,
    neo4j_driver: Optional[Driver] = None,
    milvus_client: Optional[MilvusClient] = None,
) -> dict:
    """
    完整的离线建库流水线。

    Args:
        pdf_path:      PDF 文件路径
        dry_run:       True 时只解析 PDF + 打印 LLM 抽取结果（实体和关系），不写入数据库
        llm_client:    可注入 mock client（测试用）
        neo4j_driver:  可注入 mock driver（测试用）
        milvus_client: 可注入 mock client（测试用）

    Returns:
        {
            "doc_id": str,
            "pdf_name": str,
            "total_pages": int,
            "total_chunks": int,
            "total_entities": int,
            "total_relations": int,
            "milvus_inserted": int,
        }
        返回文档的id，名称，页数，分块数，实体数，关系数，milvus插入数

    风险点:
      - 大型 （100+ PDF页）耗时较长，建议后台任务运行
      - DeepSeek API 有速率限制，llm_batch_size 过大可能触发 429
      - 首次运行需下载 bge-m3 模型（约 2.3 GB）
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
    # 用md5对pdf文档路径对象做hash处理，生成文档id
    doc_id = hashlib.md5(pdf_path_obj.name.encode("utf-8")).hexdigest()[:16]

    logger.info("=" * 62)
    logger.info(f"[bold]开始建库[/bold]: {pdf_path_obj.name}  [dim](doc_id={doc_id})[/dim]")
    if dry_run:
        logger.info("[yellow]⚠  DRY-RUN 模式: 不写入数据库[/yellow]")
    logger.info("=" * 62)

    # ── Step 1: 解析 PDF ─────────────────────────────────────────────────────
    # pafes 格式 [{"page": 1, "text": "..."}, ...]
    pages = load_pdf(str(pdf_path_obj))
    # chunks 格式 [{"chunk_idx": 0, "page": 1, "text": "..."}, ...]
    chunks = split_into_chunks(pages, settings.chunk_size, settings.chunk_overlap)
    # 获取最大页数
    total_pages = max((c["page"] for c in chunks), default=0)

    # ── Step 2: LLM 抽取 + Neo4j 写入 ───────────────────────────────────────
    _llm = llm_client or build_llm_client()

    _neo4j = None
    try:
        if not dry_run:
            _neo4j = neo4j_driver or get_neo4j_driver()
            ensure_neo4j_constraints(_neo4j)

        extractions: List[ExtractionResult] = []
        total_entities = 0
        total_relations = 0
        # 按批次处理chunks，通过大模型调用抽取实体和关系，将结果统计打印或保存到neo4j
        for batch_start in range(0, len(chunks), settings.llm_batch_size):
            batch = chunks[batch_start : batch_start + settings.llm_batch_size]
            logger.info(
                f"[LLM] chunks {batch_start:>3}~{batch_start + len(batch) - 1:<3} "
                f"/ {len(chunks) - 1}"
            )

            for chunk in batch:
                # extract_from_chunk对单个chunk调用llm提取关系
                extraction = extract_from_chunk(
                    chunk, _llm, max_retries=settings.llm_max_retries
                )
                extractions.append(extraction)

                if dry_run:
                    if not extraction.is_empty:
                        logger.info(f"  [DRY] p{chunk['page']}: {extraction.stats()}")
                    continue

                ec, rc = write_extraction_to_neo4j(extraction, _neo4j, doc_id)
                total_entities += ec
                total_relations += rc

        if not dry_run and _neo4j:
            logger.info(
                f"Neo4j 写入完成: "
                f"[bold green]{total_entities}[/bold green] 实体, "
                f"[bold green]{total_relations}[/bold green] 关系"
            )
    finally:
        if not dry_run and _neo4j:
            _neo4j.close()
            logger.info("Neo4j 连接已关闭")

    # ── Step 3: Embedding ───────────────────────────────────────────────────
    texts = [c["text"] for c in chunks]
    # texts: 待 embed 的文本列表，格式为[text1, text2, text3, ...]
    embeddings = build_embeddings(texts)

    # ── Step 4: Milvus 写入 ─────────────────────────────────────────────────
    milvus_inserted = 0
    if not dry_run:
        _milvus = None
        try:
            _milvus = milvus_client or get_milvus_client()
            ensure_milvus_collection(_milvus)
            milvus_inserted = write_to_milvus(
                chunks, extractions, embeddings, _milvus, doc_id
            )
        finally:
            if _milvus:
                _milvus.close()
                logger.info("Milvus 连接已关闭")

    summary = {
        "doc_id":          doc_id,
        "pdf_name":        pdf_path_obj.name,
        "total_pages":     total_pages,
        "total_chunks":    len(chunks),
        "total_entities":  total_entities,
        "total_relations": total_relations,
        "milvus_inserted": milvus_inserted,
    }

    logger.info("=" * 62)
    logger.info(f"[bold green]✅ 建库完成![/bold green]  摘要: {summary}")
    logger.info("=" * 62)
    return summary


# ============================================================================
# § 6  CLI 入口
# ============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GraphRAG 离线建库: PDF → Neo4j + Milvus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m graphrag.indexer --pdf ./data/annual_report.pdf
  python -m graphrag.indexer --pdf ./data/annual_report.pdf --dry-run
        """,
    )
    parser.add_argument("--pdf", required=True, help="输入 PDF 文件路径")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析 PDF 并打印 LLM 抽取结果，不写入数据库",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_indexing_pipeline(args.pdf, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
