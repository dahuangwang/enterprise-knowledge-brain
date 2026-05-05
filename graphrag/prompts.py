"""
GraphRAG Prompt 模板 + 所有结构化输出 Schema（Pydantic v2）。

原则：
- 所有 Prompt 集中在此文件，不散落在路由或 indexer 里
- Schema 显式定义，字段含义清晰
- 提供 JSON 格式示例，帮助 LLM 严格输出

关系类型字典（可扩展）:
    RELATION_TYPES = {"投资", "持股", "子公司", "参股", "合作", "管理", "发行"}
实体类型字典（可扩展）:
    ENTITY_TYPES = {"公司", "项目", "人物", "地区", "产品", "基金"}
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# 实体与关系类型常量（可直接扩展）
# ============================================================================

ENTITY_TYPES: set[str] = {"公司", "项目", "人物", "地区", "产品", "基金"}
RELATION_TYPES: set[str] = {"投资", "持股", "子公司", "参股", "合作", "管理", "发行"}


# ============================================================================
# 显式 Schema 定义
# ============================================================================

class Entity(BaseModel):
    """知识图谱节点 —— 表示一个具名实体"""

    name: str = Field(
        ...,
        description="实体全称，与原文完全一致，不缩写",
        min_length=1,
        max_length=200,
    )
    entity_type: str = Field(
        ...,
        description=f"实体类型，必须是以下之一: {' | '.join(sorted(ENTITY_TYPES))}",
    )
    description: Optional[str] = Field(
        None,
        description="实体的简短描述，来自原文，不超过 100 字",
        max_length=200,
    )
    attributes: dict = Field(
        default_factory=dict,
        description="实体的相关属性键值对（如注册资本、成立日期、法定代表人等）",
    )

    def validate_type(self) -> bool:
        return self.entity_type in ENTITY_TYPES


class Relation(BaseModel):
    """知识图谱有向边 —— 表示两个实体之间的关系"""

    source: str = Field(..., description="关系起点实体名称，必须与某个 Entity.name 对应")
    target: str = Field(..., description="关系终点实体名称，必须与某个 Entity.name 对应")
    relation_type: str = Field(
        ...,
        description=f"关系类型，必须是以下之一: {' | '.join(sorted(RELATION_TYPES))}",
    )
    evidence: str = Field(
        ...,
        description="支持该关系的原文句子（直接引用，不超过 150 字）",
        max_length=300,
    )
    amount: Optional[str] = Field(
        None,
        description="金额、股比等定量信息，仅投资/持股/参股关系需填写，如 '3.5亿元' 或 '15%'",
        max_length=50,
    )

    def validate_type(self) -> bool:
        return self.relation_type in RELATION_TYPES


class ExtractionResult(BaseModel):
    """
    单个 chunk 的 LLM 抽取结果。
    entities 和 relations 均可为空列表（文本无相关信息时）。
    """

    entities: List[Entity] = Field(
        default_factory=list,
        description="从文本中抽取的实体列表",
    )
    relations: List[Relation] = Field(
        default_factory=list,
        description="从文本中抽取的关系列表，引用 entities 中的名称",
    )
    # property是python的装饰器，表示将方法转换为属性
    @property
    def is_empty(self) -> bool:
        return not self.entities and not self.relations

    def stats(self) -> str:
        return f"{len(self.entities)} 实体, {len(self.relations)} 关系"


# ============================================================================
# Prompt 模板
# ============================================================================

EXTRACTION_SYSTEM_PROMPT: str = (
    "你是一位专业的企业知识图谱构建专家，专注于从中国上市公司年报、公告和研究报告中"
    "精准抽取实体、属性和关系信息。\n"
    "你始终严格输出合法 JSON，不添加任何 markdown 标记、注释或解释文字。\n"
    "当文本中没有符合要求的信息时，输出空列表，不要捏造内容。"
)

# 使用 .format(page=, chunk_idx=, text=) 填充
EXTRACTION_USER_TEMPLATE: str = """\
请从以下企业年报文本片段中，提取所有符合条件的实体、属性和关系。

═══ 提取规则 ═══
1. 【实体类型】只能是: {entity_types}
2. 【关系类型】只能是: {relation_types}
3. 【属性抽取】若文本中提及实体的独立数值、状态（如注册资本、成立日期），请提取到 attributes 中。
4. 【局部指代消解】文本切片中若出现“该公司”、“本集团”，请根据上下文推断其全称，禁止将“该公司”作为一个独立的实体抽取。实体 name 必须与原文完整名称一致。
5. 只提取原文明确陈述的信息，严禁推断或补全
6. evidence 必须是原文的直接引用，不得改写
7. 若无符合条件的实体或关系，返回对应空列表

═══ 文本片段（来源: 第 {page} 页，块编号 {chunk_idx}）═══
{text}

═══ 输出格式（严格 JSON，禁止其他内容）═══
{{
  "entities": [
    {{
      "name": "招商银行股份有限公司", 
      "entity_type": "公司", 
      "description": "全国性股份制商业银行",
      "attributes": {{"注册资本": "252亿元", "成立年份": "1987"}}
    }},
    {{
      "name": "科技创新产业园区项目", 
      "entity_type": "项目", 
      "description": "公司重点投资项目",
      "attributes": {{}}
    }}
  ],
  "relations": [
    {{
      "source": "招商银行股份有限公司",
      "target": "科技创新产业园区项目",
      "relation_type": "投资",
      "evidence": "本行（招商银行股份有限公司）于2023年向科技创新产业园区项目投入资金3.5亿元",
      "amount": "3.5亿元"
    }}
  ]
}}
"""

# 预填充常量，调用时只需 .format(page=, chunk_idx=, text=)
_FILLED_TEMPLATE = EXTRACTION_USER_TEMPLATE.replace(
    "{entity_types}", " | ".join(sorted(ENTITY_TYPES))
).replace(
    "{relation_types}", " | ".join(sorted(RELATION_TYPES))
)


def build_extraction_prompt(page: int, chunk_idx: int, text: str) -> str:
    """构建单个 chunk 的实体关系抽取 prompt"""
    return _FILLED_TEMPLATE.format(
        page=page,
        chunk_idx=chunk_idx,
        text=text,
    )
