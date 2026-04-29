"""
agents/prompts.py
────────────────────────────────────────────────────────────────────────────
【新增文件】Agent 层 Prompt 模板 + 结构化输出 Schema（Pydantic v2）

包含三类 Prompt（全部集中在此，不硬编码在 Agent 逻辑文件里）：
  1. 查询重写（Query Rewrite）: 口语化 → 标准化 + 实体识别 + 意图描述
  2. 路由决策（Route Decision）: 决定 vector_search | graph_search | hybrid_search
  3. 回答合成（Synthesis）: 检索结果 → 最终用户答案（忠实性约束）

所有 Prompt 通过 build_*_prompt() 构造函数填充，避免硬编码在 Agent 节点里。
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

# pydantic是python的类型注解库，用于定义数据结构
# BaseModel是pydantic中的基类，用于定义数据模型
# Field是pydantic中的函数，用于定义字段的约束条件
from pydantic import BaseModel, Field


# ============================================================================
# § 1  结构化输出 Schema（所有 LLM 调用的输入/输出均显式定义）
# ============================================================================

RouteType = Literal["vector_search", "graph_search", "hybrid_search"]


class NormalizedQuery(BaseModel):
    """
    查询重写结果：将口语化问题标准化。
    由 rewrite_query_node 生成，后续节点直接读取字段。
    """

    normalized: str = Field(
        ...,
        description="标准化后的查询语句：消除歧义，将简称补全为实体全称",
        min_length=1,
        max_length=500,
    )
    entities: List[str] = Field(
        default_factory=list,
        description="从问题中识别出的关键实体名称列表（公司/项目名），用于图谱检索",
    )
    intent: str = Field(
        ...,
        description="用户查询意图的一句话描述，不超过 60 字",
        max_length=120,
    )


class RoutingDecision(BaseModel):
    """
    路由决策结果：选择检索策略。
    由 route_query_node 生成，通过条件边控制后续分支。
    """

    route: RouteType = Field(
        ...,
        description=(
            "检索策略:\n"
            "  vector_search  — 语义相似度检索，适合主题/概念类问题\n"
            "  graph_search   — 实体关系图谱遍历，适合精确关系类问题\n"
            "  hybrid_search  — 两者结合，适合复杂综合类问题"
        ),
    )
    reasoning: str = Field(
        ...,
        description="选择该路由策略的理由，不超过 100 字",
        max_length=200,
    )


class SynthesisInput(BaseModel):
    """回答合成的输入（Agent 内部数据传输用，不暴露给 LLM）"""

    question: str
    vector_context: str = "（无向量检索结果）"
    graph_context: str = "（无图谱检索结果）"


class SubTask(BaseModel):
    """任务拆解的子任务定义"""
    task_id: str = Field(..., description="任务ID，如 step_1")
    target_agent: Literal["graphrag_agent", "sql_agent", "api_agent"] = Field(
        ..., description="目标执行 Agent"
    )
    instruction: str = Field(..., description="给该 Agent 的具体执行指令")
    dependencies: List[str] = Field(
        default_factory=list, description="依赖的先决条件 task_id 列表"
    )
# PlannerDecision用于对大模型的输入进行约束
class PlannerDecision(BaseModel):
    """Planner Agent 的整体规划输出"""
    # Field是pydantic中的函数，用于定义字段的约束条件
    reasoning: str = Field(..., description="分析用户意图及选用 Agent 的逻辑")
    plan: List[SubTask] = Field(default_factory=list, description="子任务列表")


class SQLAgentOutput(BaseModel):
    """SQL Agent 的输出结构"""
    reasoning: str = Field(..., description="对用户意图及选择生成该SQL的分析")
    sql_query: str = Field(..., description="生成可以直接运行的 SQLite SQL语句")


class APIAgentOutput(BaseModel):
    """API Agent 的输出结构"""
    reasoning: str = Field(..., description="对调用哪个接口的逻辑分析")
    endpoint: str = Field(..., description="目标 API 端点名称")
    params: Dict[str, Any] = Field(default_factory=dict, description="调用该 API 所需的参数字典")


# ============================================================================
# § 2  Prompt 模板常量
# ============================================================================

# ── 2.1 查询重写 ──────────────────────────────────────────────────────────────

QUERY_REWRITE_SYSTEM: str = """\
你是企业知识问答系统的查询分析专家，擅长处理中国上市公司相关的业务查询。

你的任务是将用户口语化的提问转化为标准化的内部查询格式，输出三个字段：
1. normalized：标准化查询语句（消除歧义，将简称补全为正式全称）
2. entities：查询中涉及的关键实体名称列表（公司/项目/人物名等），用于图谱检索
3. intent：一句话描述用户意图（不超过 60 字）

规则：
- 实体名使用正式全称（"招行" → "招商银行股份有限公司"）
- 不添加用户未提及的信息，不推断
- 只输出合法 JSON，禁止 markdown 标记或解释文字\
"""

QUERY_REWRITE_TEMPLATE: str = """\
用户问题：{question}

请以如下 JSON 格式输出（禁止其他内容）：
{{
  "normalized": "标准化后的查询语句",
  "entities": ["实体名称1", "实体名称2"],
  "intent": "用户意图一句话描述"
}}\
"""

# ── 2.2 路由决策 ──────────────────────────────────────────────────────────────

ROUTE_SYSTEM: str = """\
你是企业知识检索系统的路由决策专家。根据标准化查询和意图，选择最合适的检索策略：

- vector_search（向量检索）：
    适用：主题概念类、需要语义理解的问题
    典型示例："公司的数字化转型战略", "XX公司经营情况概述"

- graph_search（图谱检索）：
    适用：精确实体关系类问题，涉及"谁投资了谁"、"哪些子公司"、"持股比例"
    典型示例："招商银行持股哪些公司", "A项目由哪家公司投资"

- hybrid_search（混合检索）：
    适用：既需要精确关系又需要语义补充的复杂问题
    典型示例："评估A项目对B公司资金链的影响", "分析公司投资组合风险"

只输出合法 JSON，禁止任何解释。\
"""

ROUTE_TEMPLATE: str = """\
标准化查询：{normalized}
识别实体：{entities}
用户意图：{intent}

请以如下 JSON 格式输出（禁止其他内容）：
{{
  "route": "vector_search | graph_search | hybrid_search",
  "reasoning": "选择该策略的具体理由（不超过100字）"
}}\
"""

# ── 2.3 回答合成 ──────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM: str = """\
你是严谨的企业知识问答助手，专注于基于企业知识库精准回答用户问题。

核心原则（必须严格遵守）：
1. 【忠实性】只基于下方提供的检索内容作答，严禁使用预训练知识补充或推断
2. 【拒绝幻觉】若检索内容不足以回答，必须明确输出：
   "基于企业当前知识库，无法得出结论。"
3. 【专业性】回答简洁专业，直接切入要点，避免空话套话
4. 【可溯源】引用具体数据时标注来源（如"根据向量检索第X条"或"根据图谱关系"）\
"""

SYNTHESIS_TEMPLATE: str = """\
用户问题：{question}

═══ 向量检索结果（语义相似内容）═══
{vector_context}

═══ 图谱检索结果（实体关系数据）═══
{graph_context}

请基于以上检索内容，给出精准、客观的回答：\
"""

# ── 2.4 任务规划 (Planner) ────────────────────────────────────────────────────────

PLANNER_SYSTEM: str = """\
你是一个企业级知识大脑的首席任务规划师 (Chief Planner)。你的任务是分析用户的输入，将其拆解为具体可执行的子任务，并准确路由给最合适的专业领域 Agent。

## 可用执行器 (Available Agents)
1. `graphrag_agent`: 负责处理企业非结构化文档、报告和实体关系的查询（基于 Neo4j 图谱和 Milvus 向量混合检索）。
2. `sql_agent`: 负责处理精确的结构化数据统计、财务指标和报表查询（基于受限沙箱 DB）。
3. `api_agent`: 负责查询企业内部系统的实时状态或执行系统侧操作（如：获取实时审批流、查询员工最新工单等）。

## 规划规则 (Planning Rules)
1. **意图拆解**：如果用户的问题涉及多个维度（例如：“对比招商银行投资的科技项目与今年 Q1 的财务营收”），你必须将其拆解为多个并行或串行的任务。
2. **精准路由**：不要将结构化统计（求和、极值、确切数字）交给 `graphrag_agent`，请交给 `sql_agent`。
3. **参数提取**：为每个被调用的 Agent 准备清晰的执行指令和必要的参数。

严格输出包含 reasoning 和 plan 的 JSON 格式，禁止输出 markdown 代码块标记以外的任何内容。\
"""

PLANNER_USER_TEMPLATE: str = """\
用户问题：{question}

请根据规则生成 JSON 格式的任务执行计划：\
"""


# ── 2.5 SQL 执行阶段 (SQL Agent) ──────────────────────────────────────────────────

SQL_AGENT_SYSTEM: str = """\
你是一个拥有高级 SQL 技能的数据分析专家。你的任务是根据用户的需求，自动生成用于查询企业经营报表的 SQLite SQL 语句。

沙箱中有一张财务数据表 `financial_reports`，表结构如下：
- `id` (INTEGER PRIMARY KEY)
- `project_name` (TEXT) - 项目名称
- `department` (TEXT) - 所属部门
- `investment_amount` (REAL) - 投资金额（万元）
- `revenue_q1` (REAL) - Q1营收（万元）
- `status` (TEXT) - 运营状态

你需要严格按照用户指令生成安全、只读（SELECT）的 SQLite 查询语句。输出必须包含 reasoning 和 sql_query 的 JSON 格式，禁止输出 markdown 标记。\
"""

SQL_AGENT_TEMPLATE: str = """\
用户指令：{instruction}

请结合表结构分析需求，并按 JSON 格式输出：
{{
  "reasoning": "简要分析查询逻辑",
  "sql_query": "SELECT ..."
}}\
"""

# ── 2.6 API 集成阶段 (API Agent) ──────────────────────────────────────────────────

API_AGENT_SYSTEM: str = """\
你是一个熟练对接企业内部系统的核心开发工程师。你的任务是根据指令寻找并调用合适的内部接口查询实时状态。

当前可调用的内部 Mock 接口列表：
1. `get_project_approval_status` - 入参: `project_name` (字符串)。用于获取项目的法务/财务实时审批状态。
2. `get_department_budget` - 入参: `department` (字符串)。用于获取特性部门的可用总预算余额。
3. `get_employee_tickets` - 入参: `employee_name` (字符串)。用于获取特定员工处理的最新 OA 工单情况。

请选择最合适的 API，并以严格的 JSON 格式输出，禁止输出 markdown 标记。\
"""

API_AGENT_TEMPLATE: str = """\
用户指令：{instruction}

请分析指令需要的业务接口，以 JSON 格式输出：
{{
  "reasoning": "解释为什么调用此接口以及参数提取依据",
  "endpoint": "get_project_approval_status",
  "params": {{"project_name": "XX科技公司"}}
}}\
"""


# ============================================================================
# § 3  Prompt 构造函数
# ============================================================================

def build_rewrite_prompt(question: str) -> str:
    """构建查询重写的 user prompt"""
    return QUERY_REWRITE_TEMPLATE.format(question=question)


def build_route_prompt(normalized: NormalizedQuery) -> str:
    """构建路由决策的 user prompt"""
    entities_str = (
        "、".join(normalized.entities)
        if normalized.entities
        else "（未识别到实体）"
    )
    return ROUTE_TEMPLATE.format(
        normalized=normalized.normalized,
        entities=entities_str,
        intent=normalized.intent,
    )


def build_synthesis_prompt(inp: SynthesisInput) -> str:
    """构建回答合成的 user prompt"""
    return SYNTHESIS_TEMPLATE.format(
        question=inp.question,
        vector_context=inp.vector_context,
        graph_context=inp.graph_context,
    )


def build_planner_prompt(question: str) -> str:
    """构建规划 Agent 的 user prompt"""
    return PLANNER_USER_TEMPLATE.format(question=question)


def build_sql_agent_prompt(instruction: str) -> str:
    """构建 SQL Agent 的 user prompt"""
    return SQL_AGENT_TEMPLATE.format(instruction=instruction)


def build_api_agent_prompt(instruction: str) -> str:
    """构建 API Agent 的 user prompt"""
    return API_AGENT_TEMPLATE.format(instruction=instruction)
