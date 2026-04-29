"""
tools/sql_executor.py
────────────────────────────────────────────────────────────────────────────
结构化数据查询适配器。
demo profile 使用 SQLite in-memory 演示数据；production profile 使用
BUSINESS_SQL_DATABASE_URL 指向真实只读数据源。
"""
import sqlite3
from typing import Any, List, Dict

from sqlalchemy import create_engine, text

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# 单例内存数据库，确保生命周期内数据存在
_conn = sqlite3.connect(":memory:", check_same_thread=False)
_conn.row_factory = sqlite3.Row  # 允许按列名索引

def _init_dummy_data():
    """初始化 demo profile 使用的演示数据表"""
    cursor = _conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS financial_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT,
            department TEXT,
            investment_amount REAL,
            revenue_q1 REAL,
            status TEXT
        )
    """)
    
    # 填充假数据
    dummy_data = [
        ("招商银行极速开户系统", "金融科技部", 500.0, 150.0, "运行中"),
        ("智能风控决策平台", "风险控制部", 1200.0, 300.5, "研发中"),
        ("中小微企业贷款系统", "企业信贷部", 850.0, 420.0, "运行中"),
        ("AI大模型智能客服", "客户服务部", 300.0, 80.0, "测试中"),
        ("同城双活灾备系统", "IT基础设施部", 2000.0, 0.0, "建设中")
    ]
    
    cursor.executemany(
        """
        INSERT INTO financial_reports (
            project_name, department, investment_amount, revenue_q1, status
        ) VALUES (?, ?, ?, ?, ?)
        """,
        dummy_data
    )
    _conn.commit()
    logger.info("SQL demo adapter 初始化完成，`financial_reports` 演示数据已就绪。")

if settings.app_profile == "demo":
    _init_dummy_data()


def _is_readonly_select(query: str) -> bool:
    clean_query = query.strip()
    if not clean_query.upper().startswith("SELECT"):
        return False
    forbidden = (";", "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH", "PRAGMA")
    upper = clean_query.upper()
    return not any(token in upper for token in forbidden)


def _execute_demo_sql(query: str) -> str:
    cursor = _conn.cursor()
    cursor.execute(query)
    rows: List[sqlite3.Row] = cursor.fetchmany(50)
    return _format_rows(rows)


def _execute_production_sql(query: str) -> str:
    if not settings.business_sql_database_url:
        return "执行失败: production profile 必须配置 BUSINESS_SQL_DATABASE_URL。"

    engine = create_engine(settings.business_sql_database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchmany(50)
    return _format_rows(rows)


def _format_rows(rows) -> str:
    if not rows:
        return "执行成功，查询结果为空。"

    results_list = []
    for row in rows:
        if hasattr(row, "_mapping"):
            row_dict = dict(row._mapping)
        else:
            keys = row.keys()
            row_dict = {k: row[k] for k in keys}
        results_list.append(str(row_dict))

    return "执行成功，查询结果:\n" + "\n".join(results_list)

def execute_sandbox_sql(query: str) -> str:
    """
    向后兼容的工具 API。名称保留，但运行时会根据 APP_PROFILE 选择适配器。
    """
    clean_query = query.strip()
    if not _is_readonly_select(clean_query):
        return "执行失败: 安全策略限制，此沙箱仅支持 SELECT 即读操作。"
        
    try:
        if settings.app_profile == "demo":
            return _execute_demo_sql(clean_query)
        return _execute_production_sql(clean_query)
    except (sqlite3.OperationalError, Exception) as op_e:
        logger.warning(f"SQL执行出现错误: {op_e}. Query: {clean_query}")
        return f"查询执行失败: 语句存在语法错误或引用了不存在的字段: {op_e}"
