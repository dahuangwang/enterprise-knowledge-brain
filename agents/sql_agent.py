"""
agents/sql_agent.py
────────────────────────────────────────────────────────────────────────────
结构化查询 Agent (Phase 4 MVP)
负责将自然语言用户指令转化为 SQLite Query 并自动在底层沙箱中查询对应数据。
"""
from typing import Dict, Any, Optional

from openai import OpenAI

from core.logger import get_logger
from core.config import settings
from agents.prompts import (
    SQLAgentOutput,
    SQL_AGENT_SYSTEM,
    build_sql_agent_prompt
)
from agents.graphrag_agent import _llm_json_call
from tools.sql_executor import execute_sandbox_sql

logger = get_logger(__name__)


def run_sql_agent(instruction: str, llm_client: OpenAI) -> Dict[str, Any]:
    """
    SQL Agent 单体运行入口，接收任务规划器分配的 instruction。
    """
    logger.info(f"[SQL Agent] 收到数据查询指令: {instruction[:50]}...")
    
    # 这一步将指代和意图转换为 JSON 格式的结构化 Query
    data = _llm_json_call(
        client=llm_client,
        system_prompt=SQL_AGENT_SYSTEM,
        user_prompt=build_sql_agent_prompt(instruction),
        temperature=0.1,  # 使用极低温度确保 SQL 确切性
    )
    
    if data is None:
        logger.error("[SQL Agent] LLM 生成 SQL 任务失败/超时。")
        return {"error": "LLM 故障"}
        
    try:
        parsed_out = SQLAgentOutput(**data)
        
        sql_query = parsed_out.sql_query
        reasoning = parsed_out.reasoning
        
        logger.info(f"[SQL Agent] 推理完毕: {reasoning}")
        logger.info(f"[SQL Agent] 将执行 SQL 语句: {sql_query}")
        
        # 将语句送进底层沙箱执行
        exec_result: str = execute_sandbox_sql(sql_query)
        logger.info(f"[SQL Agent] 数据库成功返回，大小为 {len(exec_result)} 字节。")
        
        return {
            "reasoning": reasoning,
            "sql_query": sql_query,
            "answer": f"（通过内部结构化 DB 获取）\n执行 SQL: {sql_query}\n\n执行结果：\n{exec_result}",
            "error": None
        }
        
    except Exception as e:
        logger.error(f"[SQL Agent] 意外异常 Schema 不匹配或者底层崩溃: {e}")
        return {"error": str(e), "answer": f"SQL 处理管道出错: {e}"}
