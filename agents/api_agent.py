"""
agents/api_agent.py
────────────────────────────────────────────────────────────────────────────
企业级 API 网关操作 Agent (Phase 4 MVP)
负责基于可用接口清单寻址，并解析需要的传参。
"""
from typing import Dict, Any, Optional

from openai import OpenAI

from core.logger import get_logger
from core.config import settings
from agents.prompts import (
    APIAgentOutput,
    API_AGENT_SYSTEM,
    build_api_agent_prompt
)
from agents.graphrag_agent import _llm_json_call
from tools.internal_apis import call_internal_api

logger = get_logger(__name__)


def run_api_agent(instruction: str, llm_client: OpenAI) -> Dict[str, Any]:
    """
    API Agent 单体运行入口，接收任务规划器分配的 instruction。
    """
    logger.info(f"[API Agent] 收到外部接口调用指令: {instruction[:50]}...")
    
    # 指代与接口名称的 JSON 生成
    data = _llm_json_call(
        client=llm_client,
        system_prompt=API_AGENT_SYSTEM,
        user_prompt=build_api_agent_prompt(instruction),
        temperature=0.1,  # 确保寻找正确的 endpoint
    )
    
    if data is None:
        logger.error("[API Agent] LLM 生成 API 请求参数失败/超时。")
        return {"error": "LLM 故障"}
        
    try:
        parsed_out = APIAgentOutput(**data)
        
        endpoint = parsed_out.endpoint
        params = parsed_out.params
        reasoning = parsed_out.reasoning
        
        logger.info(f"[API Agent] 推理分析: {reasoning}")
        logger.info(f"[API Agent] 即将呼救端点: {endpoint} | 参数: {params}")
        
        # 将构造完成的请求交付内部 API 聚合层
        exec_result: str = call_internal_api(endpoint, params)
        logger.info("[API Agent] 业务 API 处理完成。")
        
        return {
            "reasoning": reasoning,
            "endpoint": endpoint,
            "params": params,
            "answer": f"（通过呼叫内部接口端点 {endpoint} 获取）\n系统回传文：\n{exec_result}",
            "error": None
        }
        
    except Exception as e:
        logger.error(f"[API Agent] Endpoint/Schema 发生严重解析或运行时异常: {e}")
        return {"error": str(e), "answer": f"API 处理管道出错: {e}"}
