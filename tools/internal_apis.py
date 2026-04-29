"""
tools/internal_apis.py
────────────────────────────────────────────────────────────────────────────
企业内部 API 调用适配器。
demo profile 使用本地演示实现；production profile 调用 INTERNAL_API_BASE_URL。
"""
from typing import Any, Callable, Dict, cast

import httpx

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# 模拟系统接口的具体实现
# ============================================================================

def _get_project_approval_status(project_name: str) -> str:
    if "招商" in project_name or "极速" in project_name:
        return f"【法务待审批】 项目：{project_name} 的合规审计尚未走完，预计 1 个工作日内结束。"
    elif "风控" in project_name:
        return f"【审批已全部通过】 项目：{project_name} 流程完结，可随时推进。"
    return f"【未找到项目审批单】 未在内部系统中查找到名称匹配 '{project_name}' 的在途单据。"

def _get_department_budget(department: str) -> str:
    budgets = {
        "金融科技部": "1500万净结余",
        "风险控制部": "300万净结余 (即将超标)",
        "IT基础设施部": "可用经费充足",
    }
    for k, v in budgets.items():
        if k in department:
            return f"部门 '{department}' 财务核对结果：当前总预算 {v}。"
    return f"部门 '{department}' 的预算池状态暂不可见或名称不匹配。"

def _get_employee_tickets(employee_name: str) -> str:
    if "李" in employee_name:
        return f"员工 '{employee_name}' 当前有 3 个未处理紧急 Bug 工单。"
    return f"员工 '{employee_name}' 当前清闲，所有分配的工单均已结单。"


# 字典映射分发器
API_ROUTER = {
    "get_project_approval_status": _get_project_approval_status,
    "get_department_budget": _get_department_budget,
    "get_employee_tickets": _get_employee_tickets,
}

# ============================================================================
# 暴露给 api_agent 利用的通用聚合函数
# ============================================================================

def call_internal_api(endpoint: str, params: Dict[str, Any]) -> str:
    """
    通用内部系统 API 测试调用口。
    Args:
        endpoint: 明确具体的 API 端点名称。
        params: 传入给这个接口的关键字参数。
    Returns:
        从对应内部系统返回来的响应文本。
    """
    logger.info(f"[API Gateway:{settings.app_profile}] endpoint={endpoint}, params={params}")
    
    if endpoint not in API_ROUTER:
        return f"接口调用失败: 未注册的端点 '{endpoint}'。请从许可 API 列表中重新挑选。"

    if settings.app_profile == "production":
        if not settings.internal_api_base_url:
            return "接口调用失败: production profile 必须配置 INTERNAL_API_BASE_URL。"
        try:
            url = f"{settings.internal_api_base_url.rstrip('/')}/{endpoint}"
            response = httpx.post(
                url,
                json=params,
                timeout=settings.internal_api_timeout_seconds,
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"真实内部 API 调用失败: {e}")
            return f"接口调用异常，内部服务不可用: {e}"

    func = cast(Callable[..., str], API_ROUTER[endpoint])
    try:
        logger.warning("正在使用 demo profile 的内部 API 演示适配器，不代表真实业务系统状态。")
        return func(**params)
    except TypeError as e:
        logger.warning(f"参数传递错误对于端点 {endpoint}: {e}")
        return f"接口调用失败: 参数与所请求的接口 '{endpoint}' 定义不一致: {e}"
    except Exception as e:
        logger.error(f"接口层发生了内部严重故障: {e}")
        return f"接口调用异常，内部服务不可用: {e}"
