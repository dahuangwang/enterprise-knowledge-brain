"""
审计事件发布适配器。

demo profile 只写结构化日志；production profile 通过 QUERY_AUDIT_WEBHOOK_URL
对接企业侧日志、Kafka Bridge 或审计网关。
"""
from __future__ import annotations

from typing import Any

import httpx

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


async def publish_query_audit_event(
    *,
    user_id: str,
    elapsed_ms: int,
    question: str,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "event_type": "enterprise_query",
        "profile": settings.app_profile,
        "user_id": user_id,
        "elapsed_ms": elapsed_ms,
        "question": question,
        "status": status,
        "extra": extra or {},
    }

    if not settings.query_audit_webhook_url:
        logger.info("[AuditEvent:%s] %s", settings.app_profile, payload)
        return

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(settings.query_audit_webhook_url, json=payload)
            response.raise_for_status()
    except Exception as exc:
        logger.error("审计事件发布失败: %s", exc)
