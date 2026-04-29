"""
统一配置层 —— pydantic-settings 读取 .env 文件。
所有模块通过 `from core.config import settings` 获取配置。
生产环境不得在源码中保留密钥或默认密码，必须通过环境变量注入。
"""
from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Runtime profile ─────────────────────────────────────────────────────
    app_profile: Literal["demo", "production"] = "demo"
    log_level: str = "INFO"

    # ── LLM: DeepSeek V3（OpenAI 兼容接口）──────────────────────────────────
    deepseek_api_key: str = Field(..., min_length=1)
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    # "deepseek-chat" 在 DeepSeek 官方 API 即指向最新 V3 生产版本
    deepseek_model: str = "deepseek-chat"

    # ── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = Field(..., min_length=1)

    # ── Milvus ───────────────────────────────────────────────────────────────
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "enterprise_docs"

    # ── Embedding（本地 sentence-transformers）───────────────────────────────
    # BAAI/bge-m3: 中英双语 SOTA，1024 维，首次运行自动从 HuggingFace 下载（约 2.3GB）
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # ── 建库参数 ─────────────────────────────────────────────────────────────
    chunk_size: int = 800        # 每块最大字符数
    chunk_overlap: int = 100     # 相邻块重叠字符数
    llm_batch_size: int = 5      # 每批发送给 LLM 的 chunk 数（控制并发/速率）
    llm_temperature: float = 0.1  # 温度越低，实体抽取越确定性
    llm_max_retries: int = 2     # LLM 调用失败最大重试次数

    # ── Agent / 检索参数（第二阶段新增）─────────────────────────────────────────
    retrieval_top_k: int = 5          # Milvus 向量检索返回 Top-K 条
    graph_hop_depth: int = 2          # Neo4j 图拓扑遍历最大跳数（建议 1~3）
    agent_temperature: float = 0.1    # 查询重写 / 路由决策 LLM 温度
    synthesis_temperature: float = 0.3  # 回答合成 LLM 温度（略高以增加流畅性）

    # ── 监控与评估参数 (第五阶段新增) ─────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "EnterpriseKnowledgeBrain"

    # ── HTTP/API 网关 ────────────────────────────────────────────────────────
    cors_allow_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    query_audit_webhook_url: str | None = None

    # ── MCP endpoints ───────────────────────────────────────────────────────
    mcp_api_server_url: str = "http://localhost:8001/mcp"
    mcp_sql_server_url: str = "http://localhost:8002/mcp"
    mcp_graphrag_server_url: str = "http://localhost:8003/mcp"
    mcp_data_analysis_server_url: str = "http://localhost:8004/mcp"
    mcp_web_search_server_url: str = "http://localhost:8005/mcp"

    # ── Redis / checkpoint / cache ──────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Structured data / internal API adapters ─────────────────────────────
    business_sql_database_url: str | None = None
    internal_api_base_url: str | None = None
    internal_api_timeout_seconds: float = 10.0

    # ── Local code model for data analysis worker ───────────────────────────
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value):
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("["):
                return value
            return [item.strip() for item in raw.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _validate_profile(self):
        if self.langchain_tracing_v2 and not self.langchain_api_key:
            raise ValueError("LANGCHAIN_API_KEY is required when LANGCHAIN_TRACING_V2=true")
        if self.app_profile == "production":
            if "*" in self.cors_allow_origins:
                raise ValueError("CORS_ALLOW_ORIGINS must not contain '*' in production")
            if not self.business_sql_database_url:
                raise ValueError("BUSINESS_SQL_DATABASE_URL is required in production")
            if not self.internal_api_base_url:
                raise ValueError("INTERNAL_API_BASE_URL is required in production")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()

# 将 Pydantic 读取到的 LangSmith 配置同步至操作系统的环境变量中
# 因为 langsmith 官方底层依赖直接读取 os.environ，如果不主动注入可能导致 Trace 丢失
if settings.langchain_tracing_v2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    if settings.langchain_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
