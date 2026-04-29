from typing import Optional
import redis.asyncio as redis

from core.config import settings
from core.logger import get_logger

logger = get_logger("redis_client")

_redis_pool: Optional[redis.ConnectionPool] = None

def get_redis_pool() -> redis.ConnectionPool:
    global _redis_pool
    if _redis_pool is None:
        logger.info(f"Initializing Redis Connection Pool at {settings.redis_url}")
        _redis_pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=50
        )
    return _redis_pool

async def get_redis_client() -> redis.Redis:
    """获取异步 Redis 客户端"""
    pool = get_redis_pool()
    return redis.Redis(connection_pool=pool)

async def close_redis_pool():
    global _redis_pool
    if _redis_pool is not None:
        logger.info("Closing Redis Connection Pool")
        await _redis_pool.disconnect()
        _redis_pool = None
