import hashlib
from typing import Optional
from core.logger import get_logger
from utils.redis_client import get_redis_client

logger = get_logger("semantic_cache")

# Layer 1: 精确匹配缓存 (Exact Match Cache)

async def get_exact_cache(user_id: str, query: str) -> Optional[str]:
    """
    获取精确匹配的缓存（Layer 1）
    使用 user_id 和 query 生成 Hash 作为 Key。
    """
    try:
        redis_client = await get_redis_client()
        query_hash = hashlib.sha256(f"{user_id}:{query}".encode('utf-8')).hexdigest()
        cache_key = f"cache:exact:{query_hash}"
        
        cached_result = await redis_client.get(cache_key)
        if cached_result:
            logger.info(f"⚡ [Cache Hit] 命中精确匹配缓存! Key: {cache_key}")
            return cached_result
    except Exception as e:
        logger.error(f"访问 Redis 缓存失败: {e}")
        
    return None

async def set_exact_cache(user_id: str, query: str, response: str, ttl_seconds: int = 86400):
    """
    设置精确匹配缓存
    """
    try:
        redis_client = await get_redis_client()
        query_hash = hashlib.sha256(f"{user_id}:{query}".encode('utf-8')).hexdigest()
        cache_key = f"cache:exact:{query_hash}"
        
        await redis_client.setex(cache_key, ttl_seconds, response)
        logger.info(f"💾 [Cache Set] 成功写入缓存. Key: {cache_key}, TTL: {ttl_seconds}s")
    except Exception as e:
        logger.error(f"写入 Redis 缓存失败: {e}")
