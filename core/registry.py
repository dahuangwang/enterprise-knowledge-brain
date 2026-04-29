import json
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from core.logger import get_logger
from utils.redis_client import get_redis_client

logger = get_logger("registry")

class AgentCard(BaseModel):
    """AgentCard (智能体名片) 定义了 Agent 的标准身份、能力与状态描述。"""
    agent_id: str = Field(..., description="全局唯一 ID (如: sql_worker_01)")
    agent_name: str = Field(..., description="智能体显示名称 (如: SQL_Worker)")
    description: str = Field(..., description="对大语言模型友好的能力描述，用于路由决策")
    protocol: str = Field(default="mcp/streamable", description="通信协议类型")
    endpoint: str = Field(..., description="服务入口 URL (用于发起连接)")
    capabilities: List[str] = Field(default_factory=list, description="能力标签 (如: ['database', 'finance'])")
    mcp_tools_summary: List[Dict[str, Any]] = Field(default_factory=list, description="暴露的工具名称及简述")
    status: str = Field(default="online", description="服务状态 (online/offline/busy)")
    last_heartbeat: float = Field(default_factory=time.time, description="最后一次心跳的时间戳")

class RegistryClient:
    """AgentCard 注册中心客户端，基于 Redis 实现服务注册与发现。"""
    
    PREFIX = "agentcard:"
    
    @classmethod
    async def register(cls, card: AgentCard, ttl: int = 30) -> bool:
        """注册或更新 AgentCard"""
        try:
            redis = await get_redis_client()
            key = f"{cls.PREFIX}{card.agent_id}"
            card.last_heartbeat = time.time()
            card.status = "online"
            
            # 使用 JSON 序列化
            data = card.model_dump_json()
            await redis.set(key, data, ex=ttl)
            logger.info(f"成功注册 AgentCard: {card.agent_id} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"注册 AgentCard 失败: {e}")
            return False

    @classmethod
    async def heartbeat(cls, agent_id: str, ttl: int = 30) -> bool:
        """发送心跳，续期 TTL 并更新 last_heartbeat"""
        try:
            redis = await get_redis_client()
            key = f"{cls.PREFIX}{agent_id}"
            
            data_str = await redis.get(key)
            if not data_str:
                logger.warning(f"心跳失败：找不到 AgentCard {agent_id}，可能已过期注销。")
                return False
                
            card_dict = json.loads(data_str)
            card_dict["last_heartbeat"] = time.time()
            
            await redis.set(key, json.dumps(card_dict), ex=ttl)
            logger.debug(f"心跳成功: {agent_id}")
            return True
        except Exception as e:
            logger.error(f"心跳更新失败: {e}")
            return False

    @classmethod
    async def discover(cls) -> List[AgentCard]:
        """发现所有活跃的 AgentCard"""
        try:
            redis = await get_redis_client()
            # 获取所有以 PREFIX 开头的 keys
            keys = await redis.keys(f"{cls.PREFIX}*")
            
            if not keys:
                return []
                
            # 批量获取内容
            # redis keys 返回的是 list，mget 也可以接受 list
            data_list = await redis.mget(keys)
            
            cards = []
            for data_str in data_list:
                if data_str:
                    try:
                        cards.append(AgentCard.model_validate_json(data_str))
                    except Exception as parse_e:
                        logger.error(f"解析 AgentCard 失败: {parse_e}")
                        
            # 根据 agent_name 排序一下，保证路由提示词顺序稳定
            cards.sort(key=lambda c: c.agent_name)
            return cards
        except Exception as e:
            logger.error(f"服务发现失败: {e}")
            return []

    @classmethod
    async def unregister(cls, agent_id: str) -> bool:
        """主动注销 AgentCard"""
        try:
            redis = await get_redis_client()
            key = f"{cls.PREFIX}{agent_id}"
            await redis.delete(key)
            logger.info(f"成功注销 AgentCard: {agent_id}")
            return True
        except Exception as e:
            logger.error(f"注销 AgentCard 失败: {e}")
            return False
