"""
agents/planner_agent.py
────────────────────────────────────────────────────────────────────────────
【第三阶段】基于 MCP 协议的全新 Planner Agent (Client 端)

功能：
  不再硬编码 import 任何底层 Agent (SQL/API/GraphRAG)。
  利用 MCP 协议动态建立 SSE 长连接，抓取外部暴露的 Tools。
  依靠 LLM 原生的 Function Calling (ReAct 循环) 自动路由派发与汇总。
"""
from __future__ import annotations

import json
import asyncio
import operator
from typing import Any, Dict, List, Annotated, Sequence, TypedDict, Literal

from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.redis import AsyncRedisSaver

from mcp import ClientSession

from core.config import settings
from core.logger import get_logger
from core.registry import RegistryClient, AgentCard
from utils.redis_client import get_redis_client
from utils.semantic_cache import get_exact_cache, set_exact_cache

logger = get_logger("planner_mcp_client")

# ============================================================================
# § 1  全局状态与系统提示词 (Supervisor State)
# ============================================================================

class AgentState(TypedDict):
    """LangGraph 全局状态流转字典"""
    messages: Annotated[Sequence[BaseMessage], operator.add] # 必须有，用来存储对话历史和每次的运行结果
    next_workers: List[str] # 必须有，用来存储接下来要执行的任务
    completed_tasks: Annotated[List[str], operator.add] # 必须有，用来存储已经完成的任务
    question: str # 必须有，用来存储用户的原始问题
    mcp_sessions: Dict[str, ClientSession] # 必须有，用来存储 MCP sessions
    mcp_tools: List[Dict[str, Any]] # 必须有，用来存储 MCP tools
    user_id: str
    active_agents: List[AgentCard] # 动态获取的活跃 Agent 列表

# 【修正】添加了 Report_Generator_Worker 的描述
def get_dynamic_supervisor_prompt(active_agents: List[AgentCard]) -> str:
    agent_descriptions = []
    for i, agent in enumerate(active_agents, 1):
        agent_descriptions.append(f"{i}. \"{agent.agent_name}\": {agent.description}")
        
    agents_str = "\n".join(agent_descriptions)
    next_idx = len(active_agents) + 1
    
    return f"""\
你是一个超级企业知识大脑调度中枢 (Supervisor)。
你手中掌握着企业各个业务线的动态工具。根据当前的对话历史和用户的原始问题，你需要决定下一步该交由谁来处理。
当前在线的可用的 Worker 列表：
{agents_str}
{next_idx}. "Report_Generator_Worker": 负责生成长篇幅（如数千字、1万字）的深度报告、带目录的分析报告。注意：一旦选择此 Worker，必须**只**返回这个 Worker。（内置能力）

如果用户明确要求撰写“长报告”、“1万字报告”、“深度研究报告”，必须将任务路由给 "Report_Generator_Worker" 以启动异步生成。
如果你认为不需要再查数据了（或者用户只是闲聊），可以返回空列表 []，系统将直接流向 FINISH 生成汇总。

必须且只能调用提供的路由工具 (route_task) 进行抉择。
"""

# ============================================================================
# § 2  核心节点定义 (Nodes)
# ============================================================================

def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """Supervisor 路由中枢：只做任务分发，不做具体查询"""
    logger.info("--> [Node] Supervisor 正在思考路由...")
    
    _llm = OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
    # 这里添加langsmith的追踪
    if settings.langchain_tracing_v2:
        _llm = wrap_openai(_llm)
    # 定义路由工具规范schema，根据在线 agents 动态生成枚举
    active_agent_names = [a.agent_name for a in state.get("active_agents", [])]
    active_agent_names.append("Report_Generator_Worker") # 内置节点
    
    route_tool = {
        "type": "function",
        "function": {
            "name": "route_task",
            "description": "决定接下来由哪些专门的 Worker 节点去并发执行任务",
            "parameters": {
                "type": "object",
                "properties": {
                    "next_workers": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": active_agent_names
                        },
                        "description": "目标节点的名称列表。如果需要多方数据，可以包含多个。如果需要生成长报告则仅返回 Report_Generator_Worker。"
                    },
                    "reason": {
                        "type": "string",
                        "description": "你做出这个路由决策的原因"
                    }
                },
                "required": ["next_workers", "reason"]
            }
        }
    }

    # 转换 langchain messages 到 openai 格式
    prompt = get_dynamic_supervisor_prompt(state.get("active_agents", []))
    # 封装系统提示词到openai_msgs
    openai_msgs = [{"role": "system", "content": prompt}]
    # 封装用户和ai的对话历史到openai_msgs
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            openai_msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            openai_msgs.append({"role": "assistant", "content": m.content})
    # 调用llm，获取路由结果
    response = _llm.chat.completions.create(
        model=settings.deepseek_model,
        messages=openai_msgs,
        tools=[route_tool],
        tool_choice={"type": "function", "function": {"name": "route_task"}}, # 强制结构化输出
        temperature=0.1, # 降低温度保证路由确定性
    )
    # 获取llm返回的消息
    msg = response.choices[0].message
    if msg.tool_calls:
        try:
            args = json.loads(msg.tool_calls[0].function.arguments) # 解析llm返回的json字符串，loads是解析json to python对象
            next_workers = args.get("next_workers", []) # 获取要调度的worker列表
            reason = args.get("reason", "") # 获取调度原因
            logger.info(f"    [Supervisor 决策]: 并发分发至 {next_workers} | 理由: {reason}")
            # 注意：这里**不返回 completed_tasks**，防止Supervisor把自己的任务标记为完成
            return {"next_workers": next_workers}
        except Exception as e:
            logger.error(f"解析路由参数失败: {e}")
            return {"next_workers": []}
            
    return {"next_workers": []}


# 真实并发 MCP Worker 节点逻辑
async def generic_mcp_worker(state: AgentState, worker_name: str) -> Dict[str, Any]:
    logger.info(f"--> [Node] {worker_name} 正在并发执行...")
    _llm = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    
    worker_sys_prompt = f"你是 {worker_name}。请根据用户的对话历史，调用适当的工具来获取数据，并直接返回结果的精简摘要。"
    # 定义work节点的系统提示词
    openai_msgs = [{"role": "system", "content": worker_sys_prompt}]
    
    # 循环state messages 吧HumanMessage和AIMessage添加到openai_msgs
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            openai_msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            openai_msgs.append({"role": "assistant", "content": m.content})
    # 调用llm，根据work节点的提示词和历史消息，获取工具调用结果
    response = _llm.chat.completions.create(
        model=settings.deepseek_model,
        messages=openai_msgs,
        tools=state["mcp_tools"],
        temperature=0.1,
    )
    
    msg = response.choices[0].message
    if not msg.tool_calls:
        return {"messages": [AIMessage(content=f"[{worker_name}] 未检索到有效数据。")]}
        
    results = []
    for tc in msg.tool_calls:
        # 获取调用的工具名称
        t_name = tc.function.name
        try:
            # 获取调用参数，并通过loads解析成python对象
            t_args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            t_args = {}
        # 根据工具名称获取对应的session
        session = state["mcp_sessions"].get(t_name)
        if session:
            try:
                # await是等待异步函数执行完成，call_tool是调用mcp工具,res是获取的结果
                res = await session.call_tool(t_name, t_args)
                # 将返回结果中的text字段提取出来
                content_texts = [c.text for c in res.content if hasattr(c, 'text')]
                final_res = "\n".join(content_texts) if content_texts else str(res)
                # 将结果添加到results列表中
                results.append(f"[{t_name} 数据]:\n{final_res}")
            except Exception as e:
                results.append(f"[{t_name} 报错]: {e}")
        else:
            results.append(f"[{t_name} 失败]: 未找到连接池")
            
    # 将results列表中的每个元素连接成一个字符串
    final_output = "\n\n".join(results)
    # 返回结果
    return {"messages": [AIMessage(content=f"[{worker_name} 返回]:\n{final_output}")]}

# 工厂方法：根据 worker_name 动态生成对应的异步 Node 函数
def create_mcp_worker_node(worker_name: str):
    """根据 worker_name 动态生成对应的异步 Node 函数"""
    # 动态生成worker节点
    async def dynamic_worker_node(state: AgentState) -> Dict[str, Any]:
        return await generic_mcp_worker(state, worker_name)
    
    # 修改函数的 __name__ 以便于调试和日志追踪
    dynamic_worker_node.__name__ = f"{worker_name.lower()}_node"
    return dynamic_worker_node

async def report_generator_worker_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--> [Node] 探测到长报告需求，触发异步 Report Agent...")
    from agents.report_agent import start_report_task
    import uuid
    import asyncio
    
    task_id = f"rep-{uuid.uuid4().hex[:8]}"
    topic = state["question"]
    
    # 触发异步长报告任务
    asyncio.create_task(
        start_report_task(
            task_id, 
            topic, 
            "来自问答界面的自动触发",
            state["mcp_sessions"],
            state["mcp_tools"]
        )
    )
    
    msg = f"已为您自动切换到异步长报告生成模式。任务ID: `{task_id}`。\n请稍后调用 `/report/{task_id}/status` 检查进度或进行大纲审核。"
    return {"messages": [AIMessage(content=msg)]}

def summarizer_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--> [Node] 真实的 Summarizer 正在汇总报告...")
    _llm = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    
    sys_prompt = (
        "你是一个专业的数据汇总专家。"
        "请根据以下所有 Worker 查回来的信息，为用户撰写一份专业、结构清晰的终态报告。"
        "如果有数据缺失，请直接指出。不需要复述你的心路历程，直接输出 Markdown 报告本身。"
    )
    
    openai_msgs = [{"role": "system", "content": sys_prompt}]
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            openai_msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            openai_msgs.append({"role": "assistant", "content": m.content})
            
    response = _llm.chat.completions.create(
        model=settings.deepseek_model,
        messages=openai_msgs,
        temperature=0.3,
    )
    
    final_content = response.choices[0].message.content
    return {"messages": [AIMessage(content=final_content)]}


# ============================================================================
# § 3  异步执行核心 (LangGraph Workflow)
# ============================================================================

async def _run_planner_async(
    question: str,
    mcp_sessions: Dict[str, ClientSession] | None = None,
    mcp_tools: List[Dict[str, Any]] | None = None,
    user_id: str = "anonymous_user"
) -> Dict[str, Any]:
    """异步执行的核心管线 - 拥抱 LangGraph Supervisor 模式"""
    
    mcp_sessions = mcp_sessions or {}
    mcp_tools = mcp_tools or []

    logger.info("=========== Enterprise Supervisor 启动 ===========")
    
    # 尝试命中语义缓存（Layer 1）
    cached_response = await get_exact_cache(user_id, question)
    if cached_response:
        return {
            "question": question,
            "answer": cached_response,
            "completed_tasks": [],
            "error": None
        }
    
    if not mcp_sessions or not mcp_tools:
        logger.warning("[MCP] 当前未传入全局长连接池，这可能导致部分功能受限。")

    # 动态发现活跃的 Agents
    active_agents = await RegistryClient.discover()
    logger.info(f"动态发现在线 Agents 数量: {len(active_agents)}")
    for a in active_agents:
        logger.info(f" - [{a.agent_name}] {a.endpoint}")

    # 构建状态机
    workflow = StateGraph(AgentState)
    
    # 注册 Supervisor 和 内置的节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("Report_Generator_Worker", report_generator_worker_node)
    workflow.add_node("summarizer", summarizer_node)
    
    # 动态注册其他业务 MCP Worker 节点
    active_agent_names = []
    for agent in active_agents:
        active_agent_names.append(agent.agent_name)
        node_func = create_mcp_worker_node(agent.agent_name)
        workflow.add_node(agent.agent_name, node_func)
        # 从该节点到 summarizer 的边
        workflow.add_edge(agent.agent_name, "summarizer")
    
    # 定义路由函数 (支持并发分发)
    def route_from_supervisor(state: AgentState):
        workers = state.get("next_workers", [])
        if "Report_Generator_Worker" in workers:
            return "Report_Generator_Worker"
            
        if not workers:
            return "summarizer"
            
        valid_workers = [w for w in workers if w in active_agent_names]
        if not valid_workers:
            return "summarizer"
            
        # 并发 Send
        return [Send(w, state) for w in valid_workers]

    # 连接图
    workflow.set_entry_point("supervisor")
    
    # 边目标节点包含所有动态节点，加上内置的 Report 和 summarizer
    edge_destinations = active_agent_names + ["Report_Generator_Worker", "summarizer"]
    workflow.add_conditional_edges(
        "supervisor", 
        route_from_supervisor,
        edge_destinations
    )
    
    # Report 直接结束
    workflow.add_edge("Report_Generator_Worker", END)
    
    workflow.add_edge("summarizer", END)
    
    redis_client = await get_redis_client()
    saver = AsyncRedisSaver(redis_client)
    
    graph = workflow.compile(checkpointer=saver)
    
    # 初始化状态
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "mcp_sessions": mcp_sessions,
        "mcp_tools": mcp_tools,
        "completed_tasks": [],
        "next_workers": [],
        "user_id": user_id,
        "active_agents": active_agents
    }
    
    # LangGraph 会话标识
    config = {"configurable": {"thread_id": user_id}}
    
    try:
        # 触发执行
        final_state = await graph.ainvoke(initial_state, config=config)
        # 提取最后一句话作为最终回答
        final_msg = final_state["messages"][-1].content if final_state["messages"] else "无结果"
        
        # 将结果写入缓存
        await set_exact_cache(user_id, question, final_msg)
        
        return {
            "question": question,
            "answer": final_msg,
            "completed_tasks": final_state.get("completed_tasks", []),
            "error": None
        }
    except Exception as e:
        logger.error(f"Graph 执行出错: {e}")
        return {
            "question": question,
            "answer": "Supervisor 执行图遭遇致命故障。",
            "completed_tasks": [],
            "error": str(e)
        }

# ============================================================================
# § 4  对外兼容的同步包装口
# ============================================================================

@traceable(name="Planner_Run_Agent_MCP")
def run_planner(question: str) -> Dict[str, Any]:
    """
    提供向下兼容的同步函数入口。
    底层逻辑彻底改为基于 HTTP SSE 连接的 LangGraph 动态编排体系。
    """
    try:
        return asyncio.run(_run_planner_async(question))
    except Exception as e:
        logger.error(f"Planner 执行发生顶层故障: {e}")
        return {
            "question": question,
            "answer": f"调度中枢致命故障: {e}",
            "completed_tasks": [],
            "error": str(e)
        }
