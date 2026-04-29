"""
agents/report_agent.py
────────────────────────────────────────────────────────────────────────────
Report Generator Agent Workflow
支持生成长文档，并在大纲阶段进行人工确认 (Human-in-the-loop)。
"""
import json
import asyncio
from typing import Any, Dict, List, Annotated, TypedDict

from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.errors import GraphInterrupt

from mcp import ClientSession

from core.config import settings
from core.logger import get_logger
from utils.redis_client import get_redis_client

logger = get_logger("report_agent")

class Section(TypedDict):
    title: str
    purpose: str

class ReportState(TypedDict):
    task_id: str
    topic: str
    requirements: str
    outline: List[Section]
    outline_approved: bool
    drafts: Dict[str, str]
    final_report: str
    mcp_sessions: Dict[str, ClientSession]
    mcp_tools: List[Dict[str, Any]]

def get_llm():
    _llm = OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
    if settings.langchain_tracing_v2:
        _llm = wrap_openai(_llm)
    return _llm

async def plan_outline_node(state: ReportState) -> Dict[str, Any]:
    logger.info(f"[Task {state['task_id']}] 正在规划大纲...")
    _llm = get_llm()
    
    prompt = f"""
    你是一个专业的高级报告撰写架构师。
    请根据以下主题和要求，为一份长达1万字的深度报告规划目录大纲。
    
    主题: {state['topic']}
    要求: {state.get('requirements', '无特别要求')}
    
    请以 JSON 数组格式返回大纲，每个元素是一个对象，包含:
    - "title": 章节标题
    - "purpose": 该章节需要调研或说明的核心内容（用于指导后续的数据收集和起草）
    
    直接输出合法的 JSON 数组，不要加额外的解释或 Markdown 标记。
    """
    
    response = _llm.chat.completions.create(
        model=settings.deepseek_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"} if "deepseek" not in settings.deepseek_model.lower() else None
    )
    
    content = response.choices[0].message.content.strip()
    try:
        # 兼容可能有 markdown 代码块包裹的情况
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        
        # DeepSeek V3 不一定完全遵从 json_object，这里做容错解析
        outline = json.loads(content)
        if isinstance(outline, dict) and "outline" in outline:
            outline = outline["outline"]
    except Exception as e:
        logger.error(f"解析大纲 JSON 失败: {e}\n原文: {content}")
        # 降级：生成一个默认大纲
        outline = [{"title": "第一章：背景与概述", "purpose": "介绍基本情况"}]
        
    return {"outline": outline, "outline_approved": False}

async def draft_sections_node(state: ReportState) -> Dict[str, Any]:
    # Interrupt 逻辑：如果大纲未确认，中断执行
    if not state.get("outline_approved"):
        logger.info(f"[Task {state['task_id']}] 大纲未确认，等待人工介入...")
        # 抛出 GraphInterrupt，让调用方捕获或让图挂起
        raise GraphInterrupt("Waiting for outline approval")
        
    logger.info(f"[Task {state['task_id']}] 大纲已确认，开始逐章起草...")
    _llm = get_llm()
    drafts = state.get("drafts", {})
    
    for idx, section in enumerate(state["outline"]):
        title = section["title"]
        purpose = section["purpose"]
        
        logger.info(f"  -> 正在起草章节: {title}")
        
        # 为了保持简单（Surgical Changes），这里使用一个通用的调研 Prompt 交给 Planner MCP 工具池。
        # 这里模拟调用底层数据，直接通过大模型+Tools解决。
        # 为避免复杂化，将系统提示词设置为仅针对当前章节的调研和写作。
        sys_prompt = f"""
        你正在撰写长篇报告的某一章节。
        报告总主题: {state['topic']}
        当前章节: {title}
        章节目标: {purpose}
        
        请调用适合的工具检索你需要的数据。如果没有合适工具，请尽力运用你的知识进行扩写。
        输出必须是该章节的完整正文段落（Markdown 格式，以 {title} 为标题）。
        """
        
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "请开始撰写本章节内容。"}]
        
        # 简单的一轮 Tool Calling 循环
        response = _llm.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            tools=state["mcp_tools"],
            temperature=0.4
        )
        msg = response.choices[0].message
        
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                t_name = tc.function.name
                try:
                    t_args = json.loads(tc.function.arguments)
                except Exception:
                    t_args = {}
                session = state["mcp_sessions"].get(t_name)
                if session:
                    try:
                        res = await session.call_tool(t_name, t_args)
                        content_texts = [c.text for c in res.content if hasattr(c, 'text')]
                        final_res = "\n".join(content_texts) if content_texts else str(res)
                    except Exception as e:
                        final_res = f"Error: {e}"
                else:
                    final_res = "Tool session not found."
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": t_name, "content": final_res})
                
            # 二次调用生成章节正文
            response2 = _llm.chat.completions.create(
                model=settings.deepseek_model,
                messages=messages,
                temperature=0.4
            )
            drafts[title] = response2.choices[0].message.content
        else:
            drafts[title] = msg.content
            
    return {"drafts": drafts}

async def review_report_node(state: ReportState) -> Dict[str, Any]:
    logger.info(f"[Task {state['task_id']}] 正在合并与审校全局报告...")
    drafts = state["drafts"]
    
    # 简单拼接，由于前置已按章节生成，这里直接组装。
    # 也可引入 LLM 对全文进行润色，但为避免超出上下文限制，先直接组装并加前言。
    full_report = f"# {state['topic']}\n\n"
    for sec in state["outline"]:
        title = sec["title"]
        content = drafts.get(title, "暂无内容。")
        # 确保不重复写标题
        if not content.startswith("#"):
            full_report += f"## {title}\n\n"
        full_report += f"{content}\n\n"
        
    return {"final_report": full_report}

# 构建状态机
def build_report_graph():
    workflow = StateGraph(ReportState)
    
    workflow.add_node("plan_outline", plan_outline_node)
    workflow.add_node("draft_sections", draft_sections_node)
    workflow.add_node("review_report", review_report_node)
    
    workflow.set_entry_point("plan_outline")
    workflow.add_edge("plan_outline", "draft_sections")
    workflow.add_edge("draft_sections", "review_report")
    workflow.add_edge("review_report", END)
    
    return workflow

# 单例图对象
_workflow = build_report_graph()

async def get_report_graph():
    redis_client = await get_redis_client()
    saver = AsyncRedisSaver(redis_client)
    return _workflow.compile(checkpointer=saver)

async def start_report_task(task_id: str, topic: str, requirements: str, mcp_sessions, mcp_tools):
    graph = await get_report_graph()
    initial_state = {
        "task_id": task_id,
        "topic": topic,
        "requirements": requirements,
        "outline": [],
        "outline_approved": False,
        "drafts": {},
        "final_report": "",
        "mcp_sessions": mcp_sessions,
        "mcp_tools": mcp_tools
    }
    config = {"configurable": {"thread_id": task_id}}
    try:
        # 执行图，遇到 Interrupt 会抛出 GraphInterrupt 异常或直接返回停滞的状态
        await graph.ainvoke(initial_state, config=config)
    except GraphInterrupt:
        logger.info(f"[Task {task_id}] 流程暂停，等待大纲审核。")
    except Exception as e:
        logger.error(f"[Task {task_id}] 长报告执行失败: {e}")

async def resume_report_task(task_id: str, approved_outline: List[Dict]):
    # 恢复执行前，更新状态为已批准，并传入用户修改后的大纲
    logger.info(f"[Task {task_id}] 接收到大纲批准，恢复执行。")
    
    # 获取当前状态以确保 mcp_sessions 仍然存在（注：在真实的分布式系统中，mcp_sessions 等非序列化对象
    # 不能直接存在于 redis checkpoint 中，可能需要重新注入。这里为了简化演示，假设它们在同一进程或通过某种方式重建）
    # 按照 Karpathy 准则，我们先做一个可行方案。如果反序列化失败，我们需要在 invoke 时传入。
    # 为了防止 mcp_sessions 丢失，我们在 API 侧重新提供这些依赖。
    pass # 稍后在 api 层调用 update_state 和 ainvoke
