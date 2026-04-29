"""
mcp_servers/data_analysis_server.py
────────────────────────────────────────────────────────────────────────────
MCP 架构：数据分析 MCP Server (本地安全沙盒方案)
负责拦截数据分析任务，调用本地 Qwen2.5-Coder32B-Instruct 模型生成代码，
并在隔离的 Docker 容器中执行，最后将图表转换为 Base64 内嵌 Markdown 返回。
"""
import sys
import os
import re
from typing import Any

# 将根目录添加到环境变量，以使包导入正常工作
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from openai import AsyncOpenAI

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from core.registry import AgentCard, RegistryClient
from core.logger import get_logger
from core.config import settings
from tools.python_sandbox import run_python_code_in_sandbox

logger = get_logger("mcp_data_analysis_server")

# ============================================================================
# § 1 配置本地 vLLM 客户端
# ============================================================================
client = AsyncOpenAI(api_key="EMPTY", base_url=settings.vllm_base_url)

# ============================================================================
# § 2 定义 Schema
# ============================================================================
class DataAnalysisSchema(BaseModel):
    data_context: str = Field(..., description="上下文数据，比如 SQL 查询出来的 JSON 或者文本数据")
    user_instruction: str = Field(..., description="用户的分析指令，比如 '请画出各部门的预算柱状图'")

# ============================================================================
# § 3 初始化 MCP Server 并注册 Tools
# ============================================================================
mcp_server = Server("data-analysis-server")

async def generate_python_code(data_context: str, user_instruction: str) -> str:
    """调用本地 Qwen2.5-Coder 生成 Python 代码"""
    prompt = f"""你是一个高级数据科学家和 Python 工程师。
请根据以下提供的数据上下文和用户指令，编写 Python 代码。
数据将被保存为运行目录下的 `data.json` 文件中，你可以直接读取它。
如果你需要绘图，请使用 matplotlib 或 seaborn，并将生成的图片保存到 `./output/` 目录下（注意：必须保存到该目录，而不是直接 show）。
你的输出必须只包含 Python 代码，不要包含任何其他解释性文本，请使用 ```python 代码块包裹。

[数据上下文]
{data_context}

[用户指令]
{user_instruction}
"""
    try:
        response = await client.chat.completions.create(
            model=settings.vllm_model_name,
            messages=[
                {"role": "system", "content": "你是一个代码助手，只输出可执行的 Python 代码。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        
        reply = response.choices[0].message.content
        # 提取 ```python ... ``` 中的代码
        match = re.search(r'```python\n(.*?)\n```', reply, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 如果没有代码块标记，尝试直接清理
        return reply.strip().strip('`')
    except Exception as e:
        logger.error(f"Failed to generate code via vLLM: {e}")
        raise

@mcp_server.tool()
async def run_analysis_and_plot(data_context: str, user_instruction: str) -> str:
    """根据提供的数据执行数据分析和 Python 绘图任务"""
    try:
        # 1. 生成代码
        logger.info("Generating Python code using local Qwen2.5-Coder...")
        code = await generate_python_code(data_context, user_instruction)
        logger.debug(f"Generated Code:\n{code}")
        
        if not code:
            return "未能生成有效的 Python 代码。"

        # 2. 执行沙盒环境
        logger.info("Executing generated code in local Python Sandbox...")
        result = run_python_code_in_sandbox(code, data_context)
        
        # 3. 构造返回的 Markdown
        markdown_output = "### 数据分析执行结果\n\n"
        
        if result["stdout"]:
            markdown_output += f"**标准输出:**\n```\n{result['stdout']}\n```\n\n"
            
        if result["stderr"]:
            markdown_output += f"**错误输出:**\n```\n{result['stderr']}\n```\n\n"
            
        if result["images"]:
            markdown_output += "**生成的可视化图表:**\n\n"
            for img in result["images"]:
                # 使用 Base64 内嵌图片
                markdown_output += f"![{img['filename']}](data:image/png;base64,{img['base64']})\n\n"
        elif not result["stderr"] and not result["stdout"]:
             markdown_output += "代码已成功执行，但没有任何输出或图表生成。"
             
        # 可以附加上生成的代码
        markdown_output += f"<details><summary>展开查看生成的 Python 代码</summary>\n\n```python\n{code}\n```\n</details>"
        
        return markdown_output

    except Exception as e:
        logger.error(f"Analysis Server Error: {e}")
        return f"数据分析执行过程中发生异常: {e}"


# ============================================================================
# § 4 Streamable HTTP 传输层与 Starlette 路由绑定
# ============================================================================
session_manager = StreamableHTTPSessionManager(mcp_server)

async def handle_mcp(request: Any):
    """处理 MCP Client 的流式请求"""
    await session_manager.handle_request(request.scope, request.receive, request._send)

async def health_check(request: Any):
    return JSONResponse({"status": "ok", "service": "data-analysis-mcp"})

# ============================================================================
# § 5  生命周期与 AgentCard 注册
# ============================================================================
data_analysis_agent_card = AgentCard(
    agent_id="data_analysis_worker_01",
    agent_name="DataAnalysis_Worker",
    description="负责利用沙盒环境执行 Python 代码来进行复杂的数据分析与图表生成。",
    endpoint=settings.mcp_data_analysis_server_url,
    capabilities=["data_analysis", "python", "chart", "sandbox"],
    mcp_tools_summary=[
        {"name": "run_analysis_and_plot", "description": "执行数据分析和 Python 绘图任务"}
    ]
)

@asynccontextmanager
async def lifespan(app: Starlette):
    await RegistryClient.register(data_analysis_agent_card, ttl=30)
    
    async def heartbeat_loop():
        while True:
            await asyncio.sleep(15)
            await RegistryClient.heartbeat("data_analysis_worker_01", ttl=30)
            
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    
    yield
    
    heartbeat_task.cancel()
    await RegistryClient.unregister("data_analysis_worker_01")

app = Starlette(routes=[
    Route("/health", endpoint=health_check, methods=["GET"]),
    Route("/mcp", endpoint=handle_mcp, methods=["GET", "POST"])
], lifespan=lifespan)

if __name__ == "__main__":
    logger.info("启动 数据分析 MCP Server (Streamable HTTP) 端口: 8004")
    uvicorn.run(app, host="0.0.0.0", port=8004, workers=1)
