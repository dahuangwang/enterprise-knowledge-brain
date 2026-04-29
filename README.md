# Enterprise Knowledge Brain 一站式企业级知识图谱决策大脑

基于 GraphRAG、LangGraph、Neo4j、Milvus 和 DeepSeek 构建的多 Agent 协同架构，具备复杂商业级业务报表查询、动态业务接口集成和双路降级检索能力的问答引擎。

---

## 1. 系统组件 (System Components)

企业级知识大脑系统采用了模块化解耦架构，划分为以下五个核心层：

* **接入与网关层 (Gateway Layer):**
    * 采用 **FastAPI** 提供高性能的异步 HTTP 接口（`/query`, `/index`）。
    * 预备用于多路并发支持及 Token 速率限制机制的承载。
* **Agent 编排层 (Orchestration Layer):**
    * 使用 **LangGraph** 管理各个 Agent 之间的有向无环图（DAG）状态流转与拓扑任务调度。
    * 现有核心智能体：
      - **GraphRAG Agent**（处理非结构化文档/图谱三元组数据的提取与融合检索）。
      - **Planner Agent**（全局指挥枢纽，负责对用户的复杂多维问题作意图分解、构建并分配子任务给底端领域执行器们串行解决）。
      - **SQL Agent**（结构化数据分析与报表探测器：生成内控 `SELECT` SQL 获取本地受限 SQLite 沙盒中的财务报表等关键性指标）。
      - **API Agent**（内部业务网关整合器：可模拟寻找和调用相应的 Restful API 来反馈动态业务流，如实时打捞企业待审核单据状态等）。
      - **WebResearcher Agent**（互联网情报收集与多步推理代理，获取最新的外部资讯）。
      - **DataAnalysis Agent**（利用沙盒环境执行生成的 Python 代码，进行复杂的数据分析与可视化）。
* **知识引擎层 (Knowledge Engine - 核心壁垒):**
    * **图数据库 (GraphDB):** Neo4j，存储实体及拓扑网络（节点标签 `Entity`，关系标签 `RELATION`）。通过深度图游走探索公司间隐含合作关联等链条关系。
    * **向量数据库 (VectorDB):** Milvus (Standalone)，缓存文档切片语义向量（支持高准确高维查阅 `IVF_FLAT` 索引及 `COSINE` 相似度验证）。
    * **离线建库引擎:** 包含能够自动执行 PDF -> 结构化切分拆解抽取 -> Neo4j & Milvus 推送动作的全套自动化管道。
    * **Embedding 模型:** `BAAI/bge-m3`（本地托管零远程泄露推理，超强单模型兼融长短上下文双语能力）。
* **可观测性与独立评估层 (Observability & Evaluation):**
    * **Tracing:** 基于 `@traceable` 透明引入的 LangSmith 系统监控大盘，能直通并绘制底层所有的调用链路径及每一个提示词步骤开销度。
    * **Evaluation:** 脱离于人工审美的纯机器审计评委。整合了独立第三方工具 Ragas 基于独立裁判模型判断执行管线是否在胡编乱造 （Faithfulness）与答非所问（Answer Relevance），并将客观分数存入机器评审表内。

---

## 2. 请求处理主链路 (Main Request Pipeline)

单次对话查询请求的完整执行拓扑链：

1. **复杂意图规划与解耦 (Multi-Agent Orchestration):**
   用户复杂指令到达 `planner_agent.py` ，提取可单点突破的目标域任务分包并放入栈列。
2. **底层专用域代理响应 (Task Execution Dispatch):** 
   若判定涉及结构沙盘，打给 `SQL Agent` 或 `DataAnalysis Agent`；若判定打通现有接口，打给 `API Agent`；若需要联网查询外部资讯，打给 `WebResearcher Agent`；若需要翻找年金财报或内部图谱关系信息，抛还给混合检索主力 `GraphRAG Agent`。长报告需求交由专门的 `Report Generator Worker` 异步处理。各处理结果将在上下文空间栈重叠聚合以便应对接续的环节。
3. **知识检索引擎的细分 (Query Rewriting & Routing):** 针对 GraphRAG
   先执行口语标准化纠正缩写；将纠偏后结果交给大模型决定后续的走线（`vector_search` 负责单纯名词主题，`graph_search` 负责纯实体嵌套网络、`hybrid_search` 全部并发结合）。
4. **统一生成与幻觉拦截 (Decision Synthesis):** 
   收集到各路返回之后由上层 Planner 汇总综合回答；如果知识库或各子系统无回执，强制大模型闭嘴承认数据真空状态。

---

## 3. 目录与各模块功能说明 (Project Layout)

```text
enterprise-knowledge-brain/
│
├── api/                        # HTTP 对外服务接口层
│   ├── routes.py               #   - 搭载 FastAPI 预留的异步网络收录门户
│
├── core/                       # 核心基础设施与运行时配置
│   ├── config.py               #   - 基于 pydantic-settings 进行依赖属性强解耦，覆盖连接串/重试率/溯源指标与各项 Token Limit。
│   └── logger.py               #   - 使用 Rich 库输出对工程师终端友好的染色高亮统一日志结构
│
├── agents/                     # 协作 Agent 多大脑编排与定义
│   ├── prompts.py              #   - 【重点】统一集中剥离的 Prompt 工厂与基于 Pydantic 的 Json Mode JsonSchema 控制器。
│   ├── planner_agent.py        #   - 拓扑依赖编排层，指挥下方所有工具的调度元首。
│   ├── graphrag_agent.py       #   - 高度独立并内置了双路回落机制的纯文本解析搜查代理。
│   ├── sql_agent.py            #   - 生成极低幻觉概率并且确保生成 SQLite 规范方言的执行机器。
│   ├── api_agent.py            #   - 参数指派拦截与系统对接员。
│   ├── web_researcher_agent.py #   - 联网多步搜索与情报收集专家。
│   └── report_agent.py         #   - 异步长篇距报告撰写与人机提纲确认节点。
│
├── graphrag/                   # 核心图算引擎和入库处理层
│   ├── indexer.py              #   - 大满贯流水线（支持本地 PDF 按需分解与自动 Neo4j 构建流）
│   └── graph_search.py         #   - 根据图模式建立 n-hop 子图追踪（Breadth First Search 等提取算法集合）
│
├── tools/                      # 被外部执行器们包裹的实战运行端/网关
│   ├── sql_executor.py         #   - 对大语言模型传回来的文字实施限制性的查询过滤投递
│   └── internal_apis.py        #   - 对大模型传来的 Json 负载提供本地拦截和映射反馈执行
│
├── evaluation/                 # 旁路评分裁定体系
│   ├── evaluator.py            #   - 内置一套裁判模型实例，负责将已跑飞完毕的内容反向投递进 RAGAS 获得最终性能量化分数。
│   └── audit_report.csv        #   - 持久的审校溯源分数日志档案。
│
└── tests/                      # 基于 PyTest 构建的高测试率单元矩阵
```

---

## 4. 部署与本地运行方式

系统高度解耦并且全链采用本地模型推理，具备轻量化的部署优势。

### 4.1 环境结构依赖

启动本项目之前请确保本地有 **Docker Compose** 以及 **Python 3.10+**。

```bash
# 1. 复制全局参数预设档
cp .env.example .env

# 2. 修改配置项
# 请使用文本编辑器打开 .env，并将您的 DEEPSEEK_API_KEY （支持兼容 OpanAI 等额接口）填入其中。

# 3. 配置 Python 包
pip install -r requirements.txt
# (注: `sentence-transformers` 首次初始化需要从 HuggingFace Hub 下载 2.3GB `BAAI/bge-m3`。
# 国内网络如果不通畅，建议终端配置: $env:HF_ENDPOINT="https://hf-mirror.com")
```

### 4.2 运行核心数据库设施 (Neo4j + Milvus)

系统基于一套 `docker-compose.yml` 把控所有外储平台资源。

```bash
# 拉起全部后台数据容器，该流程会自动部署 Milvus, Neo4j
docker-compose up -d

# 检查健康状况确保全部绿灯
docker-compose ps
```

### 4.3 文档注入（测试数据初始化）

系统自带建库解析代码能够独立识别并转换原始业务文档材料入图与缓存。

```bash
# 执行完整入库全流程，将会自动写入 Neo4j 与 Milvus
python -m graphrag.indexer --pdf data/xxxx.pdf

# 如果担心成本，可以仅使用纯推演而不做最后一次数据写表 (dry run mode)
python -m graphrag.indexer --pdf data/xxxx.pdf --dry-run
```

### 4.4 代码运行交互点指令集合

项目可支持纯脚本态执行、HTTP Server 对外态执行，和基准评率报告生成执行。

```bash
# [选项A]: 独立作为本地应用入口发起测试（基于编排主管层切入）
python
>>> from agents.planner_agent import run_planner
>>> result = run_planner("请帮我核对该系统的最新建设审批流情况和内部总投资金报表。")
>>> print(result["final_answer"])

# [选项B]: 通过 FastAPI 以 Http 微服务态提供对外可调用能力
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# 文档访问：http://127.0.0.1:8000/docs
# 接口请求示例: POST /query {"question": "招行各项科研项目今年状态？"}

# [选项C]: 脱离业务进行独立全套自动化评分验收与审计文件生成（第五阶段考核支持点）
python scripts/run_eval.py
```

### 4.5 可观测仪表盘（Tracing配置） - 选配

通过注册并且设定在您的系统全局 `LangSmith` 控制参数：
```ini
# (位于您的 .env 内部)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__xxxx_您的平台私钥_xxxx
LANGCHAIN_PROJECT=EnterpriseKnowledgeBrain
```
系统后续运行过程中所有的 Agent 回退与分叉重试等内耗明细都可以跨云在图形工作台中被监控到，且零代码侵入。
