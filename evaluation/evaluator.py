"""
evaluation/evaluator.py
────────────────────────────────────────────────────────────────────────────
【第五阶段】基于 RAGAS 的大模型幻觉与评价流水线
能够独立验证 GraphRAG 的表现，评价 Faithfulness 和 Answer Relevance。
运行完毕后，将结果持久化至本地文件中（审核报告）。
"""
import os
import json
from typing import List, Dict, Any
import datetime

import pandas as pd
from datasets import Dataset

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)

from core.logger import get_logger
from core.config import settings
from agents.graphrag_agent import run_agent

logger = get_logger(__name__)

def _get_judge_llm() -> ChatOpenAI:
    """初始化用于判卷的大语言模型（通常要求性能高的模型作为法官）"""
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.0, # 判卷要求严格、确定性
    )

def _get_judge_embeddings() -> HuggingFaceEmbeddings:
    """初始化 RAGAS 做相似度判别时的 Embeddings，复用本地 BGE 模型"""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)

def run_evaluation(testset: List[Dict[str, str]], output_file: str = "evaluation/audit_report.csv") -> pd.DataFrame:
    """
    运行完整的 RAGAS 评估并输出持久化审核文件
    
    Args:
        testset: 测试集格式应为 [{"question": "...", "ground_truth": "..."}]
        output_file: 审核报告导出路径
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    logger.info(f"========= 启动 RAGAS 自动化流水线评估，共 {len(testset)} 条用例 =========")

    # 1. 采集实际回答与检索上下文
    for item in testset:
        q = item["question"]
        gt = item.get("ground_truth", "")
        
        logger.info(f"正在测试用例: {q}")
        
        # 调用核心流水线进行推理（这里只关注信息检索向的问题）
        res = run_agent(question=q)
        ans = res.get("answer", "")
        
        if settings.app_profile == "production":
            logger.warning(
                "当前 run_agent 未返回原始检索上下文，production 评估将仅记录检索统计，"
                "不再伪造 evidence。建议扩展 Agent 输出 raw_context 后再启用 RAGAS faithfulness。"
            )
            evidence = (
                f"检索统计: vector_hits={res.get('vector_hits', 0)}, "
                f"graph_stats={res.get('graph_stats', '')}"
            )
        else:
            evidence = (
                "[DEMO_EVIDENCE] demo profile 使用回答文本辅助贯通 RAGAS 流程，"
                "该上下文不能作为生产忠实性审计依据。"
                f"V-Docs: 命中 {res.get('vector_hits', 0)} 条。"
                f"G-Docs: 图谱节点记录 [{res.get('graph_stats', '')}]。"
                f"内部参考信息: {ans}"
            )
        
        questions.append(q)
        answers.append(ans)
        contexts.append([evidence])
        ground_truths.append(gt)

    # 2. 组装支持 HuggingFace datasets 的标准结构
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data_dict)

    # 3. 装载评委模型并执行评测
    judge_llm = _get_judge_llm()
    judge_emb = _get_judge_embeddings()
    
    logger.info("准备调用 RAGAS 法官给回答打分 (Faithfulness, Answer Relevance)...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=judge_llm,
            embeddings=judge_emb,
        )
    except Exception as e:
        logger.error(f"RAGAS 评测崩溃，由于限流或网络原因: {e}")
        return pd.DataFrame()

    df = result.to_pandas()
    logger.info(f"评测完成。报告行数: {len(df)}")
    logger.info(f"全局均分如下:\n {result}")

    # 4. 将评分持久化到审核报告文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 追加时间戳记录
    df['audit_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 追加保存（如果是首次创建，则带表头；否则无表头追加）
    header = not os.path.exists(output_file)
    df.to_csv(output_file, mode='a', header=header, index=False, encoding='utf-8-sig')
    
    logger.info(f"评分已完成持久化，保存至审核文件: {output_file}")
    
    return df
