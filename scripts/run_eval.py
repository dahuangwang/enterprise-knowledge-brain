"""
scripts/run_eval.py
────────────────────────────────────────────────────────────────────────────
【第五阶段】一键跑通评测流水线的入口脚本。
运行命令： python scripts/run_eval.py
"""
import sys
import os

# 将根目录添加到环境变量，以使包导入正常工作
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import get_logger
from evaluation.evaluator import run_evaluation

logger = get_logger("run_eval")

# 测试集示例
TEST_DATA = [
    {
        "question": "招商银行极速开户系统是谁投资的？投资金额是多少？",
        "ground_truth": "极速开户系统是由招商银行金融科技部投资搭建，第一期投资金额为 500.0 万元。",
    },
    {
        "question": "同城双活灾备系统目前的运营状态怎么样了？",
        "ground_truth": "同城双活灾备系统状态为'建设中'。",
    }
]

def main():
    logger.info("=== 启动端到端自动化审核脚本 ===")
    
    # 设定持久化位置
    audit_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "evaluation", 
        "audit_report.csv"
    )
    
    df = run_evaluation(testset=TEST_DATA, output_file=audit_file)
    
    if not df.empty:
        logger.info("=== 自动化审核脚本结束 ===")
        print("\n最终生成的审核表格概览（部分）：")
        print(df[["question", "faithfulness", "answer_relevancy"]].head())
        print(f"\n完整报告已生成: {audit_file}。您可以使用 Excel 或文本编辑器查看内容。")
    else:
        logger.error("评估执行由于异常而中断，生成表格失败。")

if __name__ == "__main__":
    main()
