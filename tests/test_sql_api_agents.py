"""
tests/test_sql_api_agents.py
────────────────────────────────────────────────────────────────────────────
对第四阶段 MVP 完成的 SQL 和 API Agent 以及底层工具的单元测试。
"""
import pytest
from unittest.mock import MagicMock, patch

from agents.sql_agent import run_sql_agent
from agents.api_agent import run_api_agent
from tools.sql_executor import execute_sandbox_sql
from tools.internal_apis import call_internal_api

class TestTools:
    
    def test_sql_executor_valid_select(self):
        """测试正常只读查询"""
        result = execute_sandbox_sql("SELECT project_name FROM financial_reports WHERE department='金融科技部'")
        assert "招商银行极速开户系统" in result
        
    def test_sql_executor_invalid_dml(self):
        """测试防止破坏性查询"""
        result = execute_sandbox_sql("DROP TABLE financial_reports")
        assert "安全策略限制" in result
        
    def test_sql_executor_syntax_error(self):
        """测试对于语法错误的友好返回"""
        result = execute_sandbox_sql("SELECT xyz FROM not_exist_table")
        assert "语法错误或引用了不存在的字段" in result
        
    def test_internal_apis_valid_call(self):
        """测试能成功调用的假接口"""
        res = call_internal_api("get_project_approval_status", {"project_name": "招商平台"})
        assert "【法务待审批】" in res
        
    def test_internal_apis_invalid_endpoint(self):
        """测试对于不存在接口的优雅降级"""
        res = call_internal_api("delete_database", {})
        assert "接口调用失败" in res
        assert "未注册的端点" in res


class TestAgents:
    
    @patch("agents.sql_agent._llm_json_call")
    def test_run_sql_agent_success(self, mock_llm_call):
        """Mock LLM 行为并测试 SQL Agent 返回链路"""
        mock_llm_call.return_value = {
            "reasoning": "想要查询指定的营收",
            "sql_query": "SELECT revenue_q1 FROM financial_reports WHERE department='风险控制部'"
        }
        # 使用 MagicMock() 代表 llm_client
        res = run_sql_agent("看看风险部的营收", MagicMock())
        
        assert res["error"] is None
        assert res["sql_query"] == "SELECT revenue_q1 FROM financial_reports WHERE department='风险控制部'"
        # 断言能够在内联的 SQLite 中查阅出了正确的假数据对应项 300.5
        assert "300.5" in res["answer"]
        
    @patch("agents.api_agent._llm_json_call")
    def test_run_api_agent_success(self, mock_llm_call):
        """Mock LLM 行为并测试 API Agent 返回链路"""
        mock_llm_call.return_value = {
            "reasoning": "想要调用预算探测接口",
            "endpoint": "get_department_budget",
            "params": {"department": "金融科技部"}
        }
        
        res = run_api_agent("查询金科的预算", MagicMock())
        
        assert res["error"] is None
        assert res["endpoint"] == "get_department_budget"
        # 能够把 params 的结果转发并获得文本断言
        assert "1500万净结余" in res["answer"]
