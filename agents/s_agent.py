"""
agents/s_agent.py
S-Agent：以score为锚定指标，提出总成本最小的派单方案
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from tools.greedy import run_score_greedy

SYSTEM_MESSAGE = """你是S-Agent，负责提出总派单成本最小的方案。

你的锚定指标是：总score（期望派单成本R(S)）。

每轮你需要：
1. 调用 run_score_greedy 工具获取方案
2. 报告方案的核心指标：score、F(S)、coverage、使用骑手数
3. 用一句话说明你的方案为什么在成本上有优势

你只关心压低score，不关心覆盖率高低。
输出格式要简洁，重点突出score和F(S)数值。"""


def create_s_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="S_Agent",
        system_message=SYSTEM_MESSAGE,
        model_client=model_client,
        tools=[run_score_greedy],
        reflect_on_tool_use=True,
    )
