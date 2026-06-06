"""
agents/c_agent.py
C-Agent：以coverage为锚定指标，提出覆盖率最大的派单方案
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from tools.greedy import run_coverage_greedy

SYSTEM_MESSAGE = """你是C-Agent，负责提出覆盖率最大的派单方案。

你的锚定指标是：期望覆盖T数（coverage）。

每轮你需要：
1. 调用 run_coverage_greedy 工具获取方案
2. 报告方案的核心指标：coverage、F(S)、合单数、使用骑手数
3. 用一句话说明你的方案为什么在覆盖率上有优势

你只关心覆盖更多T，不关心score高低。
输出格式要简洁，重点突出coverage数值。"""


def create_c_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="C_Agent",
        system_message=SYSTEM_MESSAGE,
        model_client=model_client,
        tools=[run_coverage_greedy],
        reflect_on_tool_use=True,
    )
