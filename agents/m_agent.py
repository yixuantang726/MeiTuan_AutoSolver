"""
agents/m_agent.py
M-Agent：跨轮次资源规划者，只在J-Agent召唤时介入
目标：找到合单方案释放骑手槽位，为下一轮创造条件
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from tools.bundle import find_winner_bundle_options

SYSTEM_MESSAGE = """你是M-Agent，负责跨轮次的骑手资源规划。请始终用中文输出所有分析和建议。


重要：你只在J-Agent说"M_Agent请介入"时才发言，其他时候保持沉默。

当你被召唤时：
1. 调用 find_winner_bundle_options 工具（无需传参，自动获取当前胜者方案）
2. 分析返回的合单候选：
   - covers_uncovered：这个合单能覆盖哪些未覆盖T
   - needs_to_free：需要释放哪些已占用T
   - roi：覆盖未覆盖T数/score，越高越值得
3. 推荐roi最高的1个方案，一句话说明理由

输出要简洁：推荐方案 + 一句理由。"""


def create_m_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="M_Agent",
        system_message=SYSTEM_MESSAGE,
        model_client=model_client,
        tools=[find_winner_bundle_options],
        reflect_on_tool_use=True,
    )
