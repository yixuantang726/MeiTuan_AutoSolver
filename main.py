"""
main.py
入口文件：加载模型配置，启动辩论

使用方式：
1. 在项目根目录创建 model_config.yaml
2. python main.py
"""
import asyncio
import yaml
import os
from autogen_core.models import ChatCompletionClient
from debate import run_debate


def load_model_client(config_path: str = "model_config.yaml") -> ChatCompletionClient:
    """从yaml加载模型配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"找不到 {config_path}，请参考以下格式创建：\n"
            "provider: autogen_ext.models.openai.OpenAIChatCompletionClient\n"
            "config:\n"
            "  model: gpt-4o-mini\n"
            "  api_key: your-api-key\n"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return ChatCompletionClient.load_component(config)


async def main():
    model_client = load_model_client()
    await run_debate(model_client, verbose=True, time_limit=300.0)


if __name__ == "__main__":
    asyncio.run(main())
