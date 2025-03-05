# 能够进行乘法运算的代理 demo
from llama_index.core.tools import FunctionTool
from llama_index.llms.siliconflow import SiliconFlow
from llama_index.core.agent import ReActAgent

# 代理示例
# 定义示例工具
def multiply(a: int, b: int) -> int:
    """将两个整数相乘并返回结果整数"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
api_key = "sk-yrmmgcztvxoijeigwmhqqohhafaolagrmlffjuhiifmrdlcg"
# 初始化 llm
llm = SiliconFlow(
    api_key=api_key,
    model="Qwen/Qwen2.5-7B-Instruct")

# 初始化 ReAct 代理
agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)
agent.chat("将 3 和 4 相乘")