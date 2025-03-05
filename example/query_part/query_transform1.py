# 查询转换
# 路由，使用查询来选择一组相关的工具选项
# 默认模型为OPENAI
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)

# pydantic选择器将pydantic对象提供给调用API的函数
# 单个选择器（pydantic，函数调用）
# selector = PydanticSingleSelector.from_defaults()
# 多个选择器（pydantic，函数调用）
# selector = PydanticMultiSelector.from_defaults()
# LLM选择器使用文本补全端点
# 单个选择器（LLM）
# selector = LLMSingleSelector.from_defaults()
# 多个选择器（LLM）

selector = LLMMultiSelector.from_defaults()

from llama_index.core.tools import ToolMetadata

tool_choices = [
    ToolMetadata(
        name="covid_nyt",
        description=("This tool contains a NYT news article about COVID-19"),
    ),
    ToolMetadata(
        name="covid_wiki",
        description=("This tool contains the Wikipedia page about COVID-19"),
    ),
    ToolMetadata(
        name="covid_tesla",
        description=("This tool contains the Wikipedia page about apples"),
    ),
]

selector_result = selector.select(
    tool_choices, query="Tell me more about COVID-19" # 会根据COVID-19选择Tool
)
print(selector_result.selections)