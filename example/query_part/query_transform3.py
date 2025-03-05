# 查询重写（生成子问题）
# 给定一组工具和用户查询，决定要生成的子问题集合，以及每个子问题应该运行的工具

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
from llama_index.llms.siliconflow import SiliconFlow

api_key = "sk-yrmmgcztvxoijeigwmhqqohhafaolagrmlffjuhiifmrdlcg"
llm = SiliconFlow(api_key=api_key, model="Qwen/Qwen2.5-7B-Instruct")
Settings.embed_model = SiliconFlowEmbedding(
    api_key=api_key,
    model_name="netease-youdao/bce-embedding-base_v1")
Settings.llm = llm

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul").load_data()
index = VectorStoreIndex.from_documents(documents)

from llama_index.core.question_gen import LLMQuestionGenerator

question_gen = LLMQuestionGenerator.from_defaults(llm=llm) # 此时使用默认的prompt
#print(question_gen.get_prompts())

from llama_index.core.tools import ToolMetadata

tool_choices = [
    ToolMetadata(
        name="uber_2021_10k",
        description=(
            "Provides information about Uber financials for year 2021"
        ),
    ),
    ToolMetadata(
        name="lyft_2021_10k",
        description=(
            "Provides information about Lyft financials for year 2021"
        ),
    ),
]

from llama_index.core import QueryBundle

query_str = "Compare and contrast Uber and Lyft"
choices = question_gen.generate(tool_choices, QueryBundle(query_str=query_str))
print(choices)
