# 多步查询
# 将复杂的查询分解为顺序子问题，依次查询，最终得到答案
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import Settings
from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
from llama_index.llms.siliconflow import SiliconFlow

api_key = "sk-yrmmgcztvxoijeigwmhqqohhafaolagrmlffjuhiifmrdlcg"
api_base_url =  "https://api.siliconflow.cn/v1/"

Settings.embed_model = SiliconFlowEmbedding(
    api_key=api_key,
    model_name="netease-youdao/bce-embedding-base_v1")
Settings.llm = SiliconFlow( 
    api_key=api_key,
    model="Qwen/Qwen2.5-7B-Instruct")


from llama_index.core import SimpleDirectoryReader, SummaryIndex
documents = SimpleDirectoryReader("./data/paul").load_data()
index = SummaryIndex.from_documents(documents)
index_summary = "Used to answer questions about the author"


from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform
step_decompose_transform = StepDecomposeQueryTransform(llm=Settings.llm, verbose=True)


# 将日志级别设置为DEBUG，以获得更详细的输出
from llama_index.core.query_engine import MultiStepQueryEngine

query_engine = index.as_query_engine(llm=Settings.llm)
query_engine = MultiStepQueryEngine(    
    query_engine=query_engine,
    query_transform=step_decompose_transform,
    index_summary=index_summary,
)

response = query_engine.query(
    #"What are the two main things the author does outside of school before he go to college?", # 编程和写作
    "In which city did the author found his first company, Viaweb?"
)
print(response)