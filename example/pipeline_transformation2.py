# 数据摄取管道pipeline  demo2
# 添加摘要提取器等变换组件
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,Settings
from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
from llama_index.llms.siliconflow import SiliconFlow

api_key = "sk-yrmmgcztvxoijeigwmhqqohhafaolagrmlffjuhiifmrdlcg"
api_base_url =  "https://api.siliconflow.cn/v1/"

# bge-base嵌入模型
Settings.embed_model = SiliconFlowEmbedding(
    api_key=api_key,
    model_name="netease-youdao/bce-embedding-base_v1")

Settings.llm = SiliconFlow(
    api_key=api_key,
    model="Qwen/Qwen2.5-7B-Instruct")


transformations = [
    SentenceSplitter(),
    TitleExtractor(nodes=5),
    QuestionsAnsweredExtractor(questions=3),
    SummaryExtractor(summaries=["prev", "self"]), # 添加摘要提取器
    KeywordExtractor(keywords=10),
]

# 创建一个IngestionPipeline实例，并传入transformations参数
pipeline = IngestionPipeline(transformations=transformations)
documents = SimpleDirectoryReader("./data/paul").load_data()
nodes = pipeline.run(documents=documents)
print("-------node-----------\n",nodes)
index = VectorStoreIndex(nodes)
print("-------index-----------\n",index)
query_engine = index.as_query_engine()
response = query_engine.query("Where is the capital of France?")
print("-------response-----------\n",response)

