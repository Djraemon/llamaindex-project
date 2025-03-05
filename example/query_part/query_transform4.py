# HyDE查询转换
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import  SimpleDirectoryReader, Settings,Document
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

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
# 加载文档
documents = SimpleDirectoryReader("./data/paul").load_data()
# 创建向量索引
index = VectorStoreIndex.from_documents(documents)

query_str = "what did paul graham do after going to RISD"

# 首先，进行无转换的查询：相同的查询字符串用于嵌入查找和总结。
query_engine = index.as_query_engine()
response = query_engine.query(query_str)
print(response)

# 然后，使用HyDEQueryTransform来生成一个假设的文档，并将其用于嵌入查找。
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)
response = hyde_query_engine.query(query_str)
print(response)

query_bundle = hyde(query_str)
hyde_doc = query_bundle.embedding_strs[0]
print(hyde_doc)