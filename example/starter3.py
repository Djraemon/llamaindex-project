import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import SummaryIndex, SimpleDirectoryReader,Document
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

#documents = SimpleDirectoryReader("data").load_data()
document = Document.example()
document.doc_id = "1"
index = SummaryIndex.from_documents([document])

query_engine = index.as_query_engine()
response = query_engine.query("作者在成长过程中做了什么？")
print(response)