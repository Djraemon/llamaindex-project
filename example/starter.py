import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

api_key = "sk-yrmmgcztvxoijeigwmhqqohhafaolagrmlffjuhiifmrdlcg"
api_base_url =  "https://api.siliconflow.cn/v1/"

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("作者在成长过程中做了什么？")
print(response)