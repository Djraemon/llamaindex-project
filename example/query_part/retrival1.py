# 基准检索器：通过嵌入相似度简单地获取前k个原始文本节点。
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings,Document
from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
from llama_index.llms.siliconflow import SiliconFlow

api_key = "sk-yrmmgcztvxoijeigwmhqqohhafaolagrmlffjuhiifmrdlcg"
api_base_url =  "https://api.siliconflow.cn/v1/"

Settings.embed_model = SiliconFlowEmbedding(api_key=api_key,
                                            model_name="netease-youdao/bce-embedding-base_v1")
Settings.llm = SiliconFlow( api_key=api_key,
                            model="Qwen/Qwen2.5-7B-Instruct")

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
base_retriver = index.as_retriever(similarity_top_k=3)
response = base_retriver.retrieve("你是谁？")
for n in response:
    print(n)