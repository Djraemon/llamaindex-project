# 数据摄取管道pipeline  demo1
from llama_index.core import Document,VectorStoreIndex,SimpleDirectoryReader,Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor,QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
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

# 创建带有转换的管道
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
        Settings.embed_model,
    ],
)

# 运行管道
# documents = SimpleDirectoryReader("./data/paul").load_data()
# nodes = pipeline.run(documents)
nodes = pipeline.run(documents=[Document.example()])
print("-------node-----------\n",nodes)
index = VectorStoreIndex(nodes)
print("-------index-----------\n",index)
query_engine = index.as_query_engine()
response = query_engine.query("Where is the capital of France?")
print("-------response-----------\n",response)