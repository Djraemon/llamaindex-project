# 路径检索器Pathway
# 后部分还没跑出来

# ----------一个演示demo----------------
from llama_index.retrievers.pathway import PathwayRetriever

retriever = PathwayRetriever(
    url="https://demo-document-indexing.pathway.stream"
)
print(retriever.retrieve(str_or_query_bundle="what is pathway"))

print("------------------------")

# ---------构建在查询引擎中的使用------------
from llama_index.core.query_engine import RetrieverQueryEngine
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

query_engine = RetrieverQueryEngine.from_args(
    retriever,
)
response = query_engine.query("Tell me about Pathway")
print(str(response))

print("------------------------")

# ----------构建自己的数据处理流程--------------

# 定义Pathway跟踪的数据源
import pathway as pw
data_sources = []
data_sources.append(    
    pw.io.fs.read(     # 这将创建一个`pathway`连接器，用于跟踪./data目录中的所有文件    
        "./data/paul",        
        format="binary",        
        mode="streaming",       
        with_metadata=True,    
    )  
)
# 这将创建一个用于跟踪Google Drive中文件的连接器。
# 请按照https://pathway.com/developers/tutorials/connectors/gdrive-connector/上的说明获取凭据
# data_sources.append(
#     pw.io.gdrive.read(
#       object_id="17H4YpBOAKQzEJ93xmC2z170l0bP2npMy", 
#       service_user_credentials_file="credentials.json", 
#       with_metadata=True))

# 创建文档索引流水线
from pathway.xpacks.llm.vector_store import VectorStoreServer
from llama_index.core.node_parser import TokenTextSplitter

embed_model = Settings.embed_model
# 定义transformations_example列表，包括TokenTextSplitter对象和embed_model对象
transformations_example = [   
    TokenTextSplitter(        
        chunk_size=150,        
        chunk_overlap=10,        
        separator=" ",    ),    
    embed_model,]
# 通过VectorStoreServer.from_llamaindex_components创建processing_pipeline对象
processing_pipeline = VectorStoreServer.from_llamaindex_components(    
    *data_sources,    
    transformations=transformations_example,
    )

# 定义Pathway的主机和端口
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8754
# `threaded`以分离模式运行pathway，当从终端或容器中运行时，必须将其设置为False
# 有关`with_cache`的更多信息，请查看https://pathway.com/developers/api-docs/persistence-api

'''processing_pipeline.run_server(    
    host=PATHWAY_HOST, 
    port=PATHWAY_PORT, pip install docling
pip install docling

    with_cache=False, 
    threaded=True
)'''

# 将检索器连接到自定义管道
from llama_index.retrievers.pathway import PathwayRetriever

retriever = PathwayRetriever(host=PATHWAY_HOST, port=PATHWAY_PORT)
retriever.retrieve(str_or_query_bundle="what is pathway")
