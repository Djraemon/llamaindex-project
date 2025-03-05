# 元数据过滤器+自动检索器 (不过使用的是Vectara索引器)
# 为每个文档打上正确的元数据标签。
# 在查询时，使用自动检索来推断元数据过滤器，并通过查询字符串进行语义搜索
# 还没调试出来
from llama_index.core.schema import TextNode
from llama_index.core.indices.managed.types import ManagedIndexQueryMode
from llama_index.indices.managed.vectara import VectaraIndex
from llama_index.indices.managed.vectara import VectaraAutoRetriever

from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
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


from sample import movie_nodes
import os
# 这个还没弄
os.environ["VECTARA_API_KEY"] = "<YOUR_VECTARA_API_KEY>"
os.environ["VECTARA_CORPUS_ID"] = "<YOUR_VECTARA_CORPUS_ID>"
os.environ["VECTARA_CUSTOMER_ID"] = "<YOUR_VECTARA_CUSTOMER_ID>"

index = VectaraIndex(nodes=movie_nodes)
vector_store_info = VectorStoreInfo(    
    content_info="关于电影的信息",    
    metadata_info=[        
        MetadataInfo(            
            name="genre",            
            description="""                
                电影的类型。                
                可选值为 ['science fiction', 'fantasy', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']            
            """,            
            type="字符串",        
        ),       
        MetadataInfo(            
            name="year",            
            description="电影上映的年份",            
            type="整数",        
        ),        
        MetadataInfo(            
            name="director",            
            description="电影导演的姓名",            
            type="字符串",        
        ),        
        MetadataInfo(            
            name="rating",            
            description="电影的评分，范围为1-10",            
            type="浮点数",        
            ),    
    ],    
)

from llama_index.indices.managed.vectara import VectaraAutoRetriever

llm = Settings.llm

retriever = VectaraAutoRetriever(
    index,
    vector_store_info=vector_store_info,
    llm=llm,
    verbose=True,
)

# 查询
print(retriever.retrieve("movie directed by Greta Gerwig"))