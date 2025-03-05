# 存储文档层次结构（摘要-》基准块）+递归检索
# 嵌入文档摘要，并将其映射到每个文档的原始块集合。
# 在查询时，进行递归检索，先获取摘要，然后再获取文档。
# 未完成，需要修改
from llama_index.core.schema import IndexNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex,SummaryIndex
from sample import wiki_titles,wiki_metadatas

from llama_index.core import Settings,SimpleDirectoryReader
from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
from llama_index.llms.siliconflow import SiliconFlow
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager

from pathlib import Path
import requests
api_key = "sk-yrmmgcztvxoijeigwmhqqohhafaolagrmlffjuhiifmrdlcg"
api_base_url =  "https://api.siliconflow.cn/v1/"
Settings.embed_model = SiliconFlowEmbedding(
    api_key=api_key,
    model_name="netease-youdao/bce-embedding-base_v1")
Settings.llm = SiliconFlow(
    api_key=api_key,
    model="Qwen/Qwen2.5-7B-Instruct")
splitter = SentenceSplitter(chunk_size=256)
callback_manager = CallbackManager([LlamaDebugHandler()])

'''
# 加载所有维基文档
docs_dict = {}
for wiki_title in wiki_titles:    
    doc = SimpleDirectoryReader(        
        input_files=[f"data/{wiki_title}.txt"]    
    ).load_data()[0]    
    doc.metadata.update(wiki_metadatas[wiki_title])    
    docs_dict[wiki_title] = doc

# 定义顶层节点和向量检索器
nodes = []
vector_query_engines = {}
vector_retrievers = {}
for wiki_title in wiki_titles:   
     # 构建向量索引    
     vector_index = VectorStoreIndex.from_documents(        
            [docs_dict[wiki_title]],        
            transformations=[splitter],        
            callback_manager=callback_manager,   
        )    
    # 定义查询引擎    
vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)    
vector_query_engines[wiki_title] = vector_query_engine    
vector_retrievers[wiki_title] = vector_index.as_retriever()    
# 保存摘要    
out_path = Path("summaries") / f"{wiki_title}.txt"    
if not out_path.exists():        
    # 使用LLM生成的摘要        
    summary_index = SummaryIndex.from_documents(            
        [docs_dict[wiki_title]], callback_manager=callback_manager        )        
    summarizer = summary_index.as_query_engine(            
    response_mode="tree_summarize", llm=llm        )        
    response = summarizer.aquery(f"给我一个关于{wiki_title}的摘要")        
    wiki_summary = response.response        
    Path("summaries").mkdir(exist_ok=True)        
    with open(out_path, "w") as fp:            
        fp.write(wiki_summary)    
else:        
    with open(out_path, "r") as fp:            
        wiki_summary = fp.read()    
        print(f"**{wiki_title}的摘要：{wiki_summary}")    


node = IndexNode(text=wiki_summary, index_id=wiki_title)    
nodes.append(node)
'''