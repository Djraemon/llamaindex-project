# 递归检索器+节点引用
from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Document,VectorStoreIndex,Settings
import json

loader = PDFReader()
docs0 = loader.load_data(file=Path("./data/llama2.pdf"))
doc_text = "\n\n".join([d.get_content() for d in docs0])  # combine all documents into a single string
docs = [Document(text=doc_text)]

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.siliconflow import SiliconFlow

api_key = "sk-yrmmgcztvxoijeigwmhqqohhafaolagrmlffjuhiifmrdlcg"
api_base_url =  "https://api.siliconflow.cn/v1/"

# 初始化LLM + 分割器
Settings.llm = SiliconFlow(
    api_key=api_key,
    model="Qwen/Qwen2.5-7B-Instruct")
Settings.embed_model = SiliconFlowEmbedding(
    api_key=api_key,
    model_name="netease-youdao/bce-embedding-base_v1")

node_parser = SentenceSplitter(chunk_size=1024)
base_nodes = node_parser.get_nodes_from_documents(docs)
#for idx, node in enumerate(base_nodes):    node.id_ = f"node-{idx}"

'''base_index = VectorStoreIndex(base_nodes, embed_model=Settings.embed_model)
base_retriever = base_index.as_retriever(similarity_top_k=2)
retrievals = base_retriever.retrieve(
    "Can you tell me about the key concepts for safety finetuning"
)
for n in retrievals:
    print(n) # 打印检索到的节点

# 初始化查询引擎
query_engine_base = RetrieverQueryEngine.from_args(base_retriever, llm=Settings.llm)
response = query_engine_base.query("Can you tell me about the key concepts for safety finetuning")
print(str(response))
'''

# ------------以上为基准检索器-----------------
# ------------以下为块引用-----------------
sub_chunk_sizes = [128, 256, 512] # 定义三种不同的文本块大小，用于多粒度文本分割
# 创建基于不同块大小的句子分割器列表
sub_node_parsers = [  
    # 设置块之间的重叠字符数chunk_overlap，保持上下文连贯  
    SentenceSplitter(chunk_size=c, chunk_overlap=20) for c in sub_chunk_sizes 
]
all_nodes = []
for base_node in base_nodes:    # 遍历基准节点
    for n in sub_node_parsers:  # 对每个基准节点使用不同的分割器 进行多级分割   
        # 分割基准节点，分割为更小的子节点
        sub_nodes = n.get_nodes_from_documents([base_node])    
        # 将子节点转换为索引节点，并添加到all_nodes列表中
        sub_inodes = [            
            IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes  # 并建立父子节点关联关系      
        ]        
        all_nodes.extend(sub_inodes)    
    # 将原始基准节点也转换为索引节点  
    original_node = IndexNode.from_text_node(base_node, base_node.node_id)    
    all_nodes.append(original_node)

all_nodes_dict = {n.id_: n for n in all_nodes}

vector_index_chunk = VectorStoreIndex(all_nodes, embed_model=Settings.embed_model)
vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=2)

retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True,
)


nodes = retriever_chunk.retrieve(
    "Can you tell me about the key concepts for safety finetuning"
)
for node in nodes:
    print(node)