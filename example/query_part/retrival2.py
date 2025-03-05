# 路由器检索器 用到的模型选择器仅支持openai模型
# 没跑出来
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    SimpleKeywordTableIndex,
    Settings,
)
from llama_index.core import SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
from llama_index.llms.siliconflow import SiliconFlow

api_key = "sk-yrmmgcztvxoijeigwmhqqohhafaolagrmlffjuhiifmrdlcg"
api_base_url =  "https://api.siliconflow.cn/v1/"

# 加载文档
documents = SimpleDirectoryReader("./data/paul/").load_data()
# 初始化LLM + 分割器
Settings.embed_model = SiliconFlowEmbedding(api_key=api_key,
                                            model_name="netease-youdao/bce-embedding-base_v1")
Settings.llm = SiliconFlow(
    api_key=api_key,
    model="Qwen/Qwen2.5-7B-Instruct")

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# 初始化存储上下文（默认情况下是内存中的）
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# 定义 不同类型的索引
summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index  = VectorStoreIndex(nodes, storage_context=storage_context)
keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

# 定义 针对不同类型索引的检索器
list_retriever = summary_index.as_retriever()
vector_retriever = vector_index.as_retriever()
keyword_retriever = keyword_index.as_retriever()

from llama_index.core.tools import RetrieverTool

list_tool = RetrieverTool.from_defaults(
    retriever=list_retriever,
    description=(
        "Will retrieve all context from Paul Graham's essay on What I Worked"
        " On. Don't use if the question only requires more specific context."
    ),
)
vector_tool = RetrieverTool.from_defaults(
    retriever=vector_retriever,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On."
    ),
)
keyword_tool = RetrieverTool.from_defaults(
    retriever=keyword_retriever,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On (using entities mentioned in query)"
    ),
)

#from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.selectors import ( # PydanticSingleSelector仅支持openai模型
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.selectors.prompts import DEFAULT_SINGLE_SELECT_PROMPT

retriever = RouterRetriever(
    retriever_tools=[list_tool, vector_tool, keyword_tool],
    selector=PydanticSingleSelector( 
        Settings.llm, DEFAULT_SINGLE_SELECT_PROMPT, 
    ),
)

# 将从作者的生活中检索所有上下文
gained_nodes = retriever.retrieve(    "你能给我关于作者生活的所有上下文吗？")
for node in gained_nodes:    
    display_source_node(node)