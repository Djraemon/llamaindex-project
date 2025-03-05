# 自定义查询阶段，实现查询流程（检索器，节点后处理器，响应合成器） demo
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

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor,KeywordNodePostprocessor
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import get_response_synthesizer

documents = SimpleDirectoryReader("data").load_data()
nodes = SentenceSplitter().get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)

# ------配置检索器-------
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10, # 检索时返回计算相似度最高的2个节点
)

# ----配置节点后处理器-------
# (还有其他子处理器可选)
node_postprocessors=[
    # 根据required_keywords和exclude_keywords过滤节点
    # KeywordNodePostprocessor(required_keywords=["CMU"],exclude_keywords=["Italy"]),
    # 根据相似度过滤节点
    SimilarityPostprocessor(similarity_cutoff=0.1) ]

# ------配置响应合成器------
response_synthesizer = get_response_synthesizer()

# ------组合查询引擎-------
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=node_postprocessors,
)

# 响应评估器
evaluator = FaithfulnessEvaluator(llm=Settings.llm)

# 查询
response = query_engine.query("你是谁？")
print(response)
