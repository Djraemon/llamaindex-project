# 查询重写（自定义，生成多个查询）
# 将查询重写为多个查询，然后针对检索器执行所有这些查询
# 进行查询重写，可以为[集成检索]和[融合]生成多个查询，从而获得更高质量的检索结果。

from llama_index.llms.siliconflow import SiliconFlow
api_key = "sk-yrmmgcztvxoijeigwmhqqohhafaolagrmlffjuhiifmrdlcg"
llm = SiliconFlow(api_key=api_key, model="Qwen/Qwen2.5-7B-Instruct")


from llama_index.core import PromptTemplate
query_gen_str = """\
    您是一个乐于助人的助手，根据单个输入查询生成多个搜索查询。
    生成{num_queries}个搜索查询，每行一个，与以下输入查询相关：
    查询：{query}
    查询：
    """
query_gen_prompt = PromptTemplate(query_gen_str)


def generate_queries(query: str, llm, num_queries: int = 4):    
    response = llm.predict(        
        query_gen_prompt, 
        num_queries=num_queries, 
        query=query    
        )    
    
    # 假设LLM适当地将每个查询放在一行上    
    queries = response.split("\n")    
    queries_str = "\n".join(queries)    
    print(f"生成的查询：\n{queries_str}")    
    return queries




from llama_index.core import SummaryIndex, SimpleDirectoryReader, Settings,Document
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

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import FaithfulnessEvaluator

# 加载文档
documents = SimpleDirectoryReader("./data/paul").load_data()
# 创建向量索引
index = VectorStoreIndex.from_documents(documents)
# 生成查询
query_str = "What happened at Interleaf and Viaweb?"
queries = generate_queries(query_str, llm)
query_engine = index.as_query_engine()
evaluator = FaithfulnessEvaluator(llm=llm)

# 执行查询和评估
response = []
for i, query in enumerate(queries):
    print("------------------------------")
    response.append(query_engine.query(query_str))
    print("response[{}]: ".format(i), response[i],'\n')
    result = evaluator.evaluate_response(response=response[i])
    # 打印分数
    print("分数：", result.score)
    # 打印通过情况，默认相似度阈值为0.8
    print("通过：", result.passing) 



