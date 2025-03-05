# 1. 将提示和LLM链接在一起
from llama_index.core import SimpleDirectoryReader, Settings
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

from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate

# 尝试链接基本提示
prompt_str = "请生成与{movie_name}相关的电影"
prompt_tmpl = PromptTemplate(prompt_str)
llm = Settings.llm

# 创建查询管道
pipeline = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)
movies = pipeline.run(movie_name="The Departed")
print(str(movies))

''' 让我们添加一些有趣的后续提示
prompt_str2 = """\这里有一些文本：{text}你能用每部电影的摘要重写这个吗？"""
prompt_tmpl2 = PromptTemplate(prompt_str2)
llm_c = llm.as_query_component(streaming=True)
pipeline = QueryPipeline(    chain=[prompt_tmpl, llm_c, prompt_tmpl2, llm_c], verbose=True)
# p = QueryPipeline(chain=[prompt_tmpl, llm_c], verbose=True)
movies = pipeline.run(movie_name="The Departed")
print(str(movies))
'''