# 文档节点解析器和文档文本分割器示例
from llama_index.core import Document, VectorStoreIndex, Settings
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


# -------基于文件的节点解析器--------
# 根据被解析内容的类型（JSON、Markdown 等）创建节点
# SimpleFileNodeParser自动针对每种文件选择最佳的节点解析器

from llama_index.core.node_parser import SimpleFileNodeParser,MarkdownNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path

md_docs = FlatReader().load_data(Path("./data/pathway_readme.md"))
parser1 = SimpleFileNodeParser()
md_nodes1 = parser1.get_nodes_from_documents(md_docs)

parser2 = MarkdownNodeParser()
md_nodes2 = parser2.get_nodes_from_documents(md_docs)



# ------基于文本的文本分割器---------
# 根据编写语言拆分原始代码文本。

from llama_index.core.node_parser import CodeSplitter,SentenceSplitter,TokenTextSplitter

code_docs = FlatReader().load_data(Path("./agent.py"))
splitter1 = CodeSplitter(
    language="python",
    chunk_lines=40,  # 每个块的行数
    chunk_lines_overlap=15,  # 块之间的行重叠
    max_chars=1500,  # 每个块的最大字符数
)
nodes1 = splitter1.get_nodes_from_documents(code_docs)

splitter2 = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)
nodes2 = splitter2.get_nodes_from_documents(md_docs)

splitter3 = TokenTextSplitter(
    chunk_size=1024,
    chunk_overlap=20,
    separator=" ",
)
nodes3 = splitter3.get_nodes_from_documents(md_docs)



'''
txt_docs = Document.from_files(Path("data"), file_types=["txt"])
text_list = ["text1", "text2"]
documents = [Document(text=t) for t in text_list]
'''