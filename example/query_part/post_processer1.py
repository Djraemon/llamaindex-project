# 基于文件的后处理
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path

reader = FlatReader()
file1 = reader.load_data(Path("./data/pathway_readme.md"))
file2 = reader.load_data(Path("./data/pathway_readme.md"))
print(file1[0].metadata)
print(file1[0])
print("--------")
print(file2[0].metadata)
print(file2[0])

parser = SimpleFileNodeParser()
file1 = parser.get_nodes_from_documents(file1)
file2 = parser.get_nodes_from_documents(file2)
print(file1[0].metadata)
print(file1[0].text)
print(file1[1].metadata)
print(file1[1].text)
print("----")
print(file2[0].metadata)
print(file2[0].text)