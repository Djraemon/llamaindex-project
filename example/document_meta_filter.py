# 文档元数据的选择性使用
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

document = Document(
    text="这是一个超级定制的文档",
    metadata={
        "file_name": "super_secret_document.txt",
        "category": "finance",
        "author": "LlamaIndex",
    },
    excluded_llm_metadata_keys=["file_name"],
    metadata_seperator="::",
    metadata_template="{key}=>{value}",
    text_template="元数据: {metadata_str}\n-----\n内容: {content}",
)

print(
    "LLM看到的内容: \n",
    document.get_content(metadata_mode=MetadataMode.LLM),
)
print(
    "嵌入模型看到的内容: \n",
    document.get_content(metadata_mode=MetadataMode.EMBED),
)