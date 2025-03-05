import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# 检查存储是否已经存在
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # 加载文档并创建索引
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # 为以后存储它
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # 加载现有的索引
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# 无论哪种方式，现在我们都可以查询索引
query_engine = index.as_query_engine()
response = query_engine.query("作者在成长过程中做了什么？")
print(response)