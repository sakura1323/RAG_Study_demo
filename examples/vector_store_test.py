"""向量库入门测试：存入三句话，再按语义检索。"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
documents = [
    Document(page_content="FastAPI 是一个 Python Web 框架", metadata={"id": 1}),
    Document(page_content="使用 Python 开发后端接口", metadata={"id": 2}),
    Document(page_content="今天晚上吃火锅", metadata={"id": 3}),
]
vector_store = Chroma(
    collection_name="lesson_vector_store",
    embedding_function=embeddings,
    persist_directory="./data/chroma_db",
)
vector_store.add_documents(
    documents=documents,
    ids=["lesson-1", "lesson-2", "lesson-3"],
)

question = "怎样使用 Python 创建 Web 接口？"
results = vector_store.similarity_search(question, k=2)
print(f"用户问题：{question}")
print(f"数据库中的文档数量：{len(vector_store.get()['ids'])}")
for index, document in enumerate(results, start=1):
    print(f"第 {index} 名：{document.page_content}")

