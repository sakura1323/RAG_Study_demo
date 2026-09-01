"""向量库服务：负责 Embedding、Chroma 和 chunks 入库。"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import CHROMA_DIRECTORY

# 服务启动时加载一次模型，避免每次上传 PDF 都重复加载。
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu", "local_files_only": True},
    encode_kwargs={"normalize_embeddings": True},
)

# 使用单独的 collection 保存个人知识库，数据持久化到 data/chroma_db。
vector_store = Chroma(
    collection_name="personal_knowledge",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIRECTORY),
)


def add_chunks(chunks: list[Document], document_id: str, filename: str) -> int:
    """给 chunks 添加来源信息和固定 ID，然后存入 Chroma。"""
    chunk_ids = []

    for index, chunk in enumerate(chunks):
        # PyPDFLoader 原有的 page 会被保留；这里补充文件名和块编号。
        chunk.metadata["source"] = filename
        chunk.metadata["document_id"] = document_id
        chunk.metadata["chunk_index"] = index
        chunk_ids.append(f"{document_id}-chunk-{index}")

    if chunks:
        vector_store.add_documents(documents=chunks, ids=chunk_ids)

    return len(chunks)


def search_chunks(question: str, k: int = 3) -> list[Document]:
    """根据用户问题，从 Chroma 返回最相关的 k 个 chunks。"""
    return vector_store.similarity_search(question, k=k)


def delete_document_chunks(document_id: str) -> int:
    """删除某个文档在 Chroma 中的所有 chunks。"""
    data = vector_store.get(where={"document_id": document_id})
    chunk_ids = data["ids"]
    if chunk_ids:
        vector_store.delete(ids=chunk_ids)
    return len(chunk_ids)

