"""Embedding 入门测试：把三句话转换成三个向量。"""

from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

texts = [
    "FastAPI 是一个 Python Web 框架",
    "使用 Python 开发后端接口",
    "今天晚上吃火锅",
]
vectors = embeddings.embed_documents(texts)

print(f"文字数量：{len(texts)}")
print(f"向量数量：{len(vectors)}")
print(f"每个向量的维度：{len(vectors[0])}")
print(f"第一个向量的前 5 个数字：{vectors[0][:5]}")

