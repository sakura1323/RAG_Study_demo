"""接口数据结构：规定客户端怎么传，服务端怎么返回。"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="用户的问题")


class ChatResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    characters: int
    preview: str
    chunks: int
    chunk_preview: str
    chunk_contents: list[str]
    stored_chunks: int


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, description="要搜索的问题")
    k: int = Field(default=3, ge=1, le=10, description="返回结果数量")


class SearchResult(BaseModel):
    content: str
    source: str
    page: int
    chunk_index: int


class SearchResponse(BaseModel):
    results: list[SearchResult]


class RagRequest(BaseModel):
    question: str = Field(min_length=1, description="需要知识库回答的问题")
    k: int = Field(default=3, ge=1, le=10, description="使用的参考文本数量")


class RagResponse(BaseModel):
    answer: str
    sources: list[SearchResult]


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_path: str
    pages: int
    characters: int
    chunk_count: int
    created_at: str


class DeleteResponse(BaseModel):
    message: str
    deleted_chunks: int

