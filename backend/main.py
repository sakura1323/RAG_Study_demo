"""FastAPI 入口：只负责接收请求和返回响应。"""

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .agent_service import SimpleAgent
from .config import UPLOAD_DIRECTORY
from .database_service import (
    add_document,
    delete_document_record,
    get_document,
    list_documents,
)
from .document_service import parse_pdf
from .schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentInfo,
    RagRequest,
    RagResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    UploadResponse,
)
from .vector_store_service import add_chunks, delete_document_chunks, search_chunks


app = FastAPI(title="Personal RAG 学习项目", version="0.1.0")

# 独立前端运行在 5500 端口，需要允许它访问 8000 端口的后端。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)
agent = SimpleAgent()


@app.get("/")
def home() -> dict[str, str]:
    """健康检查；前端页面由 5500 端口单独提供。"""
    return {"message": "Personal RAG API is running", "docs": "/docs"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    answer = agent.invoke(request.question)
    return ChatResponse(answer=answer)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        agent.astream(request.question),
        media_type="text/plain; charset=utf-8",
    )


@app.post("/upload", response_model=UploadResponse)
def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    document_id = str(uuid4())
    try:
        result, chunks, saved_path = parse_pdf(file, document_id)
        stored_chunks = add_chunks(
            chunks, document_id, str(result["filename"])
        )
        add_document(
            document_id=document_id,
            filename=str(result["filename"]),
            file_path=str(saved_path),
            pages=int(result["pages"]),
            characters=int(result["characters"]),
            chunk_count=stored_chunks,
        )
        result["document_id"] = document_id
        result["stored_chunks"] = stored_chunks
        return UploadResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """搜索最相关的 PDF 文本块，本接口暂时不调用 DeepSeek。"""
    documents = search_chunks(request.question, request.k)
    results = [
        SearchResult(
            content=document.page_content,
            source=str(document.metadata.get("source", "未知文件")),
            # PyPDFLoader 的 page 从 0 开始，展示时转换成正常页码。
            page=int(document.metadata.get("page", 0)) + 1,
            chunk_index=int(document.metadata.get("chunk_index", 0)),
        )
        for document in documents
    ]
    return SearchResponse(results=results)


@app.post("/rag", response_model=RagResponse)
def rag(request: RagRequest) -> RagResponse:
    """检索相关 chunks，再让 DeepSeek 根据这些资料回答。"""
    documents = search_chunks(request.question, request.k)
    answer = agent.answer_with_context(request.question, documents)
    sources = [
        SearchResult(
            content=document.page_content,
            source=str(document.metadata.get("source", "未知文件")),
            page=int(document.metadata.get("page", 0)) + 1,
            chunk_index=int(document.metadata.get("chunk_index", 0)),
        )
        for document in documents
    ]
    return RagResponse(answer=answer, sources=sources)


@app.post("/rag/stream")
async def rag_stream(request: RagRequest) -> StreamingResponse:
    """先检索知识库，再把 DeepSeek 的回答逐块发送给客户端。"""
    documents = search_chunks(request.question, request.k)
    return StreamingResponse(
        agent.astream_with_context(request.question, documents),
        media_type="text/plain; charset=utf-8",
    )


@app.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[DocumentInfo]:
    """查看已经保存到知识库的原始 PDF。"""
    return [DocumentInfo(**item) for item in list_documents()]


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str) -> DeleteResponse:
    """同时删除 Chroma chunks、原始 PDF 和 SQLite 记录。"""
    document = get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    deleted_chunks = delete_document_chunks(document_id)

    # 文件名由后端 UUID 生成，并再次确认路径位于 uploads 目录内。
    file_path = Path(str(document["file_path"])).resolve()
    uploads_path = UPLOAD_DIRECTORY.resolve()
    if file_path.parent == uploads_path:
        file_path.unlink(missing_ok=True)

    delete_document_record(document_id)
    return DeleteResponse(message="文档删除成功", deleted_chunks=deleted_chunks)

