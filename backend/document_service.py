"""文档服务：负责上传文件的临时保存和 PDF 文字解析。"""

import shutil
from pathlib import Path

from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import UPLOAD_DIRECTORY



def parse_pdf(
    file: UploadFile,
    document_id: str,
) -> tuple[dict[str, str | int | list[str]], list[Document], Path]:
    filename = file.filename or "unknown.pdf"
    if Path(filename).suffix.lower() != ".pdf":
        raise ValueError("目前只支持 PDF 文件")

    saved_path = UPLOAD_DIRECTORY / f"{document_id}.pdf"
    try:
        # 使用 document_id 保存原始 PDF，避免同名文件互相覆盖。
        with saved_path.open("wb") as saved_file:
            shutil.copyfileobj(file.file, saved_file)

        documents = PyPDFLoader(str(saved_path)).load()
        full_text = "\n".join(document.page_content for document in documents)

        # 把较长的页面继续切成小段，方便后续进行向量检索。
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,     # 每块最多约 500 个字符
            chunk_overlap=50,   # 相邻两块重复约 50 个字符
        )
        chunks = text_splitter.split_documents(documents)

        result = {
            "filename": filename,
            "pages": len(documents),
            "characters": len(full_text),
            "preview": full_text[:500],
            "chunks": len(chunks),
            "chunk_preview": chunks[0].page_content if chunks else "",
            # 学习阶段返回全部切块，方便在 Swagger 中观察切分结果。
            "chunk_contents": [chunk.page_content for chunk in chunks],
        }
        return result, chunks, saved_path
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise ValueError(f"PDF 解析失败：{exc}") from exc
    finally:
        file.file.close()

