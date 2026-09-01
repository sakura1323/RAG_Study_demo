"""SQLite 服务：保存和管理知识库文档目录。"""

import sqlite3
from datetime import datetime, timezone

from .config import DATABASE_PATH



def _connect() -> sqlite3.Connection:
    """创建数据库连接，并让查询结果可以通过字段名读取。"""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database() -> None:
    """第一次运行时自动创建 documents 表。"""
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                pages INTEGER NOT NULL,
                characters INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def add_document(
    document_id: str,
    filename: str,
    file_path: str,
    pages: int,
    characters: int,
    chunk_count: int,
) -> None:
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO documents
            (id, filename, file_path, pages, characters, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                filename,
                file_path,
                pages,
                characters,
                chunk_count,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def list_documents() -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_document(document_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_document_record(document_id: str) -> None:
    with _connect() as connection:
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))


init_database()

