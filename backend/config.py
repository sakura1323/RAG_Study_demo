"""集中保存项目路径，避免从不同目录启动时找不到数据。"""

from pathlib import Path


# config.py 在 backend 目录中，因此 parent.parent 就是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
UPLOAD_DIRECTORY = DATA_DIRECTORY / "uploads"
DATABASE_PATH = DATA_DIRECTORY / "knowledge.db"
CHROMA_DIRECTORY = DATA_DIRECTORY / "chroma_db"
ENV_PATH = PROJECT_ROOT / ".env"

UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)

