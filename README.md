# Personal RAG

一个适合初学者学习的个人 PDF RAG 知识库，使用 FastAPI、LangChain、DeepSeek、BGE、Chroma 和 SQLite。

## 目录结构

```text
Personal_RAG/
├── backend/               FastAPI 与 RAG 后端代码
├── frontend/index.html    独立前端页面
├── data/                  PDF、SQLite 和 Chroma 运行数据
├── examples/              Agent、Embedding、向量库学习示例
├── docs/                  启动、项目介绍和复盘文档
├── .env.example           环境变量模板
└── README.md
```

这次整理只调整代码位置和导入路径，没有改变原来的 RAG 流程。后端没有继续拆成复杂的 `routers/services` 多层目录，便于按学习顺序阅读。

## 快速启动

在项目根目录打开两个 PowerShell。

后端：

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

前端：

```powershell
.venv\Scripts\python.exe -m http.server 5500 --directory frontend
```

访问：

- 前端：http://127.0.0.1:5500/
- Swagger：http://127.0.0.1:8000/docs

## 推荐阅读顺序

1. [启动说明](docs/01-启动说明.md)
2. [项目介绍与实现思路](docs/02-项目介绍.md)
3. [复盘总结与面试问答](docs/03-复盘总结.md)

后端代码建议依次阅读：`schemas.py → document_service.py → vector_store_service.py → agent_service.py → main.py`。

