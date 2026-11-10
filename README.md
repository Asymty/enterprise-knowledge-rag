# Enterprise Knowledge RAG

企业知识库问答系统。上传企业制度、操作手册或业务文档后，可以用自然语言提问，并查看回答所依据的文件和摘录。适合查询差旅报销、审批流程和内部操作规范。

## 功能

- 支持 PDF、Word（DOCX）、Markdown 和 TXT 文档，上传后自动解析、分块并建立索引。
- 混合检索结合向量相似度与 BM25 关键词匹配，兼顾语义相关性和专有术语。
- 提供普通问答、流式输出和多轮会话，回答附带来源。
- 前端使用 Next.js、React 和 TypeScript。后端使用 FastAPI，向量索引存储在 Qdrant。

## 本地运行

需要 Python 3.13、Node.js 20.9 或更高版本，以及可访问的 Qdrant 实例。以下命令适用于 PowerShell。

**后端**（从仓库根目录执行）

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `backend/.env`，填写 Qdrant 地址及所需凭据，并配置要使用的生成模型。使用本地 Ollama 时，先启动 Ollama 并准备对应模型。清空不用的 API Key 和 Redis 示例值。首次启动本地嵌入模型时可能需要下载模型文件。

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

**前端**（另开终端，从仓库根目录执行）

```powershell
cd frontend
npm ci
$env:NEXT_PUBLIC_API_URL = 'http://localhost:8001/api'
npm run dev
```

浏览器打开 `http://localhost:3000`，上传一份 Markdown 制度文档，再提一个能从文档中回答的问题，例如“上海员工出差住宿报销上限是多少？”。后端接口文档位于 `http://127.0.0.1:8001/docs`。

## 代码目录

| 目录 | 内容 |
| --- | --- |
| `backend/app/` | 文档导入、检索、生成和 API |
| `backend/tests/` | 后端测试与检索评估 |
| `frontend/src/` | 上传、聊天和来源展示 |

## 许可证

[MIT License](LICENSE)
