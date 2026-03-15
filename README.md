# 法律问答机器人

基于 LangChain 和 RAG 的《治安管理处罚法》智能问答系统。

## 功能

- **智能法律问答**：基于《中华人民共和国治安管理处罚法》的专业问答系统
- **混合检索**：向量检索 + BM25，提高检索准确度
- **多轮对话**：支持上下文记忆的连续对话
- **Web API**：提供 HTTP 接口，方便集成

## 技术栈

- **FastAPI** - Web 框架
- **LangChain** - LLM 应用开发框架
- **Chroma** - 向量数据库
- **Ollama** - 本地嵌入模型
- **NVIDIA AI** - 大语言模型服务

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 NVIDIA API Key：

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

### 3. 启动 Ollama 服务

```bash
# 如果 Ollama 还未安装，访问 https://ollama.ai 下载安装

# 启动 Ollama 服务
ollama serve

# 拉取嵌入模型（新终端）
ollama pull bge-m3:latest
```

### 4. 初始化数据库

```bash
python scripts/init_db.py
```

首次运行会自动从 PDF 切分文档并创建向量数据库，之后会直接加载已有数据库。

### 5. 启动服务

```bash
# 方式 1: 直接运行
python app/main.py

# 方式 2: 使用 uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

服务启动后访问：
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## API 使用

### 健康检查

```bash
curl http://localhost:8000/health
```

响应：
```json
{
  "status": "healthy",
  "service": "legal-qa-bot",
  "nvidia_configured": true
}
```

### 发起问答

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "阻碍执行紧急任务的消防车会如何处罚？",
    "session_id": "user123"
  }'
```

响应：
```json
{
  "answer": "根据《治安管理处罚法》第五十条规定，阻碍执行紧急任务的消防车、救护车、工程抢险车、警车等车辆通行的，处警告或者二百元以下罚款..."
}
```

### 多轮对话

```bash
# 第一轮
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是扰乱公共秩序的行为？",
    "session_id": "user123"
  }'

# 第二轮（保持上下文）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "这种行为一般会怎么处罚？",
    "session_id": "user123"
  }'
```

## 项目结构

```
fastapi-langchain-portfolio/
├── app/
│   ├── main.py              # FastAPI 入口和路由
│   └── rag/
│       ├── chains.py        # LangChain 链和提示词
│       ├── services.py      # 文档加载和切分
│       ├── retriever.py     # 向量数据库管理
│       └── utils.py         # 文本清洗工具
├── data/
│   └── 中华人民共和国治安管理处罚法.pdf
├── scripts/
│   └── init_db.py           # 数据库初始化脚本
├── requirements.txt
├── .env
└── README.md
```

## 开发说明

### 运行单个模块测试

```bash
# 测试文档切分
python app/rag/services.py

# 测试检索
python app/rag/retriever.py

# 测试 RAG 链
python app/rag/chains.py
```

### 重建数据库

如果 PDF 内容有更新，需要重建数据库：

```bash
# 删除旧数据库
rm -rf data/chroma_db

# 重新初始化
python scripts/init_db.py
```

## 注意事项

1. **首次启动慢**：首次请求时会初始化向量数据库，需要 10-30 秒
2. **Ollama 服务**：确保 Ollama 服务正在运行
3. **NVIDIA API Key**：需要在 `.env` 文件中配置
4. **对话历史**：当前版本对话历史保存在内存中，服务重启后会丢失

## 许可证

MIT License
