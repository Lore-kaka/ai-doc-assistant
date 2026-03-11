# LangChain-Portfolio

基于 LangChain 构建的智能法律问答系统，对《中华人民共和国治安管理处罚法》的智能检索与解答。

## 项目简介

本项目是一个使用 LangChain 框架实现的 RAG（检索增强生成）应用，通过向量数据库检索相关法律条文，结合大语言模型生成准确、专业的法律解答。系统基于 NVIDIA AI 和 Ollama 的嵌入模型，提供高效的法律咨询服务。

## 核心功能

- **智能法律问答**：基于《中华人民共和国治安管理处罚法》的专业问答系统
- **文档检索**：使用 Chroma 向量数据库进行高效的文档检索
- **语义理解**：通过 OllamaEmbeddings (qwen3-embedding:0.6b) 实现语义嵌入
- **大语言模型**：集成 NVIDIA AI 的 GPT-OSS-120B 模型生成专业回答

## 技术栈

- **FastAPI**: 现代化的 Web 框架
- **LangChain**: 大语言模型应用开发框架
- **Chroma**: 向量数据库，用于存储和检索文档嵌入
- **NVIDIA AI**: 提供大语言模型服务
- **Ollama**: 本地嵌入模型服务

## 项目结构

```
fastapi-langchain-portfolio/
├── app/
│   ├── __init__.py
│   └── rag/
│       ├── __init__.py
│       ├── chains.py        # LangChain 链式调用逻辑
│       ├── services.py      # 文档加载、分割和向量检索服务
│       └── chroma_db/       # Chroma 向量数据库存储
├── main.py                  # 应用入口
├── pyproject.toml          # 项目依赖配置
└── README.md               # 项目说明文档
```

## 环境要求

- Python >= 3.12
- Ollama (用于本地嵌入模型)
- NVIDIA API Key

## 安装步骤

1. 克隆项目仓库
```bash
git clone <repository-url>
cd fastapi-langchain-portfolio
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
创建 `.env` 文件并添加：
```
NVIDIA_API_KEY=your_nvidia_api_key_here
```

4. 启动 Ollama 服务
```bash
ollama serve
```

5. 拉取嵌入模型
```bash
ollama pull qwen3-embedding:0.6b
```

## 使用方法

### 运行主程序
```bash
python main.py
```

### 运行 RAG 链
```bash
python app/rag/chains.py
```

### 运行文档检索服务
```bash
python app/rag/services.py
```

## 主要模块说明

### chains.py
- 定义了专业的法律助手提示词模板
- 配置 NVIDIA AI 的 GPT-OSS-120B 模型
- 实现了检索增强生成链（RAG Chain）

### services.py
- 加载和分割法律文档
- 创建和管理 Chroma 向量数据库
- 提供文档检索接口

## 特点

- **准确性**：严格基于法律条文回答，不编造信息
- **专业性**：使用专业的法律语言，保持客观中立
- **引用明确**：回答时准确引用相关条文
- **高效检索**：基于向量数据库的语义检索

## 注意事项

- 首次运行时会自动创建向量数据库，后续运行会直接加载已有数据库
- 确保 Ollama 服务正常运行
- 确保 NVIDIA API Key 配置正确

## 许可证

MIT License


