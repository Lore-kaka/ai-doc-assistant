import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_legal_text(text: str) -> str:
    """
    清洗法律文档文本中的噪音数据
    """
    # 1. 去除页码标识（如 "- 2 -", "-2-" 等）
    text = re.sub(r'-\s*\d+\s*-', '', text)
    # 2. 去除连续的多个空行，替换为单行
    text = re.sub(r'\n{2,}', '\n', text)
    # 3. 去除首尾空白字符
    return text.strip()

def process_legal_document(file_path: str):
    """
    加载并切分法律文档
    """
    # ==========================
    # 1. 加载文档
    # ==========================
    print("开始加载文档...")
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # ==========================
    # 2. 数据清洗
    # ==========================
    print("正在清洗文档数据...")
    for doc in docs:
        doc.page_content = clean_legal_text(doc.page_content)
        
    # 将清洗后的多页内容合并为一个完整长文本（有助于跨页条文的完整切分）
    full_text = "\n".join([doc.page_content for doc in docs])
    
    # ==========================
    # 3. 配置针对性切分器
    # ==========================
    # 这里的核心是使用了 (?=...) 前瞻断言，这样匹配到的“第X条”不会被当作分隔符删掉，而是保留在下一个 Chunk 的开头
    legal_separators = [
        r"\n(?=第[一二三四五六七八九十百零]+章\s+)", # 最高优先级：按章切分
        r"\n(?=第[一二三四五六七八九十百零]+节\s+)", # 次高优先级：按节切分
        r"\n(?=第[一二三四五六七八九十百零]+条\s+)", # 核心优先级：按条切分
        "。\n",                                        # 降级：段落句号
        "；\n",                                        # 降级：款项分号
        "。",                                          # 降级：普通句号
        "\n",
        " "
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=legal_separators,
        is_separator_regex=True, # 必须开启此项以支持正则分隔符
        chunk_size=800,          # 《治安管理处罚法》单条长度极少超过 800 字符
        chunk_overlap=50,        # 少量重叠保证极端情况下的上下文连贯
        length_function=len,
    )

    # ==========================
    # 4. 执行切分
    # ==========================
    print("开始执行结构化切分...")
    # 由于之前合并了全文本，这里直接 create_documents
    split_docs = text_splitter.create_documents([full_text])
    
    # 可以手动将原始文件的 metadata 重新赋给切分后的 docs
    for chunk in split_docs:
        chunk.metadata = {"source": file_path}

    return split_docs

# ==========================
# 运行测试
# ==========================
if __name__ == "__main__":
    file_path = "app/rag/中华人民共和国治安管理处罚法.pdf" # 替换为您的实际路径
    
    try:
        chunks = process_legal_document(file_path)
        print(f"\n✅ 处理完成！共生成 {len(chunks)} 个文本块 (Chunks)。\n")
        
        # 打印前几个 Chunk 检查效果
        print("--- 前 3 个 Chunk 示例 ---")
        for i in range(min(3, len(chunks))):
            print(f"【Chunk {i+1} 长度: {len(chunks[i].page_content)}】")
            print(chunks[i].page_content)
            print("-" * 40)
            
    except Exception as e:
        print(f"处理出错: {e}")