from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import os 
import re
import shutil
from dotenv import load_dotenv
load_dotenv()

embedding = OllamaEmbeddings(
    model = 'qwen3-embedding:0.6b'
)
# embedding = DashScopeEmbeddings(
#     model="text-embedding-v4",
#     # other params...
# )




def clean_legal_text(text: str) -> str:
    """
    清洗法律文档文本中的噪音数据
    """
# 1. 统一处理特殊空格 (如 \xa0 全角空格)
    text = text.replace('\xa0', ' ')
    
    # 2. 匹配并去除各类页码标识：如 "- 1 -", "— 1 —", "-1-", "—1—"
    text = re.sub(r'[-—]\s*\d+\s*[-—]', '', text)
    
    # 3. 压缩连续的空行（包括带有空格的空行）
    text = re.sub(r'\n\s*\n+', '\n', text)
    
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
        r"\n(?=\s*第[一二三四五六七八九十百零]+章\s+)", 
        r"\n(?=\s*第[一二三四五六七八九十百零]+节\s+)", 
        r"\n(?=\s*第[一二三四五六七八九十百零]+条\s*)", # 这里把条后面的 \s+ 改为 \s*，防止有的条文后面没空格直接接文字
        "。\n",
        "；\n",
        "。",
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
chunks = process_legal_document('app/rag/中华人民共和国治安管理处罚法.pdf')
# 创建向量数据库
if os.path.exists('/app/rag/chroma_db') :
    shutil.rmtree('/app/rag/chroma_db')
    print('旧数据库已删除')

db = Chroma.from_documents(
        documents = chunks, 
        embedding = embedding, 
        collection_name='china_public_security_law',
        persist_directory='app/rag/chroma_db'
)
print('数据库创建成功')
def get_retrieval_result(query):
    retrievar = db.as_retriever(
        search_type = 'similarity',
        search_kwargs = {
            'k': 6,
            }
    )
    bm25_retriever = BM25Retriever.from_documents(chunks)
    ensemble_retriever = EnsembleRetriever(retrievers=[retrievar, bm25_retriever],weights=[0.6, 0.4])
    return ensemble_retriever.invoke(query)
    # return retrievar.invoke(query)




if __name__ == '__main__':
    print(get_retrieval_result('阻碍执行紧急任务的消防车'))