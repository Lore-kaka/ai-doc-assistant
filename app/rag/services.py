from langchain_community.document_loaders import TextLoader
from langchain_text_splitters  import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
# from langchain_community.embeddings import DashScopeEmbeddings
import os 
from dotenv import load_dotenv
load_dotenv()


# 加载文档

loader = TextLoader('/Users/yangyifan/Develop/fastapi-langchain-portfolio/中华人民共和国治安管理处罚法_20250627.txt')

documents = loader.load()


# 分割文档

text_splitter = RecursiveCharacterTextSplitter(
    separators = ['\n\n', '\n', ' ', '。', '！', '？'],
    )
texts = text_splitter.split_documents(documents)

# 创建向量数据库

embedding = OllamaEmbeddings(
    model = 'qwen3-embedding:0.6b'
)
# embedding = DashScopeEmbeddings(
#     model="text-embedding-v4"
# )

if os.path.exists('./chroma_db') or os.path.isfile('./chroma_db'):
    db =Chroma(
        persist_directory='app/rag/chroma_db',
    )
    print('数据库已存在,加载成功')
else:
    db = Chroma.from_documents(
        documents = texts, 
        embedding = embedding, 
        collection_name='china_public_security_law',
        persist_directory='app/rag/chroma_db'
    )
    print('数据库创建成功')
def get_retrieval_result(query):
    retrievar = db.as_retriever()
    return retrievar.invoke(query)



if __name__ == '__main__':
    print(get_retrieval_result('请解释一下治安管理处罚法第5条'))