from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from services import get_retrieval_result
import os
from dotenv import load_dotenv
load_dotenv()

prompt = ChatPromptTemplate.from_template(
    """
你是一个专业的法律助手，专门回答关于中国治安管理处罚法的问题。
请基于以下提供的法律条文内容回答用户的问题。

回答要求：
1. 严格基于提供的法律条文内容回答，不要编造信息
2. 如果提供的材料中没有相关内容，请明确说明
3. 回答时要准确引用相关条文
4. 使用清晰、专业的法律语言
5. 保持客观中立的立场

以下是相关的法律条文内容：
{context}
用户的问题：
{question}
"""
)


llm = ChatNVIDIA(
  model="openai/gpt-oss-120b",
  api_key=os.getenv('NVIDIA_API_KEY'), 
  temperature=0.6,
  max_tokens=4096,
)
chain = (
    {'context':RunnableLambda(get_retrieval_result),'question':RunnablePassthrough()}
    | prompt
    | llm 
    |StrOutputParser()
    )

if __name__ == '__main__':

    #请解释一下治安管理处罚法第50条
    # question = input('请输入你的问题：')
    print(chain.invoke('违反治安管理法，在什么情形下从重处罚'))
    print('---------------')
