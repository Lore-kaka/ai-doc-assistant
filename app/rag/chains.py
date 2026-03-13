from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
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
以下是你们聊过的历史对话记录；
{history}
用户的问题：
{question}
"""
)


llm = ChatNVIDIA(
  model="openai/gpt-oss-120b",
  api_key=os.getenv('NVIDIA_API_KEY'), 
  temperature=0.6,
)

history={}
def get_chat_history(session_id ):
    if session_id not in history:
        history[session_id] = ChatMessageHistory()
    return history[session_id]


chain = (
    {'context':RunnableLambda(lambda x:get_retrieval_result(x['question'])),
     'question':RunnablePassthrough(),
     'history':RunnableLambda(lambda x: get_chat_history(x.get('session_id','')))
    }
    | prompt
    | llm 
    |StrOutputParser()
    )

chain_with_history = RunnableWithMessageHistory(
    chain, 
    get_chat_history,
    input_messages_key='question',
    history_messages_key='history',
)
def main(query:str,session_id:str):
    result = chain_with_history.invoke(
        {'question':query,'session_id':session_id},
        config={'configurable':{'session_id':session_id}}
    )
    return result




if __name__ == '__main__':

    #请解释一下治安管理处罚法第50条
    
    # print(main(query='请解释一下治安管理处罚法第50条',session_id='123'))
    print(main(query='阻碍执行紧急任务的消防车、救护车。会如何处罚。',session_id='123'))
