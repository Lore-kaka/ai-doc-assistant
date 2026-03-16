from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain_nvidia import ChatNVIDIA
from app.agent.tools.tool_tavily import get_tavily_data
from prompt.search_agnet_prompt import prompt1

# Load environment variables from .env file

load_dotenv()


llm = ChatNVIDIA(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('NVIDIA_API_KEY'),
    temperature=0.6,
)

agent = create_agent(
    model=llm,
    tools = [get_tavily_data],
    system_prompt = prompt1,
)


if __name__ == "__main__":
    query = "介绍一下中国上海。"
    response = agent.invoke({'messages': [{'role': 'user', 'content': query}]})
    print(response)