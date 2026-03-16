from langchain.tools import tool
from langchain_tavily import TavilySearch

@tool
def get_tavily_data(query: str) :
    """
    Get data from Tavily API
    """
    import dotenv
    dotenv.load_dotenv()
    return TavilySearch(max_results=10)


