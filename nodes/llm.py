import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from tools.tools import ALL_TOOLS

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)
