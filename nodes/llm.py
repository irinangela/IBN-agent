import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

MODEL_ID = "claude-haiku-4-5-20251001"
TEMPERATURE = 0

llm = ChatAnthropic(
    model=MODEL_ID,
    temperature=TEMPERATURE,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
