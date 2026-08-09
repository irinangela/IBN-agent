import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Annotated
from langchain_core.tools import tool
from tavily import TavilyClient

from tools.simulator_tools import run_spp_simulator 

@tool
def web_search(query: str) -> dict:
    """
    Performs a web search using Tavily API to find industry benchmarks 
    or external telecommunications context.
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return client.search(query)

ALL_TOOLS = [web_search, run_spp_simulator]