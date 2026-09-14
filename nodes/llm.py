import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

MODEL_ID = os.getenv("AGENT_MODEL_ID", "claude-haiku-4-5-20251001")
TEMPERATURE = 0

llm = ChatAnthropic(
    model=MODEL_ID,
    temperature=TEMPERATURE,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def invoke_structured(chain, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call include_raw structured output. Retry once if parsed is None.

    include_raw=True returns parsed=None on schema failure instead of raising.
    """
    response = chain.invoke(payload)
    if isinstance(response, dict) and response.get("parsed") is not None:
        return response
    retry = chain.invoke(payload)
    return retry if isinstance(retry, dict) else {"parsed": retry, "raw": retry}


def structured_usage(response: Any) -> tuple[int, int]:
    raw = response.get("raw") if isinstance(response, dict) else response
    usage = getattr(raw, "usage_metadata", None) or {}
    try:
        in_tokens = int(usage.get("input_tokens") or 0)
    except (TypeError, ValueError):
        in_tokens = 0
    try:
        out_tokens = int(usage.get("output_tokens") or 0)
    except (TypeError, ValueError):
        out_tokens = 0
    return in_tokens, out_tokens
