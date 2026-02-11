"""LLM adapter using proper LangGraph tool calling pattern."""
from typing import List
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .models import Msg
from .config import config


# System prompt
# SYSTEM_PROMPT = (
#     "You are an AI assistant that helps answer questions by using tools.\n"
#     "\n## Available Tools:\n"
#     "\n**Search & Web:**\n"
#     # "- search_web: Quick web search (3 results default, 10 with expand_search=True)\n"
#     # "\n**Files:**\n"
#     # "- file_read: Read files (supports CSV, XLSX, PDF, DOCX, TXT, JSON, YAML, XML, HTML)\n"
#     "\n**Computation:**\n"
#     "- calculate: Evaluate math expressions (use Python syntax like 2**10, math.sqrt(16))\n"
#     # "- code_exec: Execute Python code in a sandbox\n"
#     # "- code_generate: Generate Python code for a task\n"
#     # "\n**Vision:**\n"
#     # "- vision_analyze: Analyze images\n"
#     # "- vision_ocr: Extract text from images\n"
#     "\n## Important Rules:\n"
#     "1. Use tools when you need information (search, calculate, read files, etc.)\n"
#     "2. After getting tool results, review them and decide if you have enough information\n"
#     "3. When you have sufficient information to answer, YOU MUST provide the final answer\n"
#     "4. Format your final answer EXACTLY as: FINAL ANSWER: <your answer here>\n"
# )


SYSTEM_PROMPT = (
    # "You are an AI assistant that helps answer questions by using tools.\n"
    # "\n## Available Tools\n"
    # "Use the tools listed below whenever you need factual information. "
    # "Every search tool takes a single argument `query`.\n"
    # "\n**Search & Web:**\n"
    # "- duckduckgo_search\n"
    # "- google_serper_search\n"
    # "- searxng_search\n"
    # "- brave_search\n"
    # "- tavily_search\n"
    # "- exa_search_results\n"
    # # "- asknews_search: Access to current news coverage (ID/secret required).\n"
    # "- arxiv_search\n"
    # # "- you_search: You.com AI-powered search results.\n"
    # # "\n**Finishing:**\n"
    # # "- finish(answer): When you are done, call this to format the response exactly as `FINAL ANSWER: ...`.\n"
    # "\n## Important Rules\n"
    # "1. Use tools proactively when you need information.\n"
    # "2. After receiving tool results, reason about them before answering.\n"
    # # "3. Once confident in the answer, call `finish` with the final text.\n"
    # # "4. Do not call `finish` until you have all required information.\n"
    # "3. When you have sufficient information to answer, YOU MUST provide the final answer\n"
    # "4. Format your final answer EXACTLY as: FINAL ANSWER: <your answer here>\n"
    ""
)

# Cache for models (initialized on first use, after .env is loaded)
_actor_model = None
_spec_model = None


def get_actor_model():
    """Get the actor model with tools bound (lazy initialization).

    Model-specific parameters:
    - OpenAI models: Can add reasoning_effort, temperature, etc.
    - Google models: Can add temperature, etc.
    """
    global _actor_model
    if _actor_model is None:
        from .tools_langchain import ALL_TOOLS
        import os

        # Initialize base model
        model_kwargs = {}
        provider = config.actor_provider

        # Add model-specific parameters
        if provider == "openai":
            if os.getenv("OPENAI_REASONING_EFFORT"):
                model_kwargs["reasoning_effort"] = os.getenv("OPENAI_REASONING_EFFORT")
            if os.getenv("OPENAI_TEMPERATURE"):
                model_kwargs["temperature"] = float(os.getenv("OPENAI_TEMPERATURE"))
            if config.actor_api_key:
                model_kwargs["api_key"] = config.actor_api_key
        elif provider == "google-genai":
            if os.getenv("GOOGLE_TEMPERATURE"):
                model_kwargs["temperature"] = float(os.getenv("GOOGLE_TEMPERATURE"))
            if config.actor_api_key:
                model_kwargs["api_key"] = config.actor_api_key

        model = init_chat_model(
            config.actor_model,
            model_provider=provider,
            **model_kwargs
        )
        _actor_model = model.bind_tools(ALL_TOOLS)
    return _actor_model


def get_spec_model():
    """Get the speculator model with tools bound (lazy initialization).

    Model-specific parameters:
    - OpenAI models: Can add reasoning_effort="low" for faster speculation
    - Google models: Can add temperature, etc.
    """
    global _spec_model
    if _spec_model is None:
        from .tools_langchain import READ_ONLY_TOOLS
        import os

        # Initialize base model
        model_kwargs = {}
        provider = config.spec_provider

        # Add model-specific parameters
        if provider == "openai":
            model_kwargs["reasoning_effort"] = os.getenv("OPENAI_SPEC_REASONING_EFFORT", "low")
            if os.getenv("OPENAI_SPEC_TEMPERATURE"):
                model_kwargs["temperature"] = float(os.getenv("OPENAI_SPEC_TEMPERATURE"))
            if config.spec_api_key:
                model_kwargs["api_key"] = config.spec_api_key
        elif provider == "google-genai":
            if os.getenv("GOOGLE_SPEC_TEMPERATURE"):
                model_kwargs["temperature"] = float(os.getenv("GOOGLE_SPEC_TEMPERATURE"))
            if config.spec_api_key:
                model_kwargs["api_key"] = config.spec_api_key

        model = init_chat_model(
            config.spec_model,
            model_provider=provider,
            **model_kwargs
        )
        _spec_model = model.bind_tools(READ_ONLY_TOOLS)
    return _spec_model


def convert_msg_to_langchain(messages: List[Msg]) -> List:
    """Convert our Msg objects to LangChain messages."""
    lc_messages = []
    for m in messages:
        if m.role == "system":
            lc_messages.append(SystemMessage(content=m.content))
        elif m.role == "user":
            lc_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=m.content))
        elif m.role == "tool":
            # Convert tool results to user messages for simplicity
            tool_result = f"[TOOL: {m.name}]\n{m.content}"
            lc_messages.append(HumanMessage(content=tool_result))
    return lc_messages
