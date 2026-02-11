"""Tool definitions using LangChain @tool decorator."""
from langchain.tools import tool

# Import existing implementations
# Using Gensee AI for web search (Wikipedia-focused) - kept for backward compatibility
from .tools.search_tool import search_gensee_web
from .tools.file_tool import read_file as read_file_enhanced
from .tools.code_exec_tool import execute_python_code, execute_calculation, generate_python_code
from .tools.vision_tool import analyze_image, extract_text_from_image

# Import multi-provider search tools (9 total: Exa, Tavily, Arxiv, AskNews, Brave, DuckDuckGo, Serper, SearXNG, You.com)
from .tools.multi_search_tool import get_all_search_tools


@tool
def search_web(query: str, expand_search: bool = False) -> str:
    """Search Wikipedia for reliable information.

    Uses Gensee AI to search Wikipedia articles. Returns 3 results by default,
    or 10 results if you need more comprehensive coverage.

    NOTE: This is kept for backward compatibility. For more search options,
    the system will also load additional search providers (DuckDuckGo, Tavily, etc.)

    Args:
        query: The search query (will automatically search Wikipedia)
        expand_search: If True, return 10 results instead of 3 (use when you need more options)
    """
    max_results = 10 if expand_search else 3
    return search_gensee_web(query, max_results)


@tool
def file_read(path: str) -> str:
    """Read a file and return its content.
    
    Args:
        path: Path to the file
    """
    result = read_file_enhanced(path)
    if isinstance(result, dict):
        if result.get("status") == "success":
            return result.get("content", str(result))
        return f"Error: {result.get('error')}"
    return str(result)


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Use Python syntax. Examples:
    - Basic: "2 + 3", "10 * 5"
    - Powers: "2**10"
    - Functions: "import math; math.sqrt(16)"
    
    Args:
        expression: Python expression to evaluate
    """
    result = execute_calculation(expression)
    if isinstance(result, dict):
        if result.get("status") == "success":
            return f"Result: {result.get('result')}"
        return f"Error: {result.get('error')}\nExpression was: {expression}"
    return str(result)


@tool
def code_generate(task_description: str, context: str = None) -> str:
    """Generate Python code using GPT-5.
    
    Args:
        task_description: What the code should do
        context: Optional additional context
    """
    result = generate_python_code(task_description, context)
    if isinstance(result, dict) and result.get("status") == "success":
        return result.get("code", "")
    return str(result)


@tool
def code_exec(code: str, timeout: int = 30) -> str:
    """Execute Python code in a sandbox.
    
    Args:
        code: Python code to execute
        timeout: Timeout in seconds
    """
    result = execute_python_code(code, timeout)
    if isinstance(result, dict):
        if result.get("status") == "success":
            return result.get("output", "")
        return f"Error: {result.get('error')}"
    return str(result)


@tool
def vision_analyze(image_path: str, question: str = None) -> str:
    """Analyze an image using vision model.
    
    Args:
        image_path: Path to the image
        question: Optional question about the image
    """
    result = analyze_image(image_path, question)
    if isinstance(result, dict):
        if result.get("status") == "success":
            return result.get("analysis", "")
        return f"Error: {result.get('error')}"
    return str(result)


@tool
def vision_ocr(image_path: str) -> str:
    """Extract text from an image using OCR.
    
    Args:
        image_path: Path to the image
    """
    result = extract_text_from_image(image_path)
    if isinstance(result, dict):
        if result.get("status") == "success":
            return result.get("text", "")
        return f"Error: {result.get('error')}"
    return str(result)


# Final answer formatting tool
@tool
def finish(answer: str) -> str:
    """Format the final answer exactly as required."""
    answer = answer.strip()
    return f"FINAL ANSWER: {answer}"


# Base tools (non-search)
BASE_TOOLS = [
    # search_web,
    # file_read,
    # calculate,
    # finish,
    # code_generate,
    # code_exec,
    # vision_analyze,
    # vision_ocr,
]

# Load multi-provider search tools dynamically
# This loads all available search tools based on installed packages and API keys
# Includes: Exa, Tavily, Arxiv, AskNews, Brave, DuckDuckGo, Google Serper, SearXNG, You.com
MULTI_SEARCH_TOOLS = get_all_search_tools(verbose=False)

# All tools (for Target/Actor model) = base tools + multi-search tools
ALL_TOOLS = BASE_TOOLS + MULTI_SEARCH_TOOLS

# Tools by name (for runtime lookup)
TOOLS_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}

# Read-only tools (safe for speculation - used by Draft model)
# All search tools are read-only (they don't modify state)
# Calculate, code_generate, and vision tools are also safe
READ_ONLY_TOOLS = [
    # search_web,
    # file_read,
    # calculate,
    # finish,
    # code_generate,
    # vision_analyze,
    # vision_ocr,
] + MULTI_SEARCH_TOOLS


def print_available_tools():
    """Print summary of available tools (useful for debugging)."""
    print(f"\n📊 Tool Registry Summary:")
    print(f"  Total tools: {len(ALL_TOOLS)}")
    print(f"  Base tools: {len(BASE_TOOLS)}")
    print(f"  Search tools: {len(MULTI_SEARCH_TOOLS)}")
    print(f"  Read-only tools (for speculation): {len(READ_ONLY_TOOLS)}")

    print(f"\n📝 Available tool names:")
    for name in sorted(TOOLS_BY_NAME.keys()):
        print(f"  - {name}")


# Print tool summary on import (can be disabled by setting env var)
import os
if os.environ.get("PRINT_TOOL_SUMMARY", "0") == "1":
    print_available_tools()
