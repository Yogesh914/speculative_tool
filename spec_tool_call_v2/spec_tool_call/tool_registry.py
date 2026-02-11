"""Tool implementations and registry for read-only operations."""
import json
from typing import Any, Dict

from .models import ToolSpec
from .tools.search_tool import web_search, enhanced_web_search
from .tools.file_tool import read_file as read_file_enhanced
from .tools.code_exec_tool import execute_python_code, execute_calculation, generate_python_code
from .tools.vision_tool import analyze_image, extract_text_from_image


# -----------------------------
# Normalizers and equality checks
# -----------------------------

def _default_normalizer(args: Dict[str, Any]) -> str:
    """Default argument normalizer: lowercase strings, sort keys."""
    def norm(v):
        if isinstance(v, str):
            v = v.strip().lower()
        return v

    normed = {k: norm(v) for k, v in sorted(args.items(), key=lambda x: x[0])}
    return json.dumps(normed, ensure_ascii=False)


def _default_equality(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Default equality check using normalization."""
    return _default_normalizer(a) == _default_normalizer(b)


# -----------------------------
# Tool implementations
# -----------------------------

async def tool_file_read(path: str) -> Dict[str, Any]:
    """
    Read file content. Supports diverse file types.
    Read-only operation suitable for speculation.
    """
    # Use the enhanced file reading tool
    return read_file_enhanced(path)


# Async wrappers for sync functions
async def tool_search_web(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web using Serper API."""
    result_text = web_search(query, max_results)
    return {"result": result_text}


async def tool_enhanced_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Search the web and extract content using Serper API."""
    result_text = enhanced_web_search(query, max_results)
    return {"result": result_text}


async def tool_code_exec(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute Python code in sandbox."""
    return execute_python_code(code, timeout)


async def tool_calculate(expression: str) -> Dict[str, Any]:
    """Execute a mathematical calculation."""
    return execute_calculation(expression)


async def tool_vision_analyze(image_path: str, question: str = None) -> Dict[str, Any]:
    """Analyze an image using vision model."""
    return analyze_image(image_path, question)


async def tool_vision_ocr(image_path: str) -> Dict[str, Any]:
    """Extract text from an image."""
    return extract_text_from_image(image_path)


async def tool_code_generate(task_description: str, context: str = None) -> Dict[str, Any]:
    """Generate Python code using GPT-5."""
    return generate_python_code(task_description, context)


# -----------------------------
# Tool registry
# -----------------------------

TOOLS: Dict[str, ToolSpec] = {
    # Search (read-only, good for speculation)
    "web_search": ToolSpec(
        name="web_search",
        read_only=True,
        normalizer=lambda args: _default_normalizer({"query": args.get("query", ""), "max_results": args.get("max_results", 5)}),
        equality=lambda a, b: _default_normalizer({"query": a.get("query", "")}) == _default_normalizer({"query": b.get("query", "")}),
        fn=tool_search_web,
    ),
    
    "enhanced_search": ToolSpec(
        name="enhanced_search",
        read_only=True,
        normalizer=lambda args: _default_normalizer({"query": args.get("query", ""), "max_results": args.get("max_results", 3)}),
        equality=lambda a, b: _default_normalizer({"query": a.get("query", "")}) == _default_normalizer({"query": b.get("query", "")}),
        fn=tool_enhanced_search,
    ),
    
    # File reading (read-only, good for speculation)
    "file_read": ToolSpec(
        name="file_read",
        read_only=True,
        normalizer=lambda args: _default_normalizer({"path": args.get("path", "")}),
        equality=lambda a, b: _default_normalizer({"path": a.get("path", "")}) == _default_normalizer({"path": b.get("path", "")}),
        fn=tool_file_read,
    ),
    
    # Vision/Multimodal (read-only, good for speculation)
    "vision_analyze": ToolSpec(
        name="vision_analyze",
        read_only=True,
        normalizer=lambda args: _default_normalizer({"image_path": args.get("image_path", ""), "question": args.get("question", "")}),
        equality=lambda a, b: _default_normalizer({"image_path": a.get("image_path", "")}) == _default_normalizer({"image_path": b.get("image_path", "")}),
        fn=tool_vision_analyze,
    ),
    
    "vision_ocr": ToolSpec(
        name="vision_ocr",
        read_only=True,
        normalizer=lambda args: _default_normalizer({"image_path": args.get("image_path", "")}),
        equality=lambda a, b: _default_normalizer({"image_path": a.get("image_path", "")}) == _default_normalizer({"image_path": b.get("image_path", "")}),
        fn=tool_vision_ocr,
    ),
    
    # Code execution (NOT read-only, should not be speculated)
    "code_exec": ToolSpec(
        name="code_exec",
        read_only=False,  # Execution has side effects
        normalizer=lambda args: _default_normalizer({"code": args.get("code", "")}),
        equality=lambda a, b: _default_normalizer({"code": a.get("code", "")}) == _default_normalizer({"code": b.get("code", "")}),
        fn=tool_code_exec,
    ),
    
    # Calculation (read-only, deterministic, good for speculation)
    "calculate": ToolSpec(
        name="calculate",
        read_only=True,
        normalizer=lambda args: _default_normalizer({"expression": args.get("expression", "")}),
        equality=lambda a, b: _default_normalizer({"expression": a.get("expression", "")}) == _default_normalizer({"expression": b.get("expression", "")}),
        fn=tool_calculate,
    ),
    
    # Code generation (read-only, generates code but doesn't execute)
    "code_generate": ToolSpec(
        name="code_generate",
        read_only=True,
        normalizer=lambda args: _default_normalizer({"task_description": args.get("task_description", "")}),
        equality=lambda a, b: _default_normalizer({"task_description": a.get("task_description", "")}) == _default_normalizer({"task_description": b.get("task_description", "")}),
        fn=tool_code_generate,
    ),
}
