"""Tool implementations and registry for read-only operations."""
import os
import re
import json
import hashlib
from typing import Any, Dict

import httpx

from .models import ToolSpec
from .tools.search_tool import search_serper_web, search_serper_with_content
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

async def tool_web_get(url: str, timeout: float = 15.0) -> Dict[str, Any]:
    """
    Fetch a web page and return status, title, digest, and preview.
    Read-only operation suitable for speculation.
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        r = await client.get(url, headers={"User-Agent": "GAIA-Agent/Spec"})

        # Keep a short digest to detect equivalence without storing full page
        digest = hashlib.sha256((r.url.__str__() + "\n" + r.text[:2000]).encode()).hexdigest()

        title_match = re.search(r"<title>(.*?)</title>", r.text, re.I | re.S)
        title = title_match.group(1).strip() if title_match else ""

        return {
            "status": r.status_code,
            "final_url": str(r.url),
            "title": title,
            "digest": digest,
            "preview": r.text[:1000]
        }


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
    result_text = search_serper_web(query, max_results)
    return {"result": result_text}


async def tool_search_with_content(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Search the web and extract content using Serper API."""
    result_text = search_serper_with_content(query, max_results)
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
    # Web browsing
    "web_get": ToolSpec(
        name="web_get",
        read_only=True,
        normalizer=lambda args: _default_normalizer({"url": args.get("url", "")}),
        equality=lambda a, b: _default_normalizer({"url": a.get("url", "")}) == _default_normalizer({"url": b.get("url", "")}),
        fn=tool_web_get,
    ),
    
    # Search (read-only, good for speculation)
    "search_web": ToolSpec(
        name="search_web",
        read_only=True,
        normalizer=lambda args: _default_normalizer({"query": args.get("query", ""), "max_results": args.get("max_results", 5)}),
        equality=lambda a, b: _default_normalizer({"query": a.get("query", "")}) == _default_normalizer({"query": b.get("query", "")}),
        fn=tool_search_web,
    ),
    
    "search_with_content": ToolSpec(
        name="search_with_content",
        read_only=True,
        normalizer=lambda args: _default_normalizer({"query": args.get("query", ""), "max_results": args.get("max_results", 3)}),
        equality=lambda a, b: _default_normalizer({"query": a.get("query", "")}) == _default_normalizer({"query": b.get("query", "")}),
        fn=tool_search_with_content,
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
