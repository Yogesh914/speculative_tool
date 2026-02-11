"""LLM adapter using proper LangGraph tool calling pattern."""
import os
import re
import base64
from typing import List
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .models import Msg
from .config import config


# System prompt for OpenAI models
SYSTEM_PROMPT_OPENAI = (
    "You are an AI assistant that helps answer questions by using tools.\n"
    "\n## Multimodal Capabilities:\n"
    "- You can receive and see images directly in the conversation when image files are attached\n"
    "- When an image is provided inline, analyze it directly without using vision tools\n"
    "- Use vision_analyze or vision_ocr tools only when you need to call them explicitly\n"
    "\n## Available Tools:\n"
    "\n**Search & Web:**\n"
    "- web_search: Fast search returning titles, URLs, and snippets (3 results)\n"
    "- enhanced_search: Deep search extracting full page content (3 results, slower but comprehensive)\n"
    "\n**Files:**\n"
    "- file_read: Read files (supports CSV, XLSX, PDF, DOCX, TXT, JSON, YAML, XML, HTML)\n"
    "\n**Computation:**\n"
    "- calculate: Evaluate SINGLE math expressions (e.g., (356400/42.195)/(2+1/60+9/3600))\n"
    "- code_exec: Execute multi-step Python code with variables in a sandbox\n"
    "- code_generate: Generate Python code for a task\n"
    "\n**Vision (for advanced image analysis only when necessary):**\n"
    "- vision_analyze: Analyze images with specific questions\n"
    "- vision_ocr: Extract text from images\n"
    "\n## CRITICAL - Always Explain Your Actions:\n"
    "Before EVERY tool call, you MUST output a brief 1-sentence explanation of why you're calling that tool.\n"
    "Example: 'I need to find the Moon's minimum perigee value from Wikipedia.' then call web_search.\n"
    "This explanation must appear in your response content, not just internally.\n"
    "\n## Important:\n"
    "- Use web_search for quick lookups, enhanced_search when you need detailed content\n"
    "- When you have the final answer, respond directly without calling tools\n"
    "\n## Final Answer Format:\n"
    "Report your thoughts, and finish your answer with: FINAL ANSWER: [YOUR FINAL ANSWER]\n"
    "\n**Formatting Rules:**\n"
    "- YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list\n"
    "- For numbers: Don't use commas or units (like $ or %) unless specified\n"
    "- For strings: Don't use articles or abbreviations, write digits in plain text\n"
    "- Pay attention to the question's units (e.g., 'how many thousand hours' means answer in thousands)\n"
    "- For lists: Apply above rules per element\n"
)

# System prompt for open-source models (vLLM) - more explicit about tool usage
SYSTEM_PROMPT_VLLM = (
    "You are an AI assistant that solves questions by using tools. You MUST use tools to gather information.\n"
    "\n## 🔧 CRITICAL: USE TOOLS FREQUENTLY!\n"
    "- DO NOT try to answer from memory or reasoning alone\n"
    "- ALWAYS use web_search or enhanced_search to look up facts, data, and information\n"
    "- Use multiple tool calls if needed to gather all required information\n"
    "- Use code_exec for calculations and data processing\n"
    "\n## Available Tools:\n"
    "\n**Search & Web (USE THIS OFTEN!):**\n"
    "- web_search: Fast search returning titles, URLs, and snippets (3 results)\n"
    "- enhanced_search: Deep search extracting full page content (3 results) - USE THIS for detailed info\n"
    "\n**Files:**\n"
    "- file_read: Read files (CSV, XLSX, PDF, DOCX, TXT, JSON, YAML, XML, HTML)\n"
    "\n**Computation:**\n"
    "- calculate: Evaluate math expressions like (356400/42.195)/(2+1/60+9/3600)\n"
    "- code_exec: Execute Python code for complex calculations\n"
    "- code_generate: Generate Python code\n"
    "\n**Vision:**\n"
    "- vision_analyze: Analyze images\n"
    "- vision_ocr: Extract text from images\n"
    "\n## CRITICAL - Always Explain Your Actions:\n"
    "Before EVERY tool call, you MUST output a brief 1-sentence explanation of why you're calling that tool.\n"
    "Example: 'I need to find the Moon's minimum perigee value from Wikipedia.' then call web_search.\n"
    "This explanation must appear in your response content, not just internally.\n"
    "\n## Step-by-Step Process:\n"
    "1. Read the question carefully\n"
    "2. Identify what information you need\n"
    "3. USE TOOLS to gather that information (search, calculate, read files, etc.)\n"
    "4. If you need more information, USE MORE TOOLS\n"
    "5. Only after gathering ALL needed information, provide your final answer\n"
    "\n## Final Answer Format:\n"
    "When ready to answer, respond with: FINAL ANSWER: [YOUR FINAL ANSWER]\n"
    "\n**Formatting Rules:**\n"
    "- YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list\n"
    "- For numbers: Don't use commas or units (like $ or %) unless specified\n"
    "- For strings: Don't use articles or abbreviations, write digits in plain text\n"
    "- Pay attention to the question's units (e.g., 'how many thousand hours' means answer in thousands)\n"
    "- For lists: Apply above rules per element\n"
)

# Cache for models (initialized on first use, after .env is loaded)
_actor_model = None
_spec_model = None


def get_system_prompt() -> str:
    """Get the appropriate system prompt based on model provider."""
    if config.model_provider == "vllm":
        return SYSTEM_PROMPT_VLLM
    return SYSTEM_PROMPT_OPENAI


# For backward compatibility
SYSTEM_PROMPT = SYSTEM_PROMPT_OPENAI

# Simplified system prompt for speculator model (fast prediction, low reasoning)
SYSTEM_PROMPT_SPEC = (
    "You are a fast prediction assistant. Your job is to quickly predict the next tool call.\n"
    "\n## Your Role:\n"
    "Predict the most likely next tool call based on the conversation. Be fast and direct.\n"
    "\n## Available Tools:\n"
    "- web_search: Fast search (3 results)\n"
    "- enhanced_search: Deep search with full content (3 results)\n"
    "- file_read: Read files (CSV, XLSX, PDF, DOCX, TXT, JSON, etc.)\n"
    "- calculate: Evaluate math expressions\n"
    "- code_exec: Execute Python code\n"
    "- code_generate: Generate Python code\n"
    "- vision_analyze: Analyze images\n"
    "- vision_ocr: Extract text from images\n"
    "\n## Instructions:\n"
    "- Predict the next tool call quickly\n"
    "- Use the same tool names and argument format as the actor\n"
    "- Don't overthink - make a reasonable prediction based on the conversation flow\n"
)


def get_spec_system_prompt() -> str:
    """Get the simplified system prompt for speculator model."""
    return SYSTEM_PROMPT_SPEC


def _create_model(model_name: str, is_spec: bool = False):
    """Create a chat model based on configured provider.
    
    Args:
        model_name: Name/path of the model
        is_spec: If True, use spec model endpoint and API key; if False, use actor endpoint and API key
    """
    if config.model_provider == "vllm":
        # Use vLLM via OpenAI-compatible API
        # Choose endpoint based on whether this is spec or actor model
        base_url = config.vllm_spec_url if is_spec else config.vllm_actor_url
        
        return ChatOpenAI(
            model=model_name,
            openai_api_key=config.vllm_api_key,
            openai_api_base=base_url,
            temperature=0,
            max_tokens=config.llm_max_tokens,
        )
    else:
        # Use OpenAI with separate API keys for actor and spec to avoid rate limiting
        api_key = config.spec_api_key if is_spec else config.actor_api_key
        
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=0,
            max_tokens=4096,
        )


def get_actor_model():
    """Get the actor model with tools bound (lazy initialization)."""
    global _actor_model
    if _actor_model is None:
        from .tools_langchain import ALL_TOOLS
        
        model = _create_model(config.actor_model, is_spec=False)
        _actor_model = model.bind_tools(ALL_TOOLS)
    return _actor_model


def get_spec_model():
    """Get the speculator model with tools bound (lazy initialization)."""
    global _spec_model
    if _spec_model is None:
        from .tools_langchain import READ_ONLY_TOOLS
        
        model = _create_model(config.spec_model, is_spec=True)
        _spec_model = model.bind_tools(READ_ONLY_TOOLS)
    return _spec_model


def _encode_image_to_base64(image_path: str) -> dict:
    """Encode local image to base64 for multimodal API."""
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Detect image type
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/jpeg')

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{image_data}"
        }
    }


def _extract_image_paths(content: str) -> list:
    """Extract image file paths from message content.

    Looks for patterns like:
    - [Note: There is an attached file at path: /path/to/image.png]
    - Standalone file paths ending in image extensions
    """
    image_paths = []

    # Pattern 1: GAIA-style attachment notation
    pattern = r'\[Note: There is an attached file at path: ([^\]]+)\]'
    matches = re.findall(pattern, content)
    for match in matches:
        path = match.strip()
        if _is_image_file(path):
            image_paths.append(path)

    # Pattern 2: Standalone paths (look for common image extensions)
    # Only if no GAIA-style match found
    if not image_paths:
        words = content.split()
        for word in words:
            word = word.strip('.,;:()[]{}"\' \n\r\t')
            if _is_image_file(word) and os.path.exists(word):
                image_paths.append(word)

    return image_paths


def _is_image_file(path: str) -> bool:
    """Check if a path is an image file."""
    if not path:
        return False
    ext = Path(path).suffix.lower()
    return ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff']


def convert_msg_to_langchain(messages: List[Msg]) -> List:
    """Convert our Msg objects to LangChain messages.

    Handles multimodal messages by detecting image paths in user messages
    and constructing appropriate multimodal content.
    """
    lc_messages = []
    for m in messages:
        if m.role == "system":
            lc_messages.append(SystemMessage(content=m.content))
        elif m.role == "user":
            # Check if message contains image paths
            image_paths = _extract_image_paths(m.content)

            if image_paths:
                # Construct multimodal message with text and images
                content_parts = [{"type": "text", "text": m.content}]

                for img_path in image_paths:
                    if os.path.exists(img_path):
                        try:
                            image_content = _encode_image_to_base64(img_path)
                            content_parts.append(image_content)
                        except Exception as e:
                            print(f"⚠️  Failed to encode image {img_path}: {e}")

                lc_messages.append(HumanMessage(content=content_parts))
            else:
                # Regular text-only message
                lc_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=m.content))
        elif m.role == "tool":
            # Convert tool results to user messages for simplicity
            tool_result = f"[TOOL: {m.name}]\n{m.content}"
            lc_messages.append(HumanMessage(content=tool_result))
    return lc_messages
