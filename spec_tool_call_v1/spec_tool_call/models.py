"""Data models for speculative tool calling framework."""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel


class Msg(BaseModel):
    """Message in conversation history."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


@dataclass
class RunState:
    """State maintained during execution."""
    messages: List[Msg] = field(default_factory=list)
    step: int = 0
    done: bool = False
    answer: Optional[str] = None
    
    # ReAct pattern - pending tool calls
    pending_tool_calls: List = field(default_factory=list)

    # Speculation cache: (tool, normalized_args) -> asyncio.Task
    spec_cache: Dict[Tuple[str, str], asyncio.Task] = field(default_factory=dict)

    # Metrics
    hits: int = 0
    misses: int = 0
    speculative_launched: int = 0
    t0: float = field(default_factory=time.time)
    
    # Timing for last operations
    last_llm_time: float = 0.0
    last_tool_time: float = 0.0


class ToolSpec(BaseModel):
    """Specification for a tool."""
    name: str
    read_only: bool = True
    normalizer: Optional[Callable[[Dict[str, Any]], str]] = None
    equality: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None
    fn: Optional[Callable[..., Any]] = None

    class Config:
        arbitrary_types_allowed = True
