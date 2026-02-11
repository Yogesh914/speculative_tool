"""Speculation logic: cache management and speculative execution."""
import asyncio
from typing import Any, Dict, Optional, Tuple

from .models import RunState
from .tool_registry import TOOLS, _default_normalizer


def cache_key(tool: str, args: Dict[str, Any]) -> Tuple[str, str]:
    """Generate normalized cache key for tool + args."""
    spec = TOOLS[tool]
    norm = spec.normalizer(args) if spec.normalizer else _default_normalizer(args)
    return (tool, norm)


async def launch_if_absent(state: RunState, tool: str, args: Dict[str, Any]) -> None:
    """
    Launch speculative tool call if not already in cache.
    Only launches read-only tools.
    """
    if tool not in TOOLS:
        return

    spec = TOOLS[tool]
    if not spec.read_only:
        return

    key = cache_key(tool, args)
    if key in state.spec_cache:
        return

    async def _runner():
        try:
            res = await spec.fn(**args)
            return {"tool": tool, "args": args, "result": res}
        except Exception as e:
            return {"tool": tool, "args": args, "error": str(e)}

    task = asyncio.create_task(_runner())
    state.spec_cache[key] = task
    state.speculative_launched += 1


async def consume_cache_if_match(
    state: RunState,
    tool: str,
    args: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Check if speculative result exists and consume it.
    Returns None if cache miss, otherwise returns the result.
    Updates hit/miss metrics.
    """
    key = cache_key(tool, args)
    task = state.spec_cache.get(key)

    if task is None:
        return None

    try:
        res = await task
        state.hits += 1
        return res
    except Exception as e:
        state.misses += 1
        return {"tool": tool, "args": args, "error": str(e)}
