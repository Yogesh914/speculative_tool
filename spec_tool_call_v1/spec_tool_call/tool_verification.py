"""
Tool Call Verification Module

Provides semantic verification of tool calls using AST comparison and embedding similarity.
This allows draft model predictions to be accepted even if they don't match exactly.
"""

import asyncio
import time
from typing import Any, Dict, List, Tuple

from .embedding_similarity import get_embedding, cosine_similarity


def tools_to_string(tools: List[Dict[str, Any]]) -> str:
    """Convert tool calls to string representation for comparison.
    
    Args:
        tools: List of tool call dictionaries with 'name' and 'args'
        
    Returns:
        String representation like "tool1(arg1=val1, arg2=val2) | tool2(...)"
    """
    if not tools:
        return ""

    parts = []
    for tool in tools:
        name = tool.get('name', '')
        args = tool.get('args', {})
        args_str = ', '.join(f"{k}={v}" for k, v in sorted(args.items()))
        parts.append(f"{name}({args_str})")

    return " | ".join(parts)


def check_ast_equivalence(
    draft_tools: List[Dict[str, Any]],
    target_tools: List[Dict[str, Any]]
) -> bool:
    """Check if tool calls are exactly equivalent using AST comparison.

    Simply compares tool names and arguments directly.
    
    Args:
        draft_tools: Tool calls from draft model
        target_tools: Tool calls from target model
        
    Returns:
        True if they are exactly equivalent
    """
    # Must have same number of tool calls
    if len(draft_tools) != len(target_tools):
        return False

    # Sort both by tool name for comparison (order doesn't matter)
    draft_sorted = sorted(draft_tools, key=lambda x: (x.get('name', ''), str(x.get('args', {}))))
    target_sorted = sorted(target_tools, key=lambda x: (x.get('name', ''), str(x.get('args', {}))))

    # Compare each pair
    for draft, target in zip(draft_sorted, target_sorted):
        # Tool names must match
        if draft.get('name') != target.get('name'):
            return False

        # Arguments must match (using string comparison for simplicity)
        draft_args = draft.get('args', {})
        target_args = target.get('args', {})

        # Compare argument keys
        if set(draft_args.keys()) != set(target_args.keys()):
            return False

        # Compare argument values (as strings to handle type variations)
        for key in draft_args:
            if str(draft_args[key]) != str(target_args[key]):
                return False

    return True


async def speculation_checker(
    draft_tools: List[Dict[str, Any]],
    target_tools: List[Dict[str, Any]],
    embedding_threshold: float = 0.5,
    embedding_method: str = "gemini",
    verbose: bool = False
) -> Tuple[bool, str, float, float]:
    """Verify if draft tool calls match target using AST + cosine similarity fallback.

    This is the main verification function that combines:
    1. Exact AST comparison (fastest, most reliable)
    2. Embedding similarity fallback (flexible, catches semantic equivalence)

    Args:
        draft_tools: Tool calls from draft model
        target_tools: Tool calls from target model
        embedding_threshold: Minimum similarity score to accept (0-1, default 0.5)
        embedding_method: "gemini" or "gemma" for embeddings
        verbose: If True, print detailed verification information

    Returns:
        Tuple of (verified, method, similarity_score, verification_time)
        - verified: True if draft matches target
        - method: "ast" or "embedding" or "mismatch"
        - similarity_score: Cosine similarity score (or 1.0 for AST match)
        - verification_time: Time spent on verification
    """
    start_time = time.time()

    # Convert to strings for display
    draft_str = tools_to_string(draft_tools)
    target_str = tools_to_string(target_tools)

    if verbose:
        print(f"[Verify] Draft:  {draft_str}")
        print(f"[Verify] Target: {target_str}")

    # Step 1: Try AST comparison (exact match)
    ast_match = check_ast_equivalence(draft_tools, target_tools)

    if ast_match:
        verification_time = time.time() - start_time
        if verbose:
            print(f"[Verify] AST Match: EXACT")
            print(f"[Verify] Result: ACCEPTED (ast)")
        return True, "ast", 1.0, verification_time

    # Step 2: Fallback to embedding similarity
    if verbose:
        print(f"[Verify] AST Match: FAILED, trying embedding similarity...")

    try:
        # Get embeddings (run in executor for sync function)
        loop = asyncio.get_event_loop()
        draft_embedding = await loop.run_in_executor(
            None,
            get_embedding,
            draft_str,
            embedding_method
        )
        target_embedding = await loop.run_in_executor(
            None,
            get_embedding,
            target_str,
            embedding_method
        )

        # Compute cosine similarity
        similarity = float(cosine_similarity(draft_embedding, target_embedding))

    except Exception as e:
        if verbose:
            print(f"[Verify] Embedding error: {e}")
        similarity = 0.0

    verification_time = time.time() - start_time

    # Check if similarity meets threshold
    verified = similarity >= embedding_threshold
    method = "embedding" if verified else "mismatch"

    if verbose:
        print(f"[Verify] Similarity: {similarity:.3f}, Threshold: {embedding_threshold}")
        print(f"[Verify] Result: {'ACCEPTED' if verified else 'REJECTED'} ({method})")

    return verified, method, similarity, verification_time


# Simplified single tool call verification (for cache lookup)
async def verify_single_tool_call(
    target_tool: str,
    target_args: Dict[str, Any],
    draft_tool: str,
    draft_args: Dict[str, Any],
    embedding_threshold: float = 0.7,
    embedding_method: str = "gemini",
    verbose: bool = False
) -> Tuple[bool, str, float]:
    """
    Verify if a single draft tool call matches a single target tool call.
    
    This is a wrapper around speculation_checker for single tool calls.
    
    Args:
        target_tool: Tool name that target model wants to call
        target_args: Arguments that target model wants to use
        draft_tool: Tool name that draft already executed
        draft_args: Arguments that draft used
        embedding_threshold: Minimum similarity score to accept
        embedding_method: "gemini" or "gemma"
        verbose: Print detailed info
    
    Returns:
        Tuple of (verified, method, similarity_score)
    """
    draft_tools = [{"name": draft_tool, "args": draft_args}]
    target_tools = [{"name": target_tool, "args": target_args}]
    
    verified, method, similarity, _ = await speculation_checker(
        draft_tools,
        target_tools,
        embedding_threshold=embedding_threshold,
        embedding_method=embedding_method,
        verbose=verbose
    )
    
    return verified, method, similarity

