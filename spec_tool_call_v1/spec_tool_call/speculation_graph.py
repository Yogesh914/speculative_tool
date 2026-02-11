"""Speculation graph with parallel draft and target model execution."""
import asyncio
import time
from typing import Any, Dict, Literal, Optional, Tuple

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .speculation_state import SpeculationState, Msg
from .llm_adapter import get_actor_model, get_spec_model, convert_msg_to_langchain, SYSTEM_PROMPT
from .tools_langchain import TOOLS_BY_NAME
from .config import config


def _log(message: str, *, force: bool = False) -> None:
    """Conditional logger to control verbosity."""
    if config.verbose_logging or force:
        print(message)


# -----------------------------
# Helper functions
# -----------------------------

def _extract_content_str(content: Any) -> str:
    """Extract string content from response (handles both str and list formats)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and 'text' in item:
                parts.append(item['text'])
        return ' '.join(parts) if parts else ""
    return str(content) if content else ""


async def tool_call_verification(
    target_tool: str,
    target_args: Dict[str, Any],
    draft_tool: str,
    draft_args: Dict[str, Any],
    embedding_threshold: float = 0.5,
    embedding_method: str = "gemini",
    verbose: bool = False
) -> Tuple[bool, str, float]:
    """
    Verify if draft tool call can be accepted for target tool call.
    
    Uses AST comparison first, then falls back to embedding similarity.
    
    Args:
        target_tool: Tool name that target model wants to call
        target_args: Arguments that target model wants to use
        draft_tool: Tool name that draft already executed
        draft_args: Arguments that draft used
        embedding_threshold: Minimum similarity for embedding fallback (default 0.5)
        embedding_method: "gemini" or "gemma" for embeddings
        verbose: Print detailed verification info
    
    Returns:
        Tuple of (verified, method, similarity_score)
        - verified: True if draft can be reused
        - method: "ast", "embedding", or "mismatch"
        - similarity_score: 1.0 for AST, cosine similarity for embedding
    """
    from .tool_verification import verify_single_tool_call
    
    return await verify_single_tool_call(
        target_tool=target_tool,
        target_args=target_args,
        draft_tool=draft_tool,
        draft_args=draft_args,
        embedding_threshold=embedding_threshold,
        embedding_method=embedding_method,
        verbose=verbose
    )


async def find_verified_match_in_cache(
    draft_cache: Dict,
    target_tool: str,
    target_args: Dict,
    embedding_threshold: float = 0.5,
    verbose: bool = False
) -> Optional[Tuple[Dict, str, float]]:
    """
    Search cache for a draft result that passes verification.
    
    Args:
        draft_cache: The draft execution cache (Dict[Tuple[str, str], Dict])
        target_tool: Tool that target wants to call
        target_args: Arguments for target's tool call
        embedding_threshold: Similarity threshold for embedding verification
        verbose: Print verification details
    
    Returns:
        Tuple of (cached_result, method, similarity) if verification passes, None otherwise
    """
    # Iterate through all cached entries and let verification function handle all checks
    for (cached_tool_name, cached_args_str), cached_result in draft_cache.items():
        
        # Extract original args from cached result
        draft_args = cached_result.get("args", {})
        
        # Call verification function (handles tool name check, AST match, and embedding similarity)
        verified, method, similarity = await tool_call_verification(
            target_tool=target_tool,
            target_args=target_args,
            draft_tool=cached_tool_name,
            draft_args=draft_args,
            embedding_threshold=embedding_threshold,
            verbose=verbose
        )
        
        if verified:
            # Verification passed!
            return cached_result, method, similarity
    
    # No draft result passed verification
    return None


# -----------------------------
# Graph nodes
# -----------------------------

async def node_plan_parallel(state: SpeculationState) -> SpeculationState:
    """
    Parallel planning node: Run draft and target models simultaneously.
    
    This is the KEY innovation - we use asyncio to truly run both models at the same time!
    """
    if state.done:
        return state
    
    _log(f"\n🚀 [Step {state.step + 1}] Starting PARALLEL draft + target planning")
    
    # Increment step
    state.step += 1
    
    # Reset flags
    state.reset_flags()
    
    # Get models
    draft_model = get_spec_model()
    target_model = get_actor_model()
    
    # Convert messages (shared by both)
    lc_messages = [SystemMessage(content=SYSTEM_PROMPT)] + convert_msg_to_langchain(state.messages)
    
    # Define async tasks for both models
    async def draft_predict_and_execute():
        """Draft model predicts AND immediately executes tool calls (in background)."""
        _log(f"  📝 [Draft] Starting prediction...")
        start_time = time.time()
        try:
            response = await draft_model.ainvoke(lc_messages)
            elapsed = time.time() - start_time
            state.draft_plan_time = elapsed
            
            if response.tool_calls:
                _log(f"  📝 [Draft] Predicted {len(response.tool_calls)} tool(s) in {elapsed:.2f}s")
                for tc in response.tool_calls:
                    _log(f"     → {tc['name']}")
                
                # 🚀 KEY INNOVATION: Immediately start executing tools (don't wait!)
                _log(f"  ⚙️  [Draft Exec] Starting IMMEDIATE execution (not waiting for target)...")
                exec_start = time.time()
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    cache_key = state.get_cache_key(tool_name, tool_args)

                    # Track draft tool names for statistics
                    state.draft_tool_names.append(tool_name)

                    if tool_name in TOOLS_BY_NAME:
                        tool = TOOLS_BY_NAME[tool_name]
                        try:
                            result = await tool.ainvoke(tool_args) if hasattr(tool, 'ainvoke') else tool.invoke(tool_args)
                            state.draft_cache[cache_key] = {
                                "tool": tool_name,
                                "args": tool_args,
                                "result": str(result),
                                "success": True
                            }
                            state.draft_tools_launched += 1
                            _log(f"     ✓ Executed: {tool_name}")
                        except Exception as e:
                            state.draft_cache[cache_key] = {
                                "tool": tool_name,
                                "args": tool_args,
                                "result": f"Error: {str(e)}",
                                "success": False
                            }
                
                state.draft_exec_time = time.time() - exec_start
                _log(f"  ⚙️  [Draft Exec] Completed {len(response.tool_calls)} tool(s) in {state.draft_exec_time:.2f}s")
                
                return response.tool_calls
            else:
                _log(f"  📝 [Draft] No tools predicted in {elapsed:.2f}s")
                return []
        except Exception as e:
            _log(f"  ❌ [Draft] Error: {e}", force=True)
            state.draft_plan_time = time.time() - start_time
            return []
    
    async def target_decide():
        """Target model decides actual tool calls (runs in parallel with draft execution)."""
        _log(f"  🎯 [Target] Starting decision...")
        start_time = time.time()
        try:
            response = await target_model.ainvoke(lc_messages)
            elapsed = time.time() - start_time
            state.target_plan_time = elapsed
            
            # Check if final answer
            if not response.tool_calls:
                content = _extract_content_str(response.content)
                
                if "FINAL ANSWER:" in content.upper():
                    answer_part = content.split("FINAL ANSWER:")[-1] if "FINAL ANSWER:" in content else content
                    answer = answer_part.strip()
                    _log(f"  🎯 [Target] Final answer in {elapsed:.2f}s: {answer[:50]}...")
                    return ("final", content, answer)
                else:
                    _log(f"  🎯 [Target] Thinking (no tools) in {elapsed:.2f}s")
                    return ("thinking", content, None)
            else:
                ai_msg_content = _extract_content_str(response.content) or "[Calling tools]"
                _log(f"  🎯 [Target] Decided {len(response.tool_calls)} tool(s) in {elapsed:.2f}s")
                for tc in response.tool_calls:
                    _log(f"     → {tc['name']}")
                return ("tools", ai_msg_content, response.tool_calls)
                
        except Exception as e:
            _log(f"  ❌ [Target] Error: {e}", force=True)
            state.target_plan_time = time.time() - start_time
            return ("error", str(e), None)
    
    # Run both models in parallel!
    _log(f"  ⚡ Running draft and target SIMULTANEOUSLY (draft executes tools immediately)...")
    draft_task = asyncio.create_task(draft_predict_and_execute())
    target_task = asyncio.create_task(target_decide())
    
    # Wait for both to complete
    draft_tools, (target_type, target_content, target_data) = await asyncio.gather(draft_task, target_task)
    
    _log(f"  ✓ Both models completed!")
    
    state.draft_tool_calls = draft_tools
    state.draft_ready = True
    
    if target_type == "final":
        state.messages.append(Msg(role="assistant", content=target_content))
        state.answer = target_data
        state.done = True
        state.target_ready = True
        return state
    elif target_type == "thinking":
        state.messages.append(Msg(role="assistant", content=target_content))
        state.target_ready = True
        
        # If target is thinking without calling tools, treat the content as final answer
        # This handles cases where model provides answer without explicit "FINAL ANSWER:" format
        if target_content and len(target_content.strip()) > 10:
            _log(f"  ✅ [Route] Target provided response without tools, treating as final answer")
            state.done = True
            state.answer = target_content
        
        return state
    elif target_type == "tools":
        state.messages.append(Msg(role="assistant", content=target_content))
        state.target_tool_calls = target_data
        state.target_ready = True
    else:  # error
        state.target_ready = True
        return state
    
    return state


async def node_execute_with_cache(state: SpeculationState) -> SpeculationState:
    """
    Execute target's tool calls, checking draft cache first.
    Draft tools were ALREADY executed in parallel during planning!
    """
    if state.done or not state.target_tool_calls:
        return state
    
    _log(f"\n  🔍 [Execute] Verifying {len(state.target_tool_calls)} target tool(s) against draft cache...")
    
    start_time = time.time()
    
    for tool_call in state.target_tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        _log(f"tool_name: {tool_name}, tool_args: {tool_args}")
        
        # Try verification-based matching (handles AST exact match + semantic similarity)
        verification_result = await find_verified_match_in_cache(
            state.draft_cache,
            tool_name,
            tool_args,
            embedding_threshold=0.5,  # Can be made configurable
            verbose=False  # Set to True for debugging
        )
        
        if verification_result:
            # ✓ Verification passed! Use draft result
            verified_result, method, similarity = verification_result
            state.cache_hits += 1
            result_str = verified_result["result"]
            state.messages.append(Msg(role="tool", name=tool_name, content=result_str))
            state.verified_results.append(verified_result)

            # Track verification method for enhanced metrics
            if method == "ast":
                state.verified_by_ast += 1
                _log(f"     ⚡ EXACT HIT: {tool_name} (AST match, similarity={similarity:.3f})")
            elif method == "embedding":
                state.verified_by_embedding += 1
                _log(f"     ✓ SEMANTIC HIT: {tool_name} (method={method}, similarity={similarity:.3f})")

            # Track tool name for statistics
            state.target_tool_names.append(tool_name)

            # Record verification details for analysis
            state.verification_details.append({
                "target_tool": tool_name,
                "target_args": tool_args,
                "draft_tool": verified_result.get("tool"),
                "draft_args": verified_result.get("args"),
                "verified_by": method,
                "similarity": similarity,
                "cache_hit": True,
            })

        else:
            # Real MISS - draft predicted differently, execute now
            state.cache_misses += 1
            state.both_failed += 1  # Track failed verifications
            _log(f"     ❌ MISS: {tool_name} (executing now)")

            # Track tool name for statistics
            state.target_tool_names.append(tool_name)

            # Record miss details
            state.verification_details.append({
                "target_tool": tool_name,
                "target_args": tool_args,
                "verified_by": "none",
                "similarity": 0.0,
                "cache_hit": False,
            })
            
            if tool_name in TOOLS_BY_NAME:
                tool = TOOLS_BY_NAME[tool_name]
                try:
                    result = await tool.ainvoke(tool_args) if hasattr(tool, 'ainvoke') else tool.invoke(tool_args)
                    result_str = str(result)
                    state.messages.append(Msg(role="tool", name=tool_name, content=result_str))
                    state.verified_results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result_str,
                        "success": True
                    })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    state.messages.append(Msg(role="tool", name=tool_name, content=error_msg))
                    _log(f"     ❌ ERROR: {tool_name} ({e})", force=True)
            else:
                error_msg = f"Error: Unknown tool '{tool_name}'"
                state.messages.append(Msg(role="tool", name=tool_name, content=error_msg))
                _log(f"     ❌ ERROR: Unknown tool '{tool_name}'", force=True)
    
    state.verify_time = time.time() - start_time
    
    # Print statistics
    total = state.cache_hits + state.cache_misses
    hit_rate = (state.cache_hits / total * 100) if total > 0 else 0
    
    if total > 0:
        _log(f"\n  📊 [Stats] Cache performance: {state.cache_hits}/{total} hits ({hit_rate:.1f}%)")
        if state.cache_hits > 0:
            _log(f"     🎉 Speculation saved time by pre-executing {state.cache_hits} tool(s)!")
    
    return state


# -----------------------------
# Routing logic
# -----------------------------

def route_after_plan(state: SpeculationState) -> Literal["execute", END]:
    """Route after parallel planning: to execute if tools needed, otherwise end."""
    if state.done:
        return END
    if state.target_tool_calls:
        return "execute"
    return END


def route_after_execute(state: SpeculationState) -> Literal["plan", END]:
    """Route after execution: back to planning or end."""
    if state.done:
        return END
    if state.step >= config.max_steps:
        _log(f"  ⏹  [Route] Max steps ({config.max_steps}) reached", force=True)
        return END
    _log(f"  🔄 [Route] Continuing to next iteration")
    return "plan"


# -----------------------------
# Build graph
# -----------------------------

def build_speculation_graph():
    """
    Build speculation graph with parallel draft/target execution.
    
    Graph structure:
    
    START → plan (draft + target run in parallel here!)
               ↓
            execute (check cache, execute if needed)
               ↓
            (loop back to plan or END)
    """
    workflow = StateGraph(SpeculationState)
    
    # Add nodes
    workflow.add_node("plan", node_plan_parallel)  # Async node!
    workflow.add_node("execute", node_execute_with_cache)  # Async node!
    
    # Define edges
    workflow.add_edge(START, "plan")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "execute": "execute",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "execute",
        route_after_execute,
        {
            "plan": "plan",
            END: END
        }
    )
    
    # Compile with checkpointer
    return workflow.compile(checkpointer=MemorySaver())

