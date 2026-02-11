"""LangGraph nodes and graph construction using proper tool calling."""
import asyncio
import time
from typing import Literal, List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .models import RunState, Msg
from .llm_adapter import get_actor_model, get_spec_model, convert_msg_to_langchain, get_system_prompt, get_spec_system_prompt
from .tools_langchain import TOOLS_BY_NAME
from .config import config
from .verifier import create_verifier

# -----------------------------
# Speculation Helpers
# -----------------------------

# Create global verifier instance based on config
# Strategy options: "exact", "tool_name_only", "none"
_verifier = create_verifier(config.verification_strategy)


async def launch_speculation(state: RunState) -> None:
    """Launch spec model to predict and PRE-EXECUTE next tool call in parallel.
    
    Simple 1-step speculation with pre-execution:
    1. Spec model predicts next tool call
    2. IMMEDIATELY execute the tool with predicted args
    3. Cache both prediction and result
    
    This runs in parallel while actor is thinking.
    """
    if not config.enable_speculation:
        return
    
    spec_model = get_spec_model()
    
    # Convert messages to LangChain format
    # Use simplified prompt for spec model (fast prediction, low reasoning)
    lc_messages = [SystemMessage(content=get_spec_system_prompt())] + convert_msg_to_langchain(state.messages)
    
    # Call spec model to predict next tool call
    try:
        spec_start = time.time()
        response = spec_model.invoke(lc_messages)
        spec_inference_time = time.time() - spec_start
        
        # Get the first tool call prediction
        if hasattr(response, 'tool_calls') and response.tool_calls:
            prediction = response.tool_calls[0]
            tool_name = prediction["name"]
            tool_args = prediction["args"]
            
            # PRE-EXECUTE the tool with predicted args
            if tool_name in TOOLS_BY_NAME:
                tool = TOOLS_BY_NAME[tool_name]
                try:
                    exec_start = time.time()
                    result = await asyncio.to_thread(tool.invoke, tool_args)
                    exec_time = time.time() - exec_start
                    
                    # Store prediction WITH pre-executed result AND spec model time
                    state.speculative_predictions = [{
                        "name": tool_name,
                        "args": tool_args,
                        "result": str(result),
                        "exec_time": exec_time,
                        "spec_inference_time": spec_inference_time
                    }]
                    state.speculative_launched += 1
                    
                except Exception as e:
                    # Tool execution failed - store prediction without result
                    state.speculative_predictions = [{
                        "name": tool_name,
                        "args": tool_args,
                        "result": None,
                        "error": str(e),
                        "exec_time": 0,
                        "spec_inference_time": spec_inference_time
                    }]
            else:
                # Unknown tool - just store prediction
                state.speculative_predictions = [{
                    "name": tool_name,
                    "args": tool_args,
                    "result": None,
                    "exec_time": 0,
                    "spec_inference_time": spec_inference_time
                }]
        else:
            state.speculative_predictions = []
            
    except Exception as e:
        print(f"⚠️  Speculation error: {e}")
        state.speculative_predictions = []


def find_best_match(
    actor_tool_call: Dict[str, Any],
    spec_predictions: List[Dict[str, Any]]
) -> Optional[tuple[Dict[str, Any], bool, float, str]]:
    """Find matching speculative prediction for actor's tool call using verifier.
    
    Args:
        actor_tool_call: {"name": str, "args": dict}
        spec_predictions: List of predicted tool calls with pre-executed results
    
    Returns:
        Tuple of (prediction, is_match, similarity_score, reason) or None if no match
    """
    if not spec_predictions:
        return None
    
    actor_name = actor_tool_call["name"]
    actor_args = actor_tool_call["args"]
    
    # Try to find a match using the verifier
    best_match = None
    best_score = 0.0
    best_reason = ""
    
    for spec_pred in spec_predictions:
        spec_name = spec_pred["name"]
        spec_args = spec_pred.get("args", {})
        
        # Use verifier to check if this prediction matches
        is_match, score, reason = _verifier.verify(
            actor_name, actor_args,
            spec_name, spec_args
        )
        
        if is_match and score > best_score:
            best_match = spec_pred
            best_score = score
            best_reason = reason
    
    if best_match:
        return (best_match, True, best_score, best_reason)
    
    # No match found - return info about best attempt for logging
    if spec_predictions:
        # Get the first prediction for comparison info
        first_pred = spec_predictions[0]
        _, score, reason = _verifier.verify(
            actor_name, actor_args,
            first_pred["name"], first_pred.get("args", {})
        )
        return (first_pred, False, score, reason)
    
    return None


# -----------------------------
# Graph nodes - ReAct Pattern
# -----------------------------

async def node_llm(state: RunState) -> RunState:
    """
    LLM node: Decides what to do next (Reason + Act).
    Either calls tools or provides final answer.
    
    Runs actor and spec model IN PARALLEL for 1-step speculation.
    """
    if state.done:
        return state
    
    # Increment step at the start (this is LLM call #N)
    state.step += 1
    
    # Check if max steps reached - force final answer
    if state.step > config.max_steps:
        print(f"\n⚠️  MAX STEPS REACHED ({config.max_steps}). Forcing final answer...")
        
        # Get model WITHOUT tools to force text response
        from .llm_adapter import _create_model
        model_no_tools = _create_model(config.actor_model, is_spec=False)
        
        # Build messages with conversation history (formatting rules already in system prompt)
        lc_messages = [SystemMessage(content=get_system_prompt())] + convert_msg_to_langchain(state.messages)
        lc_messages.append(HumanMessage(content=(
            "You have reached the maximum number of steps. Based on all the information gathered above, "
            "provide your best final answer now. Remember to follow the formatting rules in the system prompt."
        )))
        
        start_time = time.time()
        response = model_no_tools.invoke(lc_messages)
        elapsed = time.time() - start_time
        
        # DEBUG: Print forced answer response directly
        print(f"\n🔍 DEBUG FORCED ANSWER OUTPUT:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
        state.last_llm_time = elapsed
        
        # Extract answer
        content = response.content
        if "FINAL ANSWER:" in content.upper():
            answer_part = content.split("FINAL ANSWER:")[-1] if "FINAL ANSWER:" in content else content
            answer = answer_part.strip()
        else:
            # Use entire content as answer
            answer = content.strip()
        
        state.messages.append(Msg(role="assistant", content=content))
        state.answer = answer
        state.done = True
        
        return state
    
    # Get model with tools
    model = get_actor_model()
    
    # Convert messages to LangChain format
    lc_messages = [SystemMessage(content=get_system_prompt())] + convert_msg_to_langchain(state.messages)
    
    # ⚡ PARALLEL EXECUTION: Run actor and spec model at the same time
    start_time = time.time()
    
    # Create tasks for parallel execution
    actor_task = asyncio.create_task(asyncio.to_thread(model.invoke, lc_messages))
    spec_task = asyncio.create_task(launch_speculation(state))
    
    # Wait for both to complete
    response, _ = await asyncio.gather(actor_task, spec_task)
    
    elapsed = time.time() - start_time
    
    # Store timing in state
    state.last_llm_time = elapsed
    
    # DEBUG: Print full model response directly
    print(f"\n🔍 DEBUG MODEL OUTPUT (Step {state.step}):")
    print("-" * 80)
    print(response)
    print("-" * 80)
    
    # Check if final answer (no tool calls)
    if not response.tool_calls:
        content = response.content
        
        # Check if this is a final answer
        if "FINAL ANSWER:" in content.upper():
            # Extract answer
            answer_part = content.split("FINAL ANSWER:")[-1] if "FINAL ANSWER:" in content else content
            answer = answer_part.strip()
            
            state.messages.append(Msg(role="assistant", content=content))
            state.answer = answer
            state.done = True
        else:
            # Just thinking/reasoning
            state.messages.append(Msg(role="assistant", content=content))
        
        return state
    
    # Has tool calls - store them for execution
    # Store the AI message with metadata about tool calls
    ai_msg_content = response.content or "[Calling tools]"
    state.messages.append(Msg(role="assistant", content=ai_msg_content))
    
    # Store tool calls in state for the tool node to execute
    state.pending_tool_calls = response.tool_calls
    
    return state


async def node_tools(state: RunState) -> RunState:
    """
    Tools node: Executes pending tool calls (Observe).
    Returns results to LLM.
    
    Checks speculation cache first - if match found, uses predicted args.
    Otherwise executes with actor's args.
    """
    if state.done:
        return state
        
    if not hasattr(state, 'pending_tool_calls') or not state.pending_tool_calls:
        return state
    
    # Execute each tool call
    for tool_call in state.pending_tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Check if we have speculative predictions from previous step
        spec_predictions = getattr(state, 'speculative_predictions', [])
        
        # Try to find matching speculation using verifier
        match_result = find_best_match(
            {"name": tool_name, "args": tool_args},
            spec_predictions
        )
        
        # Display verification results if speculation is enabled
        if config.enable_speculation and spec_predictions:
            print(f"\n🔍 VERIFICATION: Checking {len(spec_predictions)} prediction(s)")
            print("=" * 80)
            
            if match_result:
                spec_pred, is_match, score, reason = match_result
                spec_name = spec_pred["name"]
                spec_args = spec_pred["args"]
                has_cached_result = spec_pred.get("result") is not None
                
                # Show actor's request
                print(f"Actor wants: {tool_name}({tool_args})")
                print(f"Spec cached: {spec_name}({spec_args})")
                print(f"")
                
                # Show verification result
                if is_match:
                    print(f"✅ MATCH: {reason} (score: {score:.2f})")
                    if has_cached_result:
                        exec_time = spec_pred.get("exec_time", 0)
                        print(f"   Pre-executed in {exec_time:.2f}s, ready to use!")
                    else:
                        print(f"   ⚠️ But no cached result available")
                else:
                    print(f"❌ REJECTED: {reason} (score: {score:.2f})")
            else:
                print("  No predictions available")
            
            print("=" * 80)
        
        if tool_name in TOOLS_BY_NAME:
            tool = TOOLS_BY_NAME[tool_name]
            try:
                start_time = time.time()
                
                # Check if we have a valid cache hit
                use_cached = False
                if match_result:
                    spec_pred, is_match, score, reason = match_result
                    if is_match and spec_pred.get("result") is not None:
                        use_cached = True
                
                if use_cached:
                    # ⚡ CACHE HIT: Use pre-executed result!
                    if config.enable_speculation:
                        print(f"\n✅ USING CACHED RESULT")
                    result_str = spec_pred["result"]
                    state.hits += 1
                    # Report the pre-execution time (this is what we saved)
                    elapsed = spec_pred.get("exec_time", 0.001)
                else:
                    # No match or no cached result - execute with actor's args
                    if config.enable_speculation:
                        if match_result:
                            _, is_match, _, _ = match_result
                            if not is_match:
                                print(f"\n❌ EXECUTING (verification failed)")
                            else:
                                print(f"\n⚠️  EXECUTING (no cached result)")
                        else:
                            print(f"\n❌ EXECUTING (no predictions)")
                    result = await asyncio.to_thread(tool.invoke, tool_args)
                    result_str = str(result)
                    state.misses += 1
                    elapsed = time.time() - start_time
                
                # Store timing
                state.last_tool_time = elapsed
                
                state.messages.append(Msg(role="tool", name=tool_name, content=result_str))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                state.messages.append(Msg(role="tool", name=tool_name, content=error_msg))
                state.misses += 1
        else:
            error_msg = f"Error: Unknown tool '{tool_name}'"
            state.messages.append(Msg(role="tool", name=tool_name, content=error_msg))
    
    # Clear pending tool calls and predictions
    state.pending_tool_calls = []
    state.speculative_predictions = []
    
    return state


def route_after_llm(state: RunState) -> Literal["tools", "llm", END]:
    """Route after LLM: to tools if there are tool calls, back to llm if just thinking, or end if done."""
    if state.done:
        return END
    if hasattr(state, 'pending_tool_calls') and state.pending_tool_calls:
        return "tools"
    # No tool calls and not done - continue reasoning
    return "llm"


def route_after_tools(state: RunState) -> Literal["llm", END]:
    """Route after tools: back to LLM for reasoning."""
    if state.done:
        return END
    # Always go back to LLM (even if max steps reached - LLM will be forced to answer)
    return "llm"


# -----------------------------
# Build graph
# -----------------------------

def build_graph():
    """Build and compile the LangGraph agent with ReAct pattern."""
    workflow = StateGraph(RunState)
    
    # Add nodes
    workflow.add_node("llm", node_llm)
    workflow.add_node("tools", node_tools)
    
    # Define edges
    workflow.add_edge(START, "llm")
    
    # LLM can go to tools, loop back to itself, or end
    workflow.add_conditional_edges(
        "llm",
        route_after_llm,
        {"tools": "tools", "llm": "llm", END: END}
    )
    
    # Tools always go back to LLM (for reasoning about results)
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {"llm": "llm", END: END}
    )
    
    # Compile with checkpointer
    # Note: recursion_limit is set at runtime in eval.py, not at compile time
    return workflow.compile(checkpointer=MemorySaver())
