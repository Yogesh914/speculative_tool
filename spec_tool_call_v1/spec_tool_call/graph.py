"""LangGraph nodes and graph construction using proper tool calling."""
from typing import Literal, Any
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .models import RunState, Msg
from .llm_adapter import get_actor_model, convert_msg_to_langchain, SYSTEM_PROMPT
from .tools_langchain import TOOLS_BY_NAME
from .config import config


# -----------------------------
# Helper functions
# -----------------------------

def _extract_content_str(content: Any) -> str:
    """Extract string content from response (handles both str and list formats)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Some models return content as list of dicts or strings
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and 'text' in item:
                parts.append(item['text'])
        return ' '.join(parts) if parts else ""
    return str(content) if content else ""


# -----------------------------
# Graph nodes - ReAct Pattern
# -----------------------------

def node_llm(state: RunState) -> RunState:
    """
    LLM node: Decides what to do next (Reason + Act).
    Either calls tools or provides final answer.
    """
    if state.done:
        return state
    
    # Get model with tools
    model = get_actor_model()
    
    # Convert messages to LangChain format
    lc_messages = [SystemMessage(content=SYSTEM_PROMPT)] + convert_msg_to_langchain(state.messages)
    
    # Call LLM with timing
    import time
    start_time = time.time()
    response = model.invoke(lc_messages)
    elapsed = time.time() - start_time
    
    # Store timing in state
    state.last_llm_time = elapsed
    
    # Increment step
    state.step += 1
    
    # Check if final answer (no tool calls)
    if not response.tool_calls:
        content = _extract_content_str(response.content)
        
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
            
            # If we've reached max steps and have content, treat it as final answer
            if state.step >= config.max_steps and content:
                state.answer = content
                state.done = True
        
        return state
    
    # Has tool calls - store them for execution
    # Store the AI message with metadata about tool calls
    ai_msg_content = _extract_content_str(response.content) or "[Calling tools]"
    state.messages.append(Msg(role="assistant", content=ai_msg_content))
    
    # Store tool calls in state for the tool node to execute
    state.pending_tool_calls = response.tool_calls
    
    return state


def node_tools(state: RunState) -> RunState:
    """
    Tools node: Executes pending tool calls (Observe).
    Returns results to LLM.
    """
    if state.done:
        return state
        
    if not hasattr(state, 'pending_tool_calls') or not state.pending_tool_calls:
        return state
    
    # Execute each tool call
    import time
    for tool_call in state.pending_tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name in TOOLS_BY_NAME:
            tool = TOOLS_BY_NAME[tool_name]
            try:
                start_time = time.time()
                result = tool.invoke(tool_args)
                elapsed = time.time() - start_time
                
                # Store timing
                state.last_tool_time = elapsed
                
                result_str = str(result)
                state.messages.append(Msg(role="tool", name=tool_name, content=result_str))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                state.messages.append(Msg(role="tool", name=tool_name, content=error_msg))
        else:
            error_msg = f"Error: Unknown tool '{tool_name}'"
            state.messages.append(Msg(role="tool", name=tool_name, content=error_msg))
    
    # Clear pending tool calls
    state.pending_tool_calls = []
    
    return state


def route_after_llm(state: RunState) -> Literal["tools", END]:
    """Route after LLM: to tools if there are tool calls, otherwise end."""
    if state.done:
        return END
    if hasattr(state, 'pending_tool_calls') and state.pending_tool_calls:
        return "tools"
    return END


def route_after_tools(state: RunState) -> Literal["llm", END]:
    """Route after tools: back to LLM for reasoning."""
    if state.done:
        return END
    if state.step >= config.max_steps:
        return END
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
    
    # LLM can go to tools or end
    workflow.add_conditional_edges(
        "llm",
        route_after_llm,
        {"tools": "tools", END: END}
    )
    
    # Tools always go back to LLM (for reasoning about results)
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {"llm": "llm", END: END}
    )
    
    # Compile with checkpointer
    return workflow.compile(checkpointer=MemorySaver())
