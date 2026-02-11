#!/usr/bin/env python3
"""Run actor model on a specific GAIA example."""
import asyncio
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

from spec_tool_call import build_graph, Msg
from spec_tool_call.models import RunState

load_dotenv()


async def run_gaia_example(example_dir: str):
    """Run actor model on a GAIA example."""
    
    # Load metadata
    metadata_path = Path(example_dir) / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found")
        return
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    task_id = metadata.get("task_id", "unknown")
    question = metadata.get("question", "")
    final_answer = metadata.get("final_answer", "")
    level = metadata.get("level", "")
    
    print("=" * 80)
    print(f"GAIA Example: {task_id}")
    print(f"Level: {level}")
    print("=" * 80)
    print(f"\nQuestion:\n{question}\n")
    print(f"Ground Truth Answer: {final_answer}")
    
    # Get model info
    from spec_tool_call.config import config
    
    print("\n" + "=" * 80)
    print(f"Running Actor Model: {config.actor_model}")
    print("=" * 80 + "\n")
    
    # Build graph and run
    app = build_graph()
    init_state = RunState(messages=[Msg(role="user", content=question)])
    
    final_state = None
    step = 0
    
    async for event in app.astream(
        init_state,
        config={"configurable": {"thread_id": f"gaia-{task_id}"}}
    ):
        for node_name, state in event.items():
            step += 1
            
            # Extract state info
            if isinstance(state, dict):
                messages = state.get('messages', [])
                done = state.get('done', False)
                answer = state.get('answer', None)
                pending_tools = state.get('pending_tool_calls', [])
                llm_time = state.get('last_llm_time', 0)
                tool_time = state.get('last_tool_time', 0)
            else:
                messages = state.messages
                done = state.done
                answer = state.answer
                pending_tools = state.pending_tool_calls
                llm_time = state.last_llm_time
                tool_time = state.last_tool_time
            
            print(f"\n{'='*80}")
            print(f"[Step {step}] {node_name.upper()}")
            print(f"{'='*80}")
            
            # Show what happened in this node
            if node_name == "llm":
                # LLM step - show timing and decision
                print(f"⏱️  LLM call: {llm_time:.2f}s")
                
                if pending_tools:
                    # LLM decided to use tools
                    print(f"🔧 Decision: Call tool")
                    for tc in pending_tools:
                        print(f"   Tool: {tc['name']}")
                        print(f"   Args:")
                        for k, v in tc['args'].items():
                            # Format value nicely
                            if isinstance(v, str) and len(v) > 80:
                                v_display = v[:80] + "..."
                            else:
                                v_display = v
                            print(f"      {k} = {v_display}")
                elif done:
                    # LLM provided final answer
                    print(f"✅ Decision: Provide final answer")
                    print(f"   Answer: {answer}")
                else:
                    # LLM is thinking
                    if messages and messages[-1].role == "assistant":
                        thinking = messages[-1].content
                        print(f"💭 Decision: Continue reasoning")
                        print(f"   Thought: {thinking[:150]}...")
                        
            elif node_name == "tools":
                # Tool step - show execution and result
                tool_msgs = [m for m in messages if m.role == "tool"]
                if tool_msgs:
                    latest_tool = tool_msgs[-1]
                    tool_name = latest_tool.name if hasattr(latest_tool, 'name') else "unknown"
                    result = latest_tool.content
                    
                    print(f"⏱️  Execution: {tool_time:.2f}s")
                    print(f"📤 Output from '{tool_name}':")
                    
                    # Show preview of result
                    if "Error:" in result:
                        print(f"   ❌ {result}")
                    else:
                        # Show first few lines/chars
                        lines = result.split('\n')
                        if len(lines) > 10:
                            preview = '\n'.join(lines[:10])
                            print(f"   {preview}")
                            print(f"   ... ({len(lines)-10} more lines)")
                        elif len(result) > 500:
                            print(f"   {result[:500]}")
                            print(f"   ... (truncated, {len(result)} total chars)")
                        else:
                            print(f"   {result}")
            
            final_state = state
            
            if done:
                print(f"\n{'='*80}")
                print("✅ EXECUTION COMPLETE")
                print(f"{'='*80}")
                break
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    if final_state:
        if isinstance(final_state, dict):
            predicted = final_state.get('answer', 'No answer provided')
            messages = final_state.get('messages', [])
        else:
            predicted = final_state.answer if hasattr(final_state, 'answer') else 'No answer provided'
            messages = final_state.messages if hasattr(final_state, 'messages') else []
        
        print(f"\nPredicted Answer: {predicted if predicted else 'No answer provided'}")
        print(f"Ground Truth:     {final_answer}")
        
        # Check correctness
        if predicted and final_answer:
            correct = predicted.lower().strip() == final_answer.lower().strip()
            print(f"\nCorrect: {'✓ YES' if correct else '✗ NO'}")
        else:
            print(f"\nCorrect: ✗ NO (no answer provided)")
        print(f"Total Steps: {step}")
        print(f"Total Messages: {len(messages)}")
    else:
        print("\nNo final state - execution failed")
    
    print("\n" + "=" * 80)
    
    return final_state


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        example_dir = sys.argv[1]
    else:
        # Default to level1/example_000
        example_dir = "gaia_dataset/level1/example_000"
    
    asyncio.run(run_gaia_example(example_dir))


if __name__ == "__main__":
    main()

