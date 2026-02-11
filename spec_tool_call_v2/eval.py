#!/usr/bin/env python3
"""Evaluation script for GAIA examples."""
import asyncio
import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env BEFORE importing config
load_dotenv()

from spec_tool_call import build_graph, Msg
from spec_tool_call.models import RunState
from spec_tool_call.config import config


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison by removing spaces around punctuation."""
    import re
    # Remove all whitespace and convert to lowercase
    normalized = answer.lower().strip()
    # Remove spaces around commas, periods, and other punctuation
    normalized = re.sub(r'\s*([,;.!?])\s*', r'\1', normalized)
    return normalized


async def run_example(example_dir: Path, app=None, verbose=True):
    """Run a single GAIA example."""
    
    # Build graph if not provided
    if app is None:
        app = build_graph()
    
    # Load metadata
    metadata_path = example_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    task_id = metadata.get("task_id", "unknown")
    question = metadata.get("question", "")
    final_answer = metadata.get("final_answer", "")
    level = metadata.get("level", "")
    file_name = metadata.get("file_name", None)
    
    # Get example name from directory (e.g., "example_007" from path)
    example_name = example_dir.name
    
    # If there's an attached file, add it to the question context with full path
    if file_name:
        file_path = example_dir / file_name
        if file_path.exists():
            # Modify question to include file path information
            question = f"{question}\n\n[Note: There is an attached file at path: {file_path.absolute()}]"
    
    if verbose:
        print("=" * 80)
        print(f"GAIA Example: {task_id} (Level {level})")
        print("=" * 80)
        print(f"\nQuestion:\n{question}\n")
        print(f"Ground Truth: {final_answer}")
        print("\n" + "=" * 80)
        print(f"Actor Model: {config.actor_model}")
        if config.enable_speculation:
            print(f"Spec Model:  {config.spec_model}")
            print(f"Speculation: ENABLED")
        else:
            print(f"Speculation: DISABLED")
        print(f"Max Steps:   {config.max_steps}")
        print("=" * 80 + "\n")
    
    # Build initial state
    init_state = RunState(messages=[Msg(role="user", content=question)])
    
    start_time = time.time()
    final_state = None
    
    # Track detailed trajectory
    trajectory_events = [
        {
            "node": "start",
            "question": question,
            "timestamp": 0
        }
    ]
    
    try:
        # Set recursion limit: each step = 2 nodes (llm + tools), so max_steps * 3 for safety
        recursion_limit = config.max_steps * 3
        
        async for event in app.astream(
            init_state,
            config={
                "configurable": {"thread_id": f"eval-{task_id}"},
                "recursion_limit": recursion_limit
            }
        ):
            for node_name, state in event.items():
                final_state = state
                
                # Capture trajectory event details
                event_data = {
                    "node": node_name,
                    "timestamp": time.time() - start_time
                }
                
                # Extract state info
                if isinstance(state, dict):
                    messages = state.get('messages', [])
                    done = state.get('done', False)
                    answer = state.get('answer', None)
                    pending_tools = state.get('pending_tool_calls', [])
                    llm_time = state.get('last_llm_time', 0)
                    tool_time = state.get('last_tool_time', 0)
                    step = state.get('step', 0)
                    spec_predictions = state.get('speculative_predictions', [])
                else:
                    messages = state.messages
                    done = state.done
                    answer = state.answer
                    pending_tools = state.pending_tool_calls
                    llm_time = state.last_llm_time
                    tool_time = state.last_tool_time
                    step = state.step
                    spec_predictions = getattr(state, 'speculative_predictions', [])
                
                # Capture LLM node details
                if node_name == "llm":
                    event_data["step"] = step
                    event_data["actor_model"] = config.actor_model
                    event_data["llm_time"] = llm_time
                    
                    # Get actor's reasoning/content
                    if messages and messages[-1].role == "assistant":
                        event_data["actor_reasoning"] = messages[-1].content
                    
                    # Get actor's tool calls
                    if pending_tools:
                        event_data["actor_tool_calls"] = [
                            {
                                "name": tc["name"],
                                "args": tc["args"]
                            }
                            for tc in pending_tools
                        ]
                    
                    # Get spec model predictions
                    if config.enable_speculation and spec_predictions:
                        event_data["spec_model"] = config.spec_model
                        event_data["spec_predictions"] = [
                            {
                                "name": pred["name"],
                                "args": pred.get("args", {}),
                                "pre_executed": pred.get("result") is not None,
                                "exec_time": pred.get("exec_time", 0),
                                "spec_inference_time": pred.get("spec_inference_time", 0)
                            }
                            for pred in spec_predictions
                        ]
                    
                    # Final answer
                    if done and answer:
                        event_data["final_answer"] = answer
                
                # Capture tool execution details
                elif node_name == "tools":
                    event_data["tool_time"] = tool_time
                    
                    # Get tool results from last message
                    tool_msgs = [m for m in messages if m.role == "tool"]
                    if tool_msgs:
                        latest_tool = tool_msgs[-1]
                        event_data["tool_executed"] = {
                            "name": latest_tool.name if hasattr(latest_tool, 'name') else "unknown",
                            "result": latest_tool.content
                        }
                
                trajectory_events.append(event_data)
                
                # Show LLM step (only if verbose)
                if verbose and node_name == "llm":
                    print(f"\n{'='*80}")
                    print(f"[Step {step}] LLM")
                    print(f"{'='*80}")
                    print(f"⏱️  Actor model: {llm_time:.2f}s")
                    
                    # Show spec model timing if available
                    if config.enable_speculation and spec_predictions:
                        for pred in spec_predictions:
                            spec_time = pred.get("spec_inference_time", 0)
                            tool_time = pred.get("exec_time", 0)
                            print(f"⏱️  Spec model:  {spec_time:.2f}s (+ {tool_time:.2f}s pre-execution)")
                    print()
                    
                    # Show LLM's text content if it exists
                    if messages and messages[-1].role == "assistant":
                        content = messages[-1].content
                        # Filter out placeholder text
                        if content and content.strip() and content.strip() not in ["[Calling tools]", "[Tool call]"]:
                            print(f"💭 Reasoning:")
                            if len(content) > 300:
                                print(f"   {content[:300]}...")
                            else:
                                print(f"   {content}")
                            print()
                    
                    # Show tool calls if any
                    if pending_tools:
                        print(f"🔧 Tool Call:")
                        for tc in pending_tools:
                            print(f"   Tool: {tc['name']}")
                            print(f"   Args:")
                            for k, v in tc['args'].items():
                                if isinstance(v, str) and len(v) > 80:
                                    v_display = v[:80] + "..."
                                else:
                                    v_display = v
                                print(f"      {k} = {v_display}")
                    
                    # Show final answer if done
                    if done:
                        print(f"✅ Final Answer: {answer}")
                
                # Show tool execution (only if verbose)
                elif verbose and node_name == "tools":
                    print(f"\n{'='*80}")
                    print(f"[Step {step}] TOOLS")
                    print(f"{'='*80}")
                    print(f"⏱️  Execution: {tool_time:.2f}s\n")
                    
                    tool_msgs = [m for m in messages if m.role == "tool"]
                    if tool_msgs:
                        latest_tool = tool_msgs[-1]
                        tool_name = latest_tool.name if hasattr(latest_tool, 'name') else "unknown"
                        result = latest_tool.content
                        
                        print(f"📤 Output from '{tool_name}':")
                        print("-" * 80)
                        
                        if "Error:" in result:
                            print(f"   ❌ {result}")
                        else:
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
                        print("-" * 80)
                
                if done:
                    print(f"\n{'='*80}")
                    print("✅ EXECUTION COMPLETE")
                    print(f"{'='*80}")
                    break
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        error_msg = str(e)
        # Continue to save partial results even on error
    else:
        error_msg = None
    
    elapsed = time.time() - start_time
    
    # Extract final results (even if there was an error)
    predicted = None
    hits = 0
    misses = 0
    spec_launched = 0
    steps = 0
    correct = False
    
    if final_state:
        if isinstance(final_state, dict):
            predicted = final_state.get('answer', None)
            hits = final_state.get('hits', 0)
            misses = final_state.get('misses', 0)
            spec_launched = final_state.get('speculative_launched', 0)
            steps = final_state.get('step', 0)
        else:
            predicted = final_state.answer
            hits = final_state.hits
            misses = final_state.misses
            spec_launched = final_state.speculative_launched
            steps = final_state.step
        
    
    # Check correctness
    if predicted and final_answer:
        correct = normalize_answer(predicted) == normalize_answer(final_answer)
    
    if verbose:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"\nPredicted: {predicted if predicted else 'No answer'}")
        print(f"Truth:     {final_answer}")
        print(f"\nCorrect:   {'✓ YES' if correct else '✗ NO'}")
        print(f"Steps:     {steps}")
        print(f"Time:      {elapsed:.1f}s")
        
        if error_msg:
            print(f"\n❌ Error occurred: {error_msg}")
        
        if config.enable_speculation:
            print("\n" + "=" * 80)
            print("SPECULATION")
            print("=" * 80)
            print(f"Hits:        {hits}")
            print(f"Misses:      {misses}")
            print(f"Predictions: {spec_launched}")
            total_checks = hits + misses
            if total_checks > 0:
                hit_rate = hits / total_checks * 100
                print(f"Hit Rate:    {hit_rate:.1f}%")
            
            # Simple timing breakdown
            if total_checks > 0:
                total_actor_time = sum(e.get("llm_time", 0) for e in trajectory_events if e.get("node") == "llm")
                total_spec_time = sum(
                    pred.get("spec_inference_time", 0) 
                    for e in trajectory_events if e.get("node") == "llm" 
                    for pred in e.get("spec_predictions", [])
                )
                total_tool_exec_time = sum(
                    pred.get("exec_time", 0) 
                    for e in trajectory_events if e.get("node") == "llm" 
                    for pred in e.get("spec_predictions", []) 
                    if pred.get("pre_executed")
                )
                
                print(f"\nTiming Breakdown:")
                print(f"  Total actor time: {total_actor_time:.2f}s")
                print(f"  Total spec time:  {total_spec_time:.2f}s")
                print(f"  Total tool exec:  {total_tool_exec_time:.2f}s")
        
    print("\n" + "=" * 80 + "\n")
    
    # Extract simple message trajectory
    simple_trajectory = []
    if final_state:
        if isinstance(final_state, dict):
            messages = final_state.get('messages', [])
        else:
            messages = final_state.messages
        
        # Convert messages to serializable format
        for msg in messages:
            msg_dict = {
                "role": msg.role,
                "content": msg.content,
            }
            if hasattr(msg, 'name') and msg.name:
                msg_dict["name"] = msg.name
            simple_trajectory.append(msg_dict)
    
    # Save results to JSON with complete trajectory (even on error)
    spec_status = "spec" if config.enable_speculation else "baseline"
    result_file = f"result_level{level}_{example_name}_{spec_status}.json"
    
    result_data = {
        "task_id": task_id,
        "level": level,
        "question": question,
        "ground_truth": final_answer,
        "predicted": predicted,
        "correct": correct,
        "steps": steps,
        "time_seconds": elapsed,
        "actor_model": config.actor_model,
        "spec_model": config.spec_model if config.enable_speculation else None,
        "speculation_enabled": config.enable_speculation,
        "verification_strategy": config.verification_strategy if config.enable_speculation else None,
        "spec_hits": hits if config.enable_speculation else 0,
        "spec_misses": misses if config.enable_speculation else 0,
        "spec_predictions": spec_launched if config.enable_speculation else 0,
        "trajectory": trajectory_events,  # Detailed step-by-step trajectory
        "messages": simple_trajectory,  # Simple message history
        "error": error_msg,  # Include error if present
    }
    
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)
    
    if verbose:
        print(f"💾 Results saved to: {result_file}\n")
    
    return result_data


async def run_batch(level_dir: Path, output_dir: Path, max_examples: int = None, start_idx: int = None, end_idx: int = None):
    """Run batch evaluation on all examples in a directory."""
    
    # Find all examples
    examples = sorted([
        d for d in level_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])
    
    if not examples:
        print(f"No examples found in {level_dir}")
        return
    
    # Apply start/end slicing first if specified
    if start_idx is not None or end_idx is not None:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else len(examples)
        examples = examples[start:end]
        print(f"Sliced examples from index {start} to {end}")
    
    # Then limit to max_examples if specified
    elif max_examples:
        examples = examples[:max_examples]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    spec_status = "spec" if config.enable_speculation else "baseline"
    print("=" * 80)
    print(f"BATCH EVALUATION: {level_dir.name}")
    print("=" * 80)
    print(f"Examples:    {len(examples)}")
    if start_idx is not None or end_idx is not None:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else "end"
        print(f"Range:       [{start}:{end}]")
    print(f"Max Steps:   {config.max_steps}")
    print(f"Speculation: {'ENABLED' if config.enable_speculation else 'DISABLED (baseline)'}")
    print(f"Output Dir:  {output_dir}")
    print("=" * 80 + "\n")
    
    # Build graph once
    app = build_graph()
    
    count = 0
    correct = 0
    total_time = 0
    
    for i, example_dir in enumerate(examples, 1):
        example_name = example_dir.name
        count += 1
        
        print(f"\n{'='*80}")
        print(f"[{i}/{len(examples)}] {example_name}")
        print(f"{'='*80}\n")
        
        try:
            result = await run_example(example_dir, app, verbose=False)
            
            # Move result file to output directory
            result_files = list(Path(".").glob(f"result_*_{spec_status}.json"))
            if result_files:
                result_file = result_files[0]
                dest = output_dir / result_file.name
                result_file.rename(dest)
                
                # Read for stats
                with open(dest) as f:
                    data = json.load(f)
                
                is_correct = data.get("correct", False)
                time_taken = data.get("time_seconds", 0)
                
                if is_correct:
                    correct += 1
                    print(f"\n✅ CORRECT ({time_taken:.1f}s)")
                else:
                    pred = data.get("predicted", "")
                    truth = data.get("ground_truth", "")
                    print(f"\n❌ WRONG ({time_taken:.1f}s)")
                    print(f"   Predicted: {pred}")
                    print(f"   Truth:     {truth}")
                
                total_time += time_taken
            else:
                print(f"\n❌ ERROR: No result file found")
        
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
    
    # Final summary
    accuracy = correct / count * 100 if count > 0 else 0
    avg_time = total_time / count if count > 0 else 0
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {correct}/{count} ({accuracy:.1f}%)")
    print(f"Avg Time:  {avg_time:.1f}s")
    print(f"Total:     {total_time:.1f}s")
    print(f"\nResults saved to: {output_dir}/")
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate GAIA examples")
    parser.add_argument("example", nargs="?", help="Path to example directory (for single mode)")
    parser.add_argument("--batch", action="store_true", help="Run batch evaluation")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1,
                       help="GAIA level for batch mode (default: 1)")
    parser.add_argument("--max", type=int, help="Maximum number of examples to run (default: all)")
    parser.add_argument("--start", type=int, help="Start index for examples (inclusive, 0-based)")
    parser.add_argument("--end", type=int, help="End index for examples (exclusive, 0-based)")
    parser.add_argument("--output", help="Output directory for batch results (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Batch mode
    if args.batch:
        level_dir = Path(f"gaia_dataset/level{args.level}")
        if not level_dir.exists():
            print(f"Error: {level_dir} not found")
            sys.exit(1)
        
        if args.output:
            output_dir = Path(args.output)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            spec_status = "spec" if config.enable_speculation else "baseline"
            output_dir = Path(f"results_level{args.level}_{spec_status}_{timestamp}")
        
        asyncio.run(run_batch(level_dir, output_dir, max_examples=args.max, 
                             start_idx=args.start, end_idx=args.end))
    
    # Single example mode
    else:
        if args.example:
            example_dir = Path(args.example)
        else:
            example_dir = Path("gaia_dataset/level1/example_000")
        
        if not example_dir.exists():
            print(f"Error: {example_dir} not found")
            sys.exit(1)
        
        asyncio.run(run_example(example_dir))


if __name__ == "__main__":
    main()

# python eval.py gaia_dataset/level1/example_000