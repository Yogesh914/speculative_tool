#!/usr/bin/env python3
"""Benchmark speculation: Compare speculator vs actor tool predictions on step 1."""
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from spec_tool_call.llm_adapter import SYSTEM_PROMPT, get_actor_model, get_spec_model
from spec_tool_call.config import config

load_dotenv()


def calculate_word_similarity(text1: str, text2: str) -> float:
    """Calculate word-level similarity between two strings.
    
    Returns percentage (0-100) of matching words.
    """
    # Normalize: lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 100.0
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity: intersection / union
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return (intersection / union * 100) if union > 0 else 0.0


def calculate_args_similarity(args1: Dict[str, Any], args2: Dict[str, Any]) -> float:
    """Calculate similarity percentage between two argument dictionaries.
    
    Returns percentage (0-100) of matching key-value pairs.
    Ignores 'expand_search' parameter as it's not critical for cache hits.
    For query strings, uses word-level matching.
    """
    # Filter out expand_search from both args
    args1_filtered = {k: v for k, v in args1.items() if k != 'expand_search'}
    args2_filtered = {k: v for k, v in args2.items() if k != 'expand_search'}
    
    if not args1_filtered and not args2_filtered:
        return 100.0
    
    if not args1_filtered or not args2_filtered:
        return 0.0
    
    # Get all keys from both dicts (excluding expand_search)
    all_keys = set(args1_filtered.keys()) | set(args2_filtered.keys())
    if not all_keys:
        return 100.0
    
    total_similarity = 0
    for key in all_keys:
        val1 = args1_filtered.get(key)
        val2 = args2_filtered.get(key)
        
        # Exact match
        if val1 == val2:
            total_similarity += 100
        # String similarity (if both are strings) - use word matching
        elif isinstance(val1, str) and isinstance(val2, str):
            # For 'query' parameter, use word-level similarity
            if key == 'query':
                total_similarity += calculate_word_similarity(val1, val2)
            else:
                # For other strings, use exact match
                total_similarity += 100 if val1.lower() == val2.lower() else 0
        # Type mismatch or one is None
        else:
            total_similarity += 0
    
    return total_similarity / len(all_keys)


def get_tool_predictions(model, question: str) -> List[Dict[str, Any]]:
    """Get tool call predictions from a model for a given question.
    
    Returns list of tool calls: [{"name": str, "args": dict}, ...]
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]
    
    try:
        response = model.invoke(messages)
        
        if response.tool_calls:
            return [
                {
                    "name": tc["name"],
                    "args": tc["args"]
                }
                for tc in response.tool_calls
            ]
        return []
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return []


async def benchmark_example(example_dir: Path, actor_model, spec_model) -> Dict[str, Any]:
    """Benchmark a single example.
    
    Returns:
        {
            "task_id": str,
            "question": str,
            "actor_tools": [...],
            "spec_tools": [...],
            "matches": [{
                "tool_match": bool,
                "args_similarity": float,
                "actor_tool": str,
                "spec_tool": str,
                "actor_args": dict,
                "spec_args": dict
            }],
            "exact_match": bool,
            "avg_args_similarity": float
        }
    """
    # Load metadata
    metadata_path = example_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    task_id = metadata.get("task_id", "unknown")
    question = metadata.get("question", "")
    
    print(f"\n{'='*80}")
    print(f"Benchmarking: {task_id}")
    print(f"{'='*80}")
    print(f"Question: {question[:100]}...")
    
    # Get predictions from both models
    print("\n[1/2] Getting actor predictions...")
    start = time.time()
    actor_tools = get_tool_predictions(actor_model, question)
    actor_time = time.time() - start
    print(f"  Actor: {len(actor_tools)} tool(s) in {actor_time:.2f}s")
    if actor_tools:
        for tc in actor_tools:
            print(f"    - {tc['name']}: {list(tc['args'].keys())}")
    
    print("\n[2/2] Getting speculator predictions...")
    start = time.time()
    spec_tools = get_tool_predictions(spec_model, question)
    spec_time = time.time() - start
    print(f"  Spec:  {len(spec_tools)} tool(s) in {spec_time:.2f}s")
    if spec_tools:
        for tc in spec_tools:
            print(f"    - {tc['name']}: {list(tc['args'].keys())}")
    
    # Compare predictions
    matches = []
    exact_match = False
    
    if actor_tools and spec_tools:
        # Compare first tool (most important)
        actor_tool = actor_tools[0]
        spec_tool = spec_tools[0]
        
        tool_match = actor_tool["name"] == spec_tool["name"]
        args_similarity = calculate_args_similarity(
            actor_tool["args"],
            spec_tool["args"]
        )
        
        matches.append({
            "tool_match": tool_match,
            "args_similarity": args_similarity,
            "actor_tool": actor_tool["name"],
            "spec_tool": spec_tool["name"],
            "actor_args": actor_tool["args"],
            "spec_args": spec_tool["args"]
        })
        
        exact_match = tool_match and args_similarity == 100.0
        
        print(f"\n  Tool match: {'✓' if tool_match else '✗'}")
        print(f"  Args similarity: {args_similarity:.1f}%")
        if exact_match:
            print(f"  Result: ✓ EXACT MATCH")
        elif tool_match:
            print(f"  Result: ≈ PARTIAL MATCH (tool correct, args {args_similarity:.0f}%)")
        else:
            print(f"  Result: ✗ MISMATCH")
    
    avg_args_similarity = sum(m["args_similarity"] for m in matches) / len(matches) if matches else 0.0
    
    return {
        "task_id": task_id,
        "question": question,
        "actor_tools": actor_tools,
        "spec_tools": spec_tools,
        "matches": matches,
        "exact_match": exact_match,
        "avg_args_similarity": avg_args_similarity,
        "actor_time": actor_time,
        "spec_time": spec_time
    }


async def benchmark_level(level: int, max_examples: int = None):
    """Benchmark all examples in a level."""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING SPECULATION - LEVEL {level}")
    print(f"{'='*80}")
    print(f"Actor Model:  {config.actor_model}")
    print(f"Spec Model:   {config.spec_model}")
    print(f"{'='*80}\n")
    
    # Find example directories
    dataset_root = Path("gaia_dataset")
    level_dir = dataset_root / f"level{level}"
    
    if not level_dir.exists():
        print(f"Error: {level_dir} not found. Please run download_gaia.py first.")
        return None
    
    # Get all example directories
    examples = sorted([
        d for d in level_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])
    
    if max_examples:
        examples = examples[:max_examples]
    
    print(f"Found {len(examples)} examples to benchmark")
    
    # Initialize models
    print("\nInitializing models...")
    actor_model = get_actor_model()
    spec_model = get_spec_model()
    print("Models ready!\n")
    
    # Initialize output file
    output_file = f"benchmark_speculation_level{level}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    # Benchmark each example and save incrementally
    results = []
    for i, example_dir in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] Processing {example_dir.name}...")
        
        try:
            result = await benchmark_example(example_dir, actor_model, spec_model)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "task_id": example_dir.name,
                "error": str(e)
            })
        
        # Save results incrementally after each example
        valid_results = [r for r in results if "error" not in r]
        total = len(valid_results)
        exact_matches = sum(1 for r in valid_results if r.get("exact_match", False))
        tool_matches = sum(1 for r in valid_results if r.get("matches") and r["matches"][0]["tool_match"])
        avg_args_sim = sum(r.get("avg_args_similarity", 0) for r in valid_results) / total if total > 0 else 0
        
        stats = {
            "total_examples": len(examples),
            "processed": len(results),
            "valid_results": total,
            "errors": len(results) - total,
            "exact_matches": exact_matches,
            "exact_match_rate": (exact_matches / total * 100) if total > 0 else 0,
            "tool_matches": tool_matches,
            "tool_match_rate": (tool_matches / total * 100) if total > 0 else 0,
            "avg_args_similarity": avg_args_sim,
            "actor_model": config.actor_model,
            "spec_model": config.spec_model,
        }
        
        output = {
            "metadata": stats,
            "results": results
        }
        
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"  [Saved to {output_file}]")
    
    # Print final summary (stats already calculated in last iteration)
    if results:
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"Total Examples:      {total}")
        print(f"Exact Matches:       {exact_matches} ({stats['exact_match_rate']:.1f}%)")
        print(f"Tool Name Matches:   {tool_matches} ({stats['tool_match_rate']:.1f}%)")
        print(f"Avg Args Similarity: {avg_args_sim:.1f}%")
        print(f"\nResults saved to: {output_file}")
        print(f"{'='*80}\n")
        
        return output
    else:
        print("\nNo examples processed.")
        return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark speculation on GAIA dataset")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3],
                       help="GAIA level to benchmark (default: 1)")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Maximum number of examples to benchmark (default: all)")
    
    args = parser.parse_args()
    
    asyncio.run(benchmark_level(args.level, args.max_examples))


if __name__ == "__main__":
    main()

