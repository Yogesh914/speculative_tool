#!/usr/bin/env python3
"""Analyze tool call latencies from baseline GAIA results."""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def print_tool_stats(tool_calls: Dict[str, List[float]], title: str):
    """Print statistics for a set of tool calls."""
    if not tool_calls:
        print(f"\n{title}: No data")
        return
    
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
    # Sort tools by total time (descending)
    tools_by_total_time = sorted(
        tool_calls.items(), 
        key=lambda x: sum(x[1]), 
        reverse=True
    )
    
    print(f"\n{'Tool Name':<20} {'Count':<8} {'Total (s)':<12} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
    print("-" * 80)
    
    for tool_name, latencies in tools_by_total_time:
        count = len(latencies)
        total = sum(latencies)
        avg = total / count if count > 0 else 0
        min_lat = min(latencies) if latencies else 0
        max_lat = max(latencies) if latencies else 0
        
        print(f"{tool_name:<20} {count:<8} {total:<12.3f} {avg:<12.3f} {min_lat:<12.3f} {max_lat:<12.3f}")
    
    # Calculate percentiles for each tool
    print(f"\n{'Tool Name':<20} {'p50':<10} {'p75':<10} {'p90':<10} {'p95':<10} {'p99':<10}")
    print("-" * 80)
    
    for tool_name, latencies in tools_by_total_time:
        if not latencies:
            continue
        
        sorted_lats = sorted(latencies)
        n = len(sorted_lats)
        
        def percentile(p):
            k = (n - 1) * p / 100
            f = int(k)
            c = int(k) + 1
            if c >= n:
                return sorted_lats[-1]
            return sorted_lats[f] + (sorted_lats[c] - sorted_lats[f]) * (k - f)
        
        p50 = percentile(50)
        p75 = percentile(75)
        p90 = percentile(90)
        p95 = percentile(95)
        p99 = percentile(99)
        
        print(f"{tool_name:<20} {p50:<10.3f} {p75:<10.3f} {p90:<10.3f} {p95:<10.3f} {p99:<10.3f}")

def analyze_results(results_dir: str):
    """Analyze all result files in the directory."""
    results_path = Path(results_dir)
    
    # Statistics - split by correctness
    all_tool_calls = defaultdict(list)  # tool_name -> [latencies]
    correct_tool_calls = defaultdict(list)
    incorrect_tool_calls = defaultdict(list)
    
    total_tool_time = 0
    total_llm_time = 0
    num_examples = 0
    num_correct = 0
    num_incorrect = 0
    num_skipped = 0
    
    # Process all JSON files
    json_files = sorted(results_path.glob("*.json"))
    
    print(f"Found {len(json_files)} result files\n")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if trajectory exists
            if 'trajectory' not in data or not data['trajectory']:
                print(f"⚠️  Skipping {json_file.name}: No trajectory data")
                num_skipped += 1
                continue
            
            num_examples += 1
            is_correct = data.get('correct', False)
            
            if is_correct:
                num_correct += 1
            else:
                num_incorrect += 1
            
            # Extract tool calls from trajectory
            for entry in data['trajectory']:
                if entry.get('node') == 'tools':
                    tool_time = entry.get('tool_time', 0)
                    tool_executed = entry.get('tool_executed', {})
                    tool_name = tool_executed.get('name', 'unknown')
                    
                    all_tool_calls[tool_name].append(tool_time)
                    if is_correct:
                        correct_tool_calls[tool_name].append(tool_time)
                    else:
                        incorrect_tool_calls[tool_name].append(tool_time)
                    total_tool_time += tool_time
                
                elif entry.get('node') == 'llm':
                    llm_time = entry.get('llm_time', 0)
                    total_llm_time += llm_time
                        
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")
            num_skipped += 1
            continue
    
    # Print summary statistics
    print("=" * 80)
    print("TOOL LATENCY ANALYSIS - BASELINE RESULTS")
    print("=" * 80)
    print(f"\nDataset: {results_dir}")
    print(f"Total files found: {len(json_files)}")
    print(f"Examples analyzed: {num_examples}")
    print(f"  ✅ Correct: {num_correct}")
    print(f"  ❌ Incorrect: {num_incorrect}")
    print(f"  ⚠️  Skipped: {num_skipped}")
    print(f"\nTotal LLM time: {total_llm_time:.2f}s")
    print(f"Total tool time: {total_tool_time:.2f}s")
    print(f"Tool time as % of LLM time: {(total_tool_time/total_llm_time*100) if total_llm_time > 0 else 0:.1f}%")
    
    # Print tool statistics for all, correct, and incorrect
    print_tool_stats(all_tool_calls, "ALL EXAMPLES - TOOL STATISTICS")
    print_tool_stats(correct_tool_calls, "✅ CORRECT EXAMPLES - TOOL STATISTICS")
    print_tool_stats(incorrect_tool_calls, "❌ INCORRECT EXAMPLES - TOOL STATISTICS")
    
    # Analysis for speculation viability
    print("\n" + "=" * 80)
    print("SPECULATION VIABILITY ANALYSIS (ALL EXAMPLES)")
    print("=" * 80)
    
    print("\nFor speculation to be viable, tool latency should be >> spec model latency (~10-20s)")
    print("\nTool categories:")
    
    tools_by_total_time = sorted(
        all_tool_calls.items(), 
        key=lambda x: sum(x[1]), 
        reverse=True
    )
    
    for tool_name, latencies in tools_by_total_time:
        avg = sum(latencies) / len(latencies) if latencies else 0
        
        if avg < 1:
            category = "❌ TOO FAST (< 1s) - Speculation will add overhead"
        elif avg < 5:
            category = "⚠️  MARGINAL (1-5s) - Speculation unlikely to help"
        elif avg < 10:
            category = "🤔 MAYBE (5-10s) - Speculation might break even"
        else:
            category = "✅ VIABLE (> 10s) - Speculation can save time"
        
        print(f"  {tool_name:<20} avg={avg:>6.3f}s  {category}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    # Count tools in each category
    too_fast = sum(1 for _, lats in all_tool_calls.items() if sum(lats)/len(lats) < 1)
    marginal = sum(1 for _, lats in all_tool_calls.items() if 1 <= sum(lats)/len(lats) < 5)
    maybe = sum(1 for _, lats in all_tool_calls.items() if 5 <= sum(lats)/len(lats) < 10)
    viable = sum(1 for _, lats in all_tool_calls.items() if sum(lats)/len(lats) >= 10)
    
    print(f"\nTool categories:")
    print(f"  Too fast (< 1s):     {too_fast} tools")
    print(f"  Marginal (1-5s):     {marginal} tools")
    print(f"  Maybe (5-10s):       {maybe} tools")
    print(f"  Viable (> 10s):      {viable} tools")
    
    if viable == 0 and maybe == 0:
        print("\n⚠️  WARNING: No tools have latency suitable for speculation!")
        print("    With spec model at 10-20s and tools at < 5s, speculation adds overhead.")
        print("    Recommendation: GAIA is not suitable for speculative execution.")
    elif viable > 0:
        print(f"\n✅ {viable} tool(s) may benefit from speculation")
        
        # Show which tool(s) are viable
        viable_tools = [name for name, lats in all_tool_calls.items() if sum(lats)/len(lats) >= 10]
        print(f"    Viable tools: {', '.join(viable_tools)}")
        
        # Count how many calls to viable tools
        viable_calls = sum(len(lats) for name, lats in all_tool_calls.items() if sum(lats)/len(lats) >= 10)
        total_calls = sum(len(lats) for lats in all_tool_calls.values())
        print(f"    {viable_calls}/{total_calls} tool calls ({viable_calls/total_calls*100:.1f}%) use viable tools")
    else:
        print(f"\n⚠️  Only {maybe} tool(s) in marginal range - speculation break-even at best")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "results_level1_baseline_20251203_002006"
    
    analyze_results(results_dir)

