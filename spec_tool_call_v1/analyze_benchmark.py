#!/usr/bin/env python3
"""Analyze benchmark results and generate report."""
import json
import sys
from pathlib import Path
from collections import Counter


def analyze_benchmark(json_file: str):
    """Analyze benchmark results from JSON file."""
    with open(json_file) as f:
        data = json.load(f)
    
    metadata = data["metadata"]
    results = data["results"]
    
    # Filter valid results early
    valid_results = [r for r in results if "error" not in r]
    
    print("\n" + "="*80)
    print("SPECULATION BENCHMARK ANALYSIS")
    print("="*80)
    print(f"Models: {metadata['actor_model']} (actor) vs {metadata['spec_model']} (spec)")
    print(f"Total Examples: {metadata['total_examples']}")
    print("="*80 + "\n")
    
    # Overall statistics
    print("OVERALL STATISTICS")
    print("-"*80)
    print(f"Exact Matches:       {metadata['exact_matches']:3d} / {metadata['valid_results']} ({metadata['exact_match_rate']:.1f}%)")
    print(f"Tool Name Matches:   {metadata['tool_matches']:3d} / {metadata['valid_results']} ({metadata['tool_match_rate']:.1f}%)")
    print(f"Avg Args Similarity: {metadata['avg_args_similarity']:.1f}%")
    if metadata['errors'] > 0:
        print(f"Errors:              {metadata['errors']}")
    
    # Timing statistics
    if valid_results:
        actor_times = [r.get('actor_time', 0) for r in valid_results if 'actor_time' in r]
        spec_times = [r.get('spec_time', 0) for r in valid_results if 'spec_time' in r]
        
        if actor_times and spec_times:
            avg_actor = sum(actor_times) / len(actor_times)
            avg_spec = sum(spec_times) / len(spec_times)
            speedup = avg_actor / avg_spec if avg_spec > 0 else 1.0
            
            print(f"\nTiming:")
            print(f"  Actor avg:         {avg_actor:6.2f}s per call")
            print(f"  Spec avg:          {avg_spec:6.2f}s per call")
            print(f"  Speedup:           {speedup:.2f}x (spec is faster)")
    print()
    
    # Tool distribution
    if valid_results:
        print("TOOL NAME DISTRIBUTION")
        print("-"*80)
        
        actor_tools = Counter()
        spec_tools = Counter()
        
        for r in valid_results:
            if r.get("actor_tools"):
                actor_tools[r["actor_tools"][0]["name"]] += 1
            if r.get("spec_tools"):
                spec_tools[r["spec_tools"][0]["name"]] += 1
        
        print("\nActor (actual) tool choices:")
        for tool, count in actor_tools.most_common():
            print(f"  {tool:25s} {count:3d} ({count/len(valid_results)*100:.1f}%)")
        
        print("\nSpeculator tool predictions:")
        for tool, count in spec_tools.most_common():
            print(f"  {tool:25s} {count:3d} ({count/len(valid_results)*100:.1f}%)")
        print()
    
    # Match analysis
    print("MATCH ANALYSIS")
    print("-"*80)
    
    exact = []
    partial = []
    tool_match = []
    mismatch = []
    
    for r in valid_results:
        if not r.get("matches"):
            continue
        
        match = r["matches"][0]
        if r["exact_match"]:
            exact.append(r)
        elif match["tool_match"]:
            if match["args_similarity"] >= 50:
                partial.append(r)
            else:
                tool_match.append(r)
        else:
            mismatch.append(r)
    
    print(f"✓ Exact matches:        {len(exact):3d} ({len(exact)/len(valid_results)*100:.1f}%)")
    print(f"≈ Partial matches:      {len(partial):3d} ({len(partial)/len(valid_results)*100:.1f}%) [tool correct, args ≥50%]")
    print(f"~ Tool-only matches:    {len(tool_match):3d} ({len(tool_match)/len(valid_results)*100:.1f}%) [tool correct, args <50%]")
    print(f"✗ Complete mismatches:  {len(mismatch):3d} ({len(mismatch)/len(valid_results)*100:.1f}%)")
    print()
    
    # Detailed mismatch analysis
    if mismatch:
        print("MISMATCH DETAILS")
        print("-"*80)
        
        mismatch_patterns = Counter()
        for r in mismatch:
            match = r["matches"][0]
            pattern = f"{match['spec_tool']} → {match['actor_tool']}"
            mismatch_patterns[pattern] += 1
        
        print("Most common spec → actor mismatches:")
        for pattern, count in mismatch_patterns.most_common(10):
            print(f"  {pattern:50s} {count:3d}x")
        print()
    
    # Web Search Query Analysis
    web_search_tools = ['search_web', 'search_with_content']
    web_search_results = []
    
    for r in valid_results:
        if not r.get("matches"):
            continue
        match = r["matches"][0]
        actor_tool = match.get("actor_tool", "")
        spec_tool = match.get("spec_tool", "")
        
        # Check if either actor or spec used a web search tool
        if actor_tool in web_search_tools or spec_tool in web_search_tools:
            web_search_results.append(r)
    
    if web_search_results:
        print("WEB SEARCH QUERY ANALYSIS")
        print("-"*80)
        
        # Calculate statistics for web search tools only
        web_exact = sum(1 for r in web_search_results if r.get("exact_match"))
        web_tool_match = sum(1 for r in web_search_results if r.get("matches") and r["matches"][0]["tool_match"])
        web_avg_sim = sum(r["matches"][0]["args_similarity"] for r in web_search_results if r.get("matches")) / len(web_search_results)
        
        print(f"Web search tool calls:     {len(web_search_results)}")
        print(f"Tool name matches:         {web_tool_match} / {len(web_search_results)} ({web_tool_match/len(web_search_results)*100:.1f}%)")
        print(f"Avg query similarity:      {web_avg_sim:.1f}%")
        print(f"Exact query matches:       {web_exact} / {len(web_search_results)} ({web_exact/len(web_search_results)*100:.1f}%)")
        
        # Query similarity distribution for web searches
        print("\nQuery similarity distribution:")
        web_sim_ranges = {
            "100% (Exact)": 0,
            " 90-99%": 0,
            " 75-89%": 0,
            " 50-74%": 0,
            " 25-49%": 0,
            "  0-24%": 0,
        }
        
        for r in web_search_results:
            if not r.get("matches"):
                continue
            sim = r["matches"][0]["args_similarity"]
            if sim == 100:
                web_sim_ranges["100% (Exact)"] += 1
            elif sim >= 90:
                web_sim_ranges[" 90-99%"] += 1
            elif sim >= 75:
                web_sim_ranges[" 75-89%"] += 1
            elif sim >= 50:
                web_sim_ranges[" 50-74%"] += 1
            elif sim >= 25:
                web_sim_ranges[" 25-49%"] += 1
            else:
                web_sim_ranges["  0-24%"] += 1
        
        for range_name, count in web_sim_ranges.items():
            if count > 0:
                print(f"  {range_name}: {count:3d} ({count/len(web_search_results)*100:.1f}%)")
        
        # Show examples of query differences
        print("\nExample query comparisons:")
        example_count = 0
        for r in web_search_results:
            if not r.get("matches") or example_count >= 3:
                break
            
            match = r["matches"][0]
            if match["tool_match"] and match["args_similarity"] < 100:
                actor_args = match.get("actor_args", {})
                spec_args = match.get("spec_args", {})
                actor_query = actor_args.get("query", "N/A")
                spec_query = spec_args.get("query", "N/A")
                
                print(f"\n  [{match['actor_tool']}] Similarity: {match['args_similarity']:.1f}%")
                print(f"  Actor:  \"{actor_query[:80]}...\"" if len(actor_query) > 80 else f"  Actor:  \"{actor_query}\"")
                print(f"  Spec:   \"{spec_query[:80]}...\"" if len(spec_query) > 80 else f"  Spec:   \"{spec_query}\"")
                example_count += 1
        
        print()
    
    # Args similarity distribution
    print("ARGS SIMILARITY DISTRIBUTION")
    print("-"*80)
    
    sim_ranges = {
        "100% (Exact)": [],
        " 90-99%": [],
        " 75-89%": [],
        " 50-74%": [],
        " 25-49%": [],
        "  0-24%": [],
    }
    
    for r in valid_results:
        if not r.get("matches"):
            continue
        sim = r["matches"][0]["args_similarity"]
        if sim == 100:
            sim_ranges["100% (Exact)"].append(r)
        elif sim >= 90:
            sim_ranges[" 90-99%"].append(r)
        elif sim >= 75:
            sim_ranges[" 75-89%"].append(r)
        elif sim >= 50:
            sim_ranges[" 50-74%"].append(r)
        elif sim >= 25:
            sim_ranges[" 25-49%"].append(r)
        else:
            sim_ranges["  0-24%"].append(r)
    
    for range_name, items in sim_ranges.items():
        if items:
            print(f"{range_name}: {len(items):3d} ({len(items)/len(valid_results)*100:.1f}%)")
    print()
    
    # Example details
    if len(sys.argv) > 2 and sys.argv[2] == "--verbose":
        print("\nDETAILED RESULTS")
        print("="*80)
        for r in valid_results[:10]:  # Show first 10
            print(f"\nTask: {r['task_id']}")
            print(f"Question: {r['question'][:100]}...")
            if r.get("matches"):
                match = r["matches"][0]
                print(f"Actor:  {match['actor_tool']}")
                print(f"  Args: {match['actor_args']}")
                print(f"Spec:   {match['spec_tool']}")
                print(f"  Args: {match['spec_args']}")
                print(f"Match:  {'✓' if match['tool_match'] else '✗'} tool, {match['args_similarity']:.1f}% args")
    
    print("="*80 + "\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_benchmark.py <benchmark_results.json> [--verbose]")
        print("\nLook for files like: benchmark_speculation_level1_YYYYMMDD_HHMMSS.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    if not Path(json_file).exists():
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    analyze_benchmark(json_file)


if __name__ == "__main__":
    main()

