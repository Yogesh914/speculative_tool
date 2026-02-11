#!/usr/bin/env python3
"""Compare two evaluation result directories."""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_results(result_dir: Path) -> Dict[str, dict]:
    """Load all result JSON files from a directory.
    
    Returns:
        Dict mapping task_id to result data (with added 'example_name' field)
    """
    results = {}
    
    if not result_dir.exists():
        print(f"Error: {result_dir} not found")
        return results
    
    for json_file in result_dir.glob("result_*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                task_id = data.get("task_id")
                
                # Extract example name from filename: result_{example_name}_{spec/baseline}.json
                filename = json_file.stem  # e.g., "result_example_007_baseline"
                parts = filename.split("_")
                # Reconstruct example name (handle case where it might have underscores)
                # Pattern: result_{example_name}_{status}
                # So we need to find where the status starts (last underscore before spec/baseline)
                if parts[-1] in ["spec", "baseline"]:
                    example_name = "_".join(parts[1:-1])  # Skip 'result' and 'spec/baseline'
                else:
                    example_name = "_".join(parts[1:])  # Fallback
                
                if task_id:
                    data['example_name'] = example_name
                    results[task_id] = data
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return results


def compare_results(baseline_dir: str, spec_dir: str, only_speedups: bool = False):
    """Compare baseline and speculation results."""
    
    baseline_path = Path(baseline_dir)
    spec_path = Path(spec_dir)
    
    print("=" * 80)
    print("RESULT COMPARISON")
    print("=" * 80)
    print(f"Baseline: {baseline_dir}")
    print(f"Spec:     {spec_dir}")
    print("=" * 80 + "\n")
    
    # Load results
    baseline_results = load_results(baseline_path)
    spec_results = load_results(spec_path)
    
    print(f"Loaded {len(baseline_results)} baseline results")
    print(f"Loaded {len(spec_results)} speculation results\n")
    
    # Find common task IDs
    common_ids = set(baseline_results.keys()) & set(spec_results.keys())
    
    if not common_ids:
        print("Error: No common task IDs found")
        return
    
    print(f"Found {len(common_ids)} examples in common\n")
    
    # Compare each example
    comparisons = []
    
    for task_id in sorted(common_ids):
        baseline = baseline_results[task_id]
        spec = spec_results[task_id]
        
        comparison = {
            "task_id": task_id,
            "example_name": baseline.get("example_name", task_id[:10]),  # Use example name or fallback to truncated ID
            "question": baseline.get("question", "")[:80] + "...",
            "baseline_correct": baseline.get("correct", False),
            "spec_correct": spec.get("correct", False),
            "baseline_time": baseline.get("time_seconds", 0),
            "spec_time": spec.get("time_seconds", 0),
            "baseline_steps": baseline.get("steps", 0),
            "spec_steps": spec.get("steps", 0),
            "spec_hits": spec.get("spec_hits", 0),
            "spec_misses": spec.get("spec_misses", 0),
        }
        
        # Calculate speedup
        if comparison["baseline_time"] > 0:
            comparison["speedup"] = (comparison["baseline_time"] - comparison["spec_time"]) / comparison["baseline_time"] * 100
            comparison["time_saved"] = comparison["baseline_time"] - comparison["spec_time"]
        else:
            comparison["speedup"] = 0
            comparison["time_saved"] = 0
        
        comparisons.append(comparison)
    
    # Filter for speedups if requested
    if only_speedups:
        comparisons = [c for c in comparisons if c["speedup"] > 0]
        if not comparisons:
            print("No examples with speedups found!")
            return
        print(f"Filtered to {len(comparisons)} examples with speedups\n")
    
    # Print per-example comparison
    print("=" * 80)
    print("PER-EXAMPLE COMPARISON")
    print("=" * 80)
    print(f"{'Example':<20} {'Baseline':<15} {'Spec':<15} {'Speedup':<10} {'Steps':<12} {'Hits/Total'}")
    print("-" * 97)
    
    for c in comparisons:
        baseline_status = "✓" if c["baseline_correct"] else "✗"
        spec_status = "✓" if c["spec_correct"] else "✗"
        
        baseline_info = f"{baseline_status} {c['baseline_time']:.1f}s"
        spec_info = f"{spec_status} {c['spec_time']:.1f}s"
        speedup = f"{c['speedup']:+.1f}%"
        steps_info = f"{c['baseline_steps']} → {c['spec_steps']}"
        
        total_checks = c["spec_hits"] + c["spec_misses"]
        hit_rate = f"{c['spec_hits']}/{total_checks}" if total_checks > 0 else "N/A"
        
        # Use full example_name for display (no truncation)
        display_name = c['example_name']
        print(f"{display_name:<20} {baseline_info:<15} {spec_info:<15} {speedup:<10} {steps_info:<12} {hit_rate}")
    
    # Calculate aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)
    
    baseline_correct = sum(1 for c in comparisons if c["baseline_correct"])
    spec_correct = sum(1 for c in comparisons if c["spec_correct"])
    total = len(comparisons)
    
    baseline_accuracy = baseline_correct / total * 100
    spec_accuracy = spec_correct / total * 100
    accuracy_diff = spec_accuracy - baseline_accuracy
    
    baseline_avg_time = sum(c["baseline_time"] for c in comparisons) / total
    spec_avg_time = sum(c["spec_time"] for c in comparisons) / total
    avg_speedup = (baseline_avg_time - spec_avg_time) / baseline_avg_time * 100
    
    baseline_total_time = sum(c["baseline_time"] for c in comparisons)
    spec_total_time = sum(c["spec_time"] for c in comparisons)
    total_time_saved = baseline_total_time - spec_total_time
    
    baseline_avg_steps = sum(c["baseline_steps"] for c in comparisons) / total
    spec_avg_steps = sum(c["spec_steps"] for c in comparisons) / total
    steps_diff = spec_avg_steps - baseline_avg_steps
    
    total_hits = sum(c["spec_hits"] for c in comparisons)
    total_misses = sum(c["spec_misses"] for c in comparisons)
    total_checks = total_hits + total_misses
    overall_hit_rate = total_hits / total_checks * 100 if total_checks > 0 else 0
    
    print(f"\n📊 Accuracy:")
    print(f"  Baseline:  {baseline_correct}/{total} ({baseline_accuracy:.1f}%)")
    print(f"  Spec:      {spec_correct}/{total} ({spec_accuracy:.1f}%)")
    print(f"  Difference: {accuracy_diff:+.1f}%")
    
    print(f"\n⏱️  Latency:")
    print(f"  Baseline avg:  {baseline_avg_time:.1f}s")
    print(f"  Spec avg:      {spec_avg_time:.1f}s")
    print(f"  Speedup:       {avg_speedup:+.1f}%")
    print(f"  Time saved:    {total_time_saved:.1f}s total ({total_time_saved/total:.1f}s per example)")
    
    print(f"\n🔢 Steps:")
    print(f"  Baseline avg:  {baseline_avg_steps:.1f}")
    print(f"  Spec avg:      {spec_avg_steps:.1f}")
    print(f"  Difference:    {steps_diff:+.1f}")
    
    print(f"\n💾 Cache Performance:")
    print(f"  Hits:      {total_hits}")
    print(f"  Misses:    {total_misses}")
    print(f"  Hit rate:  {overall_hit_rate:.1f}%")
    
    print("\n" + "=" * 80)
    
    # Determine if spec is better
    if spec_accuracy >= baseline_accuracy and avg_speedup > 0:
        print("✅ SPECULATION WINS: Same or better accuracy with lower latency")
    elif spec_accuracy >= baseline_accuracy:
        print("⚠️  MIXED: Same or better accuracy but higher latency")
    elif avg_speedup > 0:
        print("⚠️  MIXED: Lower latency but worse accuracy")
    else:
        print("❌ BASELINE WINS: Better accuracy and latency")
    
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare baseline and speculation results")
    parser.add_argument("baseline_dir", help="Baseline results directory")
    parser.add_argument("spec_dir", help="Speculation results directory")
    parser.add_argument("--only-speedups", action="store_true", help="Show only examples with speedups")
    
    args = parser.parse_args()
    
    compare_results(args.baseline_dir, args.spec_dir, args.only_speedups)


if __name__ == "__main__":
    main()

