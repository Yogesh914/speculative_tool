#!/usr/bin/env python3
"""
Compare Baseline vs Speculation Results

Analyzes and compares results from baseline and speculation experiments
to quantify the benefits of speculative tool calling.

Usage:
    python compare_results.py results/baseline_hotpot_*.json results/speculation_hotpot_*.json
"""

import argparse
import json
from pathlib import Path
from rich import print as rprint
from rich.table import Table
from rich.console import Console

console = Console()


def load_results(filepath: str) -> dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_experiments(baseline_file: str, speculation_file: str):
    """
    Compare baseline and speculation experiments.

    Args:
        baseline_file: Path to baseline results JSON
        speculation_file: Path to speculation results JSON
    """
    rprint("\n" + "="*80)
    rprint("[bold cyan]BASELINE VS SPECULATION COMPARISON[/bold cyan]")
    rprint("="*80)

    # Load both results
    baseline = load_results(baseline_file)
    speculation = load_results(speculation_file)

    # Extract metadata
    baseline_meta = baseline.get("metadata", {})
    speculation_meta = speculation.get("metadata", {})

    rprint(f"\n📋 Experiment Details:")
    rprint(f"  Baseline: {baseline_file}")
    rprint(f"  Speculation: {speculation_file}")
    rprint(f"  Dataset: {baseline_meta.get('dataset', 'N/A')}")
    rprint(f"  Actor Model: {baseline_meta.get('actor_model', 'N/A')}")
    rprint(f"  Spec Model: {speculation_meta.get('spec_model', 'N/A')}")

    # Extract summaries
    baseline_summary = baseline.get("summary", {})
    speculation_summary = speculation.get("summary", {})

    if not baseline_summary or not speculation_summary:
        rprint("\n[yellow]⚠️  Warning: Summary data not found in one or both files[/yellow]")
        return

    # Create comparison table
    table = Table(title="Performance Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=40)
    table.add_column("Baseline", style="yellow", justify="right", width=15)
    table.add_column("Speculation", style="green", justify="right", width=15)
    table.add_column("Improvement", style="bold", justify="right", width=15)

    # Overall metrics
    baseline_queries = baseline_summary.get("total_queries", 0)
    speculation_queries = speculation_summary.get("total_queries", 0)
    table.add_row("Total Queries", str(baseline_queries), str(speculation_queries), "-")

    baseline_success = baseline_summary.get("success_rate", 0)
    speculation_success = speculation_summary.get("success_rate", 0)
    table.add_row(
        "Success Rate",
        f"{baseline_success:.1f}%",
        f"{speculation_success:.1f}%",
        f"{speculation_success - baseline_success:+.1f}%"
    )

    baseline_accuracy = baseline_summary.get("accuracy", 0)
    speculation_accuracy = speculation_summary.get("accuracy", 0)
    table.add_row(
        "Accuracy",
        f"{baseline_accuracy:.1f}%",
        f"{speculation_accuracy:.1f}%",
        f"{speculation_accuracy - baseline_accuracy:+.1f}%"
    )

    table.add_row("", "", "", "")  # Separator

    # Time metrics
    time_metrics_b = baseline_summary.get("time_metrics", {})
    time_metrics_s = speculation_summary.get("time_metrics", {})

    baseline_avg_time = time_metrics_b.get("avg_time_per_query", 0)
    speculation_avg_time = time_metrics_s.get("avg_time_per_query", 0)
    time_saved = baseline_avg_time - speculation_avg_time
    time_saved_pct = (time_saved / baseline_avg_time * 100) if baseline_avg_time > 0 else 0

    table.add_row(
        "Avg Time per Query",
        f"{baseline_avg_time:.2f}s",
        f"{speculation_avg_time:.2f}s",
        f"{time_saved:+.2f}s ({time_saved_pct:+.1f}%)"
    )

    baseline_total_time = time_metrics_b.get("total_time", 0)
    speculation_total_time = time_metrics_s.get("total_time", 0)
    table.add_row(
        "Total Time",
        f"{baseline_total_time:.2f}s",
        f"{speculation_total_time:.2f}s",
        f"{baseline_total_time - speculation_total_time:+.2f}s"
    )

    table.add_row("", "", "", "")  # Separator

    # Speculation-specific metrics
    cache_metrics = speculation_summary.get("cache_metrics", {})
    table.add_row(
        "Cache Hit Rate",
        "N/A",
        f"{cache_metrics.get('hit_rate', 0):.1f}%",
        "✨ Speculation"
    )

    table.add_row(
        "Cache Hits",
        "N/A",
        str(cache_metrics.get('total_hits', 0)),
        "✨ Speculation"
    )

    table.add_row(
        "Cache Misses",
        "N/A",
        str(cache_metrics.get('total_misses', 0)),
        "✨ Speculation"
    )

    table.add_row("", "", "", "")  # Separator

    # Verification metrics
    verif_metrics = speculation_summary.get("verification_metrics", {})
    table.add_row(
        "Strict AST Verification",
        "N/A",
        f"{verif_metrics.get('strict_only_rate', 0):.1f}%",
        "✨ Speculation"
    )

    table.add_row(
        "With Embedding Fallback",
        "N/A",
        f"{verif_metrics.get('with_fallback_rate', 0):.1f}%",
        f"+{verif_metrics.get('fallback_improvement', 0):.1f}%"
    )

    console.print("\n")
    console.print(table)

    # Key insights
    rprint("\n📊 Key Insights:")

    if time_saved > 0:
        rprint(f"  ✅ Speculation saved {time_saved:.2f}s per query ({time_saved_pct:.1f}% faster)")
    elif time_saved < 0:
        rprint(f"  ⚠️  Speculation was {abs(time_saved):.2f}s slower per query")
    else:
        rprint(f"  ➖ No significant time difference")

    hit_rate = cache_metrics.get('hit_rate', 0)
    if hit_rate > 50:
        rprint(f"  ✅ High cache hit rate ({hit_rate:.1f}%) - spec model predicting well")
    elif hit_rate > 25:
        rprint(f"  ⚠️  Moderate cache hit rate ({hit_rate:.1f}%) - room for improvement")
    else:
        rprint(f"  ❌ Low cache hit rate ({hit_rate:.1f}%) - spec model struggling")

    fallback_improvement = verif_metrics.get('fallback_improvement', 0)
    if fallback_improvement > 20:
        rprint(f"  ✅ Embedding fallback highly effective (+{fallback_improvement:.1f}%)")
    elif fallback_improvement > 10:
        rprint(f"  ✓ Embedding fallback moderately helpful (+{fallback_improvement:.1f}%)")
    elif fallback_improvement > 0:
        rprint(f"  ➖ Embedding fallback slightly helpful (+{fallback_improvement:.1f}%)")

    # Per-query analysis
    rprint("\n📈 Per-Query Analysis:")

    baseline_queries_data = baseline.get("queries", [])
    speculation_queries_data = speculation.get("queries", [])

    if len(baseline_queries_data) == len(speculation_queries_data):
        faster_count = 0
        slower_count = 0
        same_count = 0

        for b_query, s_query in zip(baseline_queries_data, speculation_queries_data):
            b_time = b_query.get("execution", {}).get("total_time", 0)
            s_time = s_query.get("execution", {}).get("total_time", 0)

            if s_time < b_time * 0.95:  # At least 5% faster
                faster_count += 1
            elif s_time > b_time * 1.05:  # At least 5% slower
                slower_count += 1
            else:
                same_count += 1

        total = len(baseline_queries_data)
        rprint(f"  Faster: {faster_count}/{total} ({faster_count/total*100:.1f}%)")
        rprint(f"  Slower: {slower_count}/{total} ({slower_count/total*100:.1f}%)")
        rprint(f"  Similar: {same_count}/{total} ({same_count/total*100:.1f}%)")

    rprint("\n" + "="*80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare baseline and speculation experiment results"
    )
    parser.add_argument(
        "baseline",
        type=str,
        help="Path to baseline results JSON file"
    )
    parser.add_argument(
        "speculation",
        type=str,
        help="Path to speculation results JSON file"
    )

    args = parser.parse_args()

    # Check files exist
    if not Path(args.baseline).exists():
        rprint(f"[red]❌ Error: Baseline file not found: {args.baseline}[/red]")
        return

    if not Path(args.speculation).exists():
        rprint(f"[red]❌ Error: Speculation file not found: {args.speculation}[/red]")
        return

    # Run comparison
    compare_experiments(args.baseline, args.speculation)


if __name__ == "__main__":
    main()
