"""
Results Saver

Saves comprehensive experiment results to JSON files with detailed metrics,
tool usage statistics, and per-query verification information.

Compatible with analysis scripts for comparing baseline vs speculation performance.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .enhanced_metrics import ExperimentMetrics


class ResultsSaver:
    """Manages saving and loading experiment results."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize results saver.

        Args:
            output_dir: Directory to save results (default: "results/")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_single_result(
        self,
        metrics: ExperimentMetrics,
        filename: Optional[str] = None
    ) -> str:
        """
        Save a single experiment result.

        Args:
            metrics: ExperimentMetrics instance to save
            filename: Optional custom filename (default: auto-generated)

        Returns:
            Path to saved file
        """
        if filename is None:
            # Auto-generate filename: experiment_id_timestamp.json
            filename = f"{metrics.experiment_id}_{metrics.timestamp}.json"

        filepath = self.output_dir / filename

        # Convert to dict and save
        data = metrics.to_dict()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Saved result to: {filepath}")
        return str(filepath)

    def save_batch_results(
        self,
        metrics_list: List[ExperimentMetrics],
        experiment_name: str,
        include_summary: bool = True,
        filename: Optional[str] = None
    ) -> str:
        """
        Save multiple experiment results in a single file.

        Args:
            metrics_list: List of ExperimentMetrics instances
            experiment_name: Name for this batch of experiments
            include_summary: If True, include aggregate summary statistics

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename:
            filepath = Path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)
        else:
            auto_name = f"{experiment_name}_{timestamp}.json"
            filepath = self.output_dir / auto_name

        # Convert all metrics to dicts
        queries = [m.to_dict() for m in metrics_list]

        # Build output structure
        output = {
            "metadata": {
                "experiment_name": experiment_name,
                "timestamp": timestamp,
                "total_queries": len(metrics_list),
                "dataset": metrics_list[0].dataset if metrics_list else "unknown",
                "actor_model": metrics_list[0].actor_model if metrics_list else "unknown",
                "spec_model": metrics_list[0].spec_model if metrics_list else "unknown",
                "actor_provider": metrics_list[0].actor_provider if metrics_list else "unknown",
                "spec_provider": metrics_list[0].spec_provider if metrics_list else "unknown",
                "speculation_enabled": metrics_list[0].speculation_enabled if metrics_list else False,
            },
            "queries": queries,
        }

        # Add summary statistics if requested
        if include_summary and metrics_list:
            output["summary"] = self._calculate_summary(metrics_list)

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"💾 Saved {len(metrics_list)} results to: {filepath}")
        return str(filepath)

    def _calculate_summary(self, metrics_list: List[ExperimentMetrics]) -> Dict[str, Any]:
        """Calculate aggregate summary statistics across all experiments."""
        total_queries = len(metrics_list)

        # Aggregate metrics
        total_cache_hits = sum(m.total_cache_hits for m in metrics_list)
        total_cache_misses = sum(m.total_cache_misses for m in metrics_list)
        total_ast_verified = sum(m.total_verified_by_ast for m in metrics_list)
        total_embedding_verified = sum(m.total_verified_by_embedding for m in metrics_list)
        total_both_failed = sum(m.total_both_failed for m in metrics_list)

        # Calculate rates
        total_target_tools = total_cache_hits + total_cache_misses
        cache_hit_rate = (total_cache_hits / total_target_tools * 100) if total_target_tools > 0 else 0.0

        total_verified = total_ast_verified + total_embedding_verified
        total_attempts = total_verified + total_both_failed

        strict_only_rate = (total_ast_verified / total_attempts * 100) if total_attempts > 0 else 0.0
        with_fallback_rate = (total_verified / total_attempts * 100) if total_attempts > 0 else 0.0
        fallback_improvement = with_fallback_rate - strict_only_rate

        # Time statistics
        total_time = sum(m.total_time for m in metrics_list)
        avg_time = total_time / total_queries if total_queries > 0 else 0.0

        total_draft_time = sum(m.total_draft_plan_time + m.total_draft_exec_time for m in metrics_list)
        total_target_time = sum(m.total_target_plan_time + m.total_target_exec_time for m in metrics_list)
        total_verification_time = sum(m.total_verification_time for m in metrics_list)

        # Success rate
        successful = sum(1 for m in metrics_list if m.success)
        success_rate = (successful / total_queries * 100) if total_queries > 0 else 0.0

        # Correctness judged by LLM (if available)
        judged = [m for m in metrics_list if m.judge_correct is not None]
        judged_correct = sum(1 for m in judged if m.judge_correct)
        accuracy = (judged_correct / len(judged) * 100) if judged else 0.0

        return {
            "total_queries": total_queries,
            "successful_queries": successful,
            "success_rate": success_rate,
            "accuracy": accuracy,
            "cache_metrics": {
                "total_hits": total_cache_hits,
                "total_misses": total_cache_misses,
                "total_target_tools": total_target_tools,
                "hit_rate": cache_hit_rate,
            },
            "verification_metrics": {
                "strict_ast_verified": total_ast_verified,
                "embedding_fallback_verified": total_embedding_verified,
                "both_failed": total_both_failed,
                "total_attempts": total_attempts,
                "strict_only_rate": strict_only_rate,
                "with_fallback_rate": with_fallback_rate,
                "fallback_improvement": fallback_improvement,
            },
            "time_metrics": {
                "total_time": total_time,
                "avg_time_per_query": avg_time,
                "total_draft_time": total_draft_time,
                "total_target_time": total_target_time,
                "total_verification_time": total_verification_time,
                "avg_verification_time": total_verification_time / total_queries if total_queries > 0 else 0.0,
            },
            "accuracy_metrics": {
                "judged_queries": len(judged),
                "judged_correct": judged_correct,
            },
        }

    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load results from a JSON file.

        Args:
            filepath: Path to results file

        Returns:
            Dictionary containing results data
        """
        with open(filepath, 'r') as f:
            return json.load(f)

    def print_summary_from_file(self, filepath: str) -> None:
        """
        Print a summary from a saved results file.

        Args:
            filepath: Path to results file
        """
        data = self.load_results(filepath)

        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)

        # Print metadata
        if "metadata" in data:
            meta = data["metadata"]
            print(f"\n📋 Experiment:")
            print(f"  Name: {meta.get('experiment_name', 'N/A')}")
            print(f"  Timestamp: {meta.get('timestamp', 'N/A')}")
            print(f"  Total Queries: {meta.get('total_queries', 0)}")
            print(f"  Dataset: {meta.get('dataset', 'N/A')}")
            print(f"  Actor Model: {meta.get('actor_model', 'N/A')}")
            print(f"  Spec Model: {meta.get('spec_model', 'N/A')}")
            print(f"  Speculation: {'ENABLED' if meta.get('speculation_enabled') else 'DISABLED'}")

        # Print summary if available
        if "summary" in data:
            summary = data["summary"]

            print(f"\n📊 Performance:")
            print(f"  Success Rate: {summary.get('success_rate', 0):.1f}%")
            print(f"  Accuracy: {summary.get('accuracy', 0):.1f}%")

            if "cache_metrics" in summary:
                cache = summary["cache_metrics"]
                print(f"\n🎯 Cache:")
                print(f"  Hit Rate: {cache.get('hit_rate', 0):.1f}%")
                print(f"  Hits: {cache.get('total_hits', 0)}")
                print(f"  Misses: {cache.get('total_misses', 0)}")

            if "verification_metrics" in summary:
                verif = summary["verification_metrics"]
                print(f"\n✓ Verification:")
                print(f"  Strict Only: {verif.get('strict_only_rate', 0):.1f}%")
                print(f"  With Fallback: {verif.get('with_fallback_rate', 0):.1f}%")
                print(f"  Improvement: +{verif.get('fallback_improvement', 0):.1f}%")

            if "time_metrics" in summary:
                time_m = summary["time_metrics"]
                print(f"\n⏱️  Time:")
                print(f"  Total: {time_m.get('total_time', 0):.2f}s")
                print(f"  Avg per Query: {time_m.get('avg_time_per_query', 0):.2f}s")
                print(f"  Avg Verification: {time_m.get('avg_verification_time', 0):.3f}s")

        print("\n" + "="*80)


# ==================== Convenience Functions ====================

def save_experiment_results(
    metrics_list: List[ExperimentMetrics],
    experiment_name: str,
    output_dir: str = "results",
    filename: Optional[str] = None
) -> str:
    """
    Convenience function to save experiment results.

    Args:
        metrics_list: List of ExperimentMetrics instances
        experiment_name: Name for this experiment
        output_dir: Directory to save results

    Returns:
        Path to saved file
    """
    saver = ResultsSaver(output_dir=output_dir)
    return saver.save_batch_results(metrics_list, experiment_name, filename=filename)


def load_and_compare_results(
    baseline_file: str,
    speculation_file: str
) -> Dict[str, Any]:
    """
    Load and compare baseline vs speculation results.

    Args:
        baseline_file: Path to baseline results
        speculation_file: Path to speculation results

    Returns:
        Comparison statistics
    """
    saver = ResultsSaver()

    baseline = saver.load_results(baseline_file)
    speculation = saver.load_results(speculation_file)

    # Extract summaries
    baseline_summary = baseline.get("summary", {})
    speculation_summary = speculation.get("summary", {})

    # Calculate improvements
    time_improvement = (
        baseline_summary.get("time_metrics", {}).get("avg_time_per_query", 0) -
        speculation_summary.get("time_metrics", {}).get("avg_time_per_query", 0)
    )

    return {
        "baseline": baseline_summary,
        "speculation": speculation_summary,
        "improvements": {
            "time_saved_per_query": time_improvement,
            "cache_hit_rate": speculation_summary.get("cache_metrics", {}).get("hit_rate", 0),
            "verification_improvement": speculation_summary.get("verification_metrics", {}).get("fallback_improvement", 0),
        }
    }


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RESULTS SAVER TEST")
    print("="*80)

    # Create test metrics
    from enhanced_metrics import create_experiment_metrics, StepMetrics

    metrics = create_experiment_metrics(
        experiment_id="test_001",
        query="What is 2 + 2?",
        ground_truth="4"
    )
    metrics.model_output = "4"
    metrics.predicted_answer = metrics.model_output
    metrics.judge_decision = "CORRECT"
    metrics.judge_correct = True
    metrics.total_steps = 2
    metrics.total_time = 5.5
    metrics.success = True
    metrics.total_cache_hits = 2
    metrics.total_cache_misses = 1

    # Save single result
    saver = ResultsSaver(output_dir="test_results")
    filepath = saver.save_single_result(metrics)

    # Load and print
    print("\n" + "="*80)
    print("LOADING SAVED RESULT")
    print("="*80)
    saver.print_summary_from_file(filepath)

    print("\n✅ TEST COMPLETE")
