#!/usr/bin/env python3
"""
Run Speculation Experiments

Runs experiments with speculation enabled (draft + target models in parallel)
to measure performance improvements, cache hit rates, and verification metrics.

Usage:
    python run_speculation.py --dataset hotpot --num-queries 10
    python run_speculation.py --dataset gaia --level 1 --num-queries 5
"""

import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from rich import print as rprint
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from spec_tool_call import (
    build_speculation_graph,
    GAIADataset,
    HotPotQADataset,
    config,
    Msg,
)
from spec_tool_call.speculation_state import SpeculationState
from spec_tool_call.enhanced_metrics import create_experiment_metrics, StepMetrics
from spec_tool_call.results_saver import save_experiment_results


def _get_last_assistant_message(messages):
    """Return the content of the most recent assistant message."""
    for msg in reversed(messages or []):
        if getattr(msg, "role", None) == "assistant":
            return msg.content
    return ""


def _get_state_attr(state, attr, default=None):
    """Safely access attributes whether LangGraph returns dicts or dataclasses."""
    if isinstance(state, dict):
        return state.get(attr, default)
    return getattr(state, attr, default)


async def run_speculation_experiment(
    dataset_name: str,
    num_queries: int = 10,
    level: str = None,
    results_file: str = None,
):
    """
    Run speculation experiments with draft and target models.

    Args:
        dataset_name: "gaia" or "hotpot"
        num_queries: Number of queries to run
        level: Dataset level filter (optional)
    """
    # Ensure speculation is enabled
    import os
    os.environ["DISABLE_SPECULATION"] = "0"

    # Reload config
    from spec_tool_call.config import SpecConfig
    spec_config = SpecConfig.from_env()

    rprint("\n" + "="*80)
    rprint("[bold cyan]SPECULATION EXPERIMENT (Draft + Target Models)[/bold cyan]")
    rprint("="*80)
    spec_config.print_config()

    # Load dataset
    if dataset_name == "hotpot":
        dataset = HotPotQADataset()
        dataset.load(max_examples=num_queries, random_seed=spec_config.dataset_random_seed)
        if level:
            examples = dataset.get_by_level(level)
        else:
            examples = dataset.get_all()
    else:  # gaia
        dataset = GAIADataset()
        dataset.load()
        if level:
            examples = dataset.get_level(level)
        else:
            examples = dataset.get_all()
        examples = examples[:num_queries]

    query_count = len(examples)
    verbose_per_query = query_count <= 10
    config.verbose_logging = verbose_per_query

    rprint(f"\n📊 Running speculation on {query_count} queries")

    # Build speculation graph
    app = build_speculation_graph()

    # Track metrics for all queries
    all_metrics = []
    progress_bar = tqdm(range(query_count), desc="Queries", unit="query")

    for i in progress_bar:
        idx = i + 1
        example = examples[i]

        if verbose_per_query:
            rprint(f"\n{'='*80}")
            rprint(f"[bold]Query {idx}/{query_count}[/bold]")
            rprint(f"{'='*80}")

        # Get question and ground truth
        if hasattr(example, 'question'):
            question = example.question
            ground_truth = example.answer if hasattr(example, 'answer') else example.final_answer
            example_id = example.id if hasattr(example, 'id') else example.task_id
        else:
            question = str(example)
            ground_truth = None
            example_id = f"query_{idx}"

        if verbose_per_query:
            rprint(f"Question: {question[:100]}...")

        # Create metrics tracker
        metrics = create_experiment_metrics(
            experiment_id=example_id,
            query=question,
            ground_truth=ground_truth
        )

        # Initialize state
        init_state = SpeculationState(messages=[Msg(role="user", content=question)])

        # Run graph
        import time
        start_time = time.time()
        final_state = None

        try:
            async for event in app.astream(
                init_state,
                config={"configurable": {"thread_id": f"speculation-{example_id}"}}
            ):
                for node_name, state in event.items():
                    final_state = state

            # Record results from speculation state
            if final_state:
                metrics.total_time = time.time() - start_time
                metrics.total_steps = int(_get_state_attr(final_state, "step", 0))

                final_messages = _get_state_attr(final_state, "messages", [])
                model_output = _get_last_assistant_message(final_messages)
                fallback_answer = _get_state_attr(final_state, "answer", "") or ""
                if not model_output and fallback_answer:
                    model_output = fallback_answer

                metrics.model_output = model_output
                metrics.predicted_answer = model_output
                metrics.success = bool(model_output)

                # Copy speculation-specific metrics
                cache_hits = _get_state_attr(final_state, "cache_hits", 0)
                cache_misses = _get_state_attr(final_state, "cache_misses", 0)
                draft_tools_launched = _get_state_attr(final_state, "draft_tools_launched", 0)
                verified_by_ast = _get_state_attr(final_state, "verified_by_ast", 0)
                verified_by_embedding = _get_state_attr(final_state, "verified_by_embedding", 0)
                both_failed = _get_state_attr(final_state, "both_failed", 0)
                draft_tool_names = list(_get_state_attr(final_state, "draft_tool_names", []))
                target_tool_names = list(_get_state_attr(final_state, "target_tool_names", []))
                draft_plan_time = _get_state_attr(final_state, "draft_plan_time", 0.0)
                draft_exec_time = _get_state_attr(final_state, "draft_exec_time", 0.0)
                target_plan_time = _get_state_attr(final_state, "target_plan_time", 0.0)
                verify_time = _get_state_attr(final_state, "verify_time", 0.0)
                verification_details = list(_get_state_attr(final_state, "verification_details", []))

                metrics.total_cache_hits = cache_hits
                metrics.total_cache_misses = cache_misses
                metrics.total_draft_tools_launched = draft_tools_launched
                metrics.total_verified_by_ast = verified_by_ast
                metrics.total_verified_by_embedding = verified_by_embedding
                metrics.total_both_failed = both_failed
                metrics.draft_tool_names = draft_tool_names
                metrics.target_tool_names = target_tool_names
                metrics.total_draft_plan_time = draft_plan_time
                metrics.total_draft_exec_time = draft_exec_time
                metrics.total_target_plan_time = target_plan_time
                metrics.total_verification_time = verify_time

                # Create step metrics
                step = StepMetrics(
                    step_number=metrics.total_steps,
                    draft_plan_time=draft_plan_time,
                    draft_exec_time=draft_exec_time,
                    target_plan_time=target_plan_time,
                    verification_time=verify_time,
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                    verified_by_ast=verified_by_ast,
                    verified_by_embedding=verified_by_embedding,
                    verification_details=verification_details,
                )
                metrics.add_step(step)

                # Print results
                rates = metrics.calculate_rates()
                if verbose_per_query:
                    rprint(f"\n✅ Completed in {metrics.total_time:.2f}s")
                    rprint(f"   Steps: {metrics.total_steps}")
                    rprint(f"   Cache Hit Rate: {rates['cache_hit_rate']:.1f}%")
                    rprint(f"   Verification Rate (with fallback): {rates['with_fallback_rate']:.1f}%")
                    rprint(f"   Model Output: {model_output if model_output else '[no assistant reply]'}")

                    if ground_truth:
                        rprint(f"   Ground Truth: {ground_truth}")
                else:
                    progress_bar.set_postfix({
                        "time": f"{metrics.total_time:.1f}s",
                        "steps": metrics.total_steps,
                        "hits": metrics.total_cache_hits,
                    })

            all_metrics.append(metrics)

        except Exception as e:
            rprint(f"[red]❌ Error: {e}[/red]")
            import traceback
            traceback.print_exc()
            continue

    progress_bar.close()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"speculation_{dataset_name}_{timestamp}"
    if results_file:
        experiment_name = Path(results_file).stem
    output_file = save_experiment_results(
        all_metrics,
        experiment_name,
        filename=results_file,
    )

    # Print comprehensive summary
    rprint("\n" + "="*80)
    rprint("[bold green]SPECULATION EXPERIMENT COMPLETE[/bold green]")
    rprint("="*80)

    if all_metrics:
        # Calculate aggregate statistics
        total_hits = sum(m.total_cache_hits for m in all_metrics)
        total_misses = sum(m.total_cache_misses for m in all_metrics)
        total_tools = total_hits + total_misses
        hit_rate = (total_hits / total_tools * 100) if total_tools > 0 else 0.0

        total_ast = sum(m.total_verified_by_ast for m in all_metrics)
        total_embedding = sum(m.total_verified_by_embedding for m in all_metrics)
        total_failed = sum(m.total_both_failed for m in all_metrics)
        total_attempts = total_ast + total_embedding + total_failed

        strict_rate = (total_ast / total_attempts * 100) if total_attempts > 0 else 0.0
        fallback_rate = ((total_ast + total_embedding) / total_attempts * 100) if total_attempts > 0 else 0.0

        rprint(f"\n📊 Summary:")
        rprint(f"   Total Queries: {len(all_metrics)}")
        rprint(f"   Successful: {sum(1 for m in all_metrics if m.success)}")
        rprint(f"   Average Time: {sum(m.total_time for m in all_metrics) / len(all_metrics):.2f}s")

        rprint(f"\n🎯 Cache Performance:")
        rprint(f"   Total Hit Rate: {hit_rate:.1f}%")
        rprint(f"   Total Hits: {total_hits}")
        rprint(f"   Total Misses: {total_misses}")

        rprint(f"\n✓ Verification:")
        rprint(f"   Strict AST Rate: {strict_rate:.1f}%")
        rprint(f"   With Embedding Fallback: {fallback_rate:.1f}%")
        rprint(f"   Improvement: +{fallback_rate - strict_rate:.1f}%")

        rprint(f"\n🎯 Accuracy: pending (run judge_results.py on the saved file)")

    rprint(f"\n💾 Results saved to: {output_file}")
    rprint(f"🔍 Next: python judge_results.py {output_file}")
    rprint("\n" + "="*80)

    return output_file


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run speculation experiments (draft + target models in parallel)"
    )
    parser.add_argument(
        "--dataset",
        choices=["gaia", "hotpot"],
        default="hotpot",
        help="Dataset to use (default: hotpot)"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of queries to run (default: 10)"
    )
    parser.add_argument(
        "--level",
        type=str,
        help="Filter by level (e.g., 'easy', 'medium', 'hard' for HotPotQA; '1', '2', '3' for GAIA)"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        help="Optional path to save the JSON results (will overwrite if exists)"
    )

    args = parser.parse_args()

    # Run experiment
    asyncio.run(run_speculation_experiment(
        dataset_name=args.dataset,
        num_queries=args.num_queries,
        level=args.level,
        results_file=args.results_file
    ))


if __name__ == "__main__":
    main()
