#!/usr/bin/env python3
"""
Run Baseline Experiments (Target Model Only)

Runs experiments with the target model only (no speculation) to establish
baseline performance metrics for comparison with speculation-enabled runs.

Usage:
    python run_baseline.py --dataset hotpot --num-queries 10
    python run_baseline.py --dataset gaia --level 1 --num-queries 5
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
    build_graph,
    GAIADataset,
    HotPotQADataset,
    config,
    Msg,
)
from spec_tool_call.models import RunState
from spec_tool_call.enhanced_metrics import create_experiment_metrics
from spec_tool_call.results_saver import save_experiment_results


def _get_state_attr(state, attr, default=None):
    """Helper to read attributes whether LangGraph gives a dict or dataclass."""
    if isinstance(state, dict):
        return state.get(attr, default)
    return getattr(state, attr, default)


def _get_last_assistant_message(messages):
    """Return the content of the most recent assistant message."""
    for msg in reversed(messages or []):
        if getattr(msg, "role", None) == "assistant":
            return msg.content
    return ""


async def run_baseline_experiment(
    dataset_name: str,
    num_queries: int = 10,
    level: str = None,
    results_file: str = None,
):
    """
    Run baseline experiments with target model only.

    Args:
        dataset_name: "gaia" or "hotpot"
        num_queries: Number of queries to run
        level: Dataset level filter (optional)
    """
    # Ensure speculation is disabled for baseline
    import os
    os.environ["DISABLE_SPECULATION"] = "1"

    # Reload config to pick up changes
    from spec_tool_call.config import SpecConfig
    baseline_config = SpecConfig.from_env()

    rprint("\n" + "="*80)
    rprint("[bold cyan]BASELINE EXPERIMENT (Target Model Only)[/bold cyan]")
    rprint("="*80)
    baseline_config.print_config()

    # Load dataset
    if dataset_name == "hotpot":
        dataset = HotPotQADataset()
        dataset.load(max_examples=num_queries, random_seed=baseline_config.dataset_random_seed)
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

    rprint(f"\n📊 Running baseline on {query_count} queries")

    # Build graph (will use standard graph without speculation)
    app = build_graph()

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
        init_state = RunState(messages=[Msg(role="user", content=question)])

        # Run graph
        import time
        start_time = time.time()
        final_state = None

        try:
            async for event in app.astream(
                init_state,
                config={"configurable": {"thread_id": f"baseline-{example_id}"}}
            ):
                for node_name, state in event.items():
                    final_state = state

            # Record results
            if final_state is not None:
                metrics.total_time = time.time() - start_time
                metrics.total_steps = int(_get_state_attr(final_state, "step", 0))

                # Capture raw LLM output (prefer the last assistant message)
                final_messages = _get_state_attr(final_state, "messages", [])
                model_output = _get_last_assistant_message(final_messages)
                fallback_answer = _get_state_attr(final_state, "answer", "") or ""
                if not model_output and fallback_answer:
                    model_output = fallback_answer

                metrics.model_output = model_output
                metrics.predicted_answer = model_output  # keep legacy field populated
                metrics.success = bool(model_output)

                if verbose_per_query:
                    rprint(f"\n✅ Completed in {metrics.total_time:.2f}s")
                    rprint(f"   Steps: {metrics.total_steps}")
                    rprint(f"   Model Output: {model_output if model_output else '[no assistant reply]'}")
                    if ground_truth:
                        rprint(f"   Ground Truth: {ground_truth}")
                else:
                    progress_bar.set_postfix({
                        "steps": metrics.total_steps,
                        "time": f"{metrics.total_time:.1f}s",
                        "success": "yes" if metrics.success else "no",
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
    experiment_name = f"baseline_{dataset_name}_{timestamp}"
    if results_file:
        experiment_name = Path(results_file).stem
    output_file = save_experiment_results(
        all_metrics,
        experiment_name,
        filename=results_file,
    )

    # Print summary
    rprint("\n" + "="*80)
    rprint("[bold green]BASELINE EXPERIMENT COMPLETE[/bold green]")
    rprint("="*80)
    rprint(f"\n📊 Summary:")
    rprint(f"   Total Queries: {len(all_metrics)}")
    rprint(f"   Successful: {sum(1 for m in all_metrics if m.success)}")
    rprint(f"   Average Time: {sum(m.total_time for m in all_metrics) / len(all_metrics):.2f}s")

    rprint("   Accuracy: pending (run judge_results.py on the saved file)")

    rprint(f"\n💾 Results saved to: {output_file}")
    rprint(f"🔍 Next: python judge_results.py {output_file}")
    rprint("\n" + "="*80)

    return output_file


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run baseline experiments (target model only, no speculation)"
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
    asyncio.run(run_baseline_experiment(
        dataset_name=args.dataset,
        num_queries=args.num_queries,
        level=args.level,
        results_file=args.results_file
    ))


if __name__ == "__main__":
    main()
