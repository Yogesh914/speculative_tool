#!/usr/bin/env python3
"""
Compare baseline vs speculation on sampled GAIA questions.

Usage:
    python compare_gaia.py --level 1 --num-queries 3
"""
import argparse
import asyncio
import random
from pathlib import Path

from dotenv import load_dotenv
from rich import print as rprint

load_dotenv()

from spec_tool_call import (
    build_graph,
    build_speculation_graph,
    GAIADataset,
    Msg,
)
from spec_tool_call.models import RunState
from spec_tool_call.speculation_state import SpeculationState


async def run_baseline(question: str):
    """Run baseline (no speculation) on a single GAIA question."""
    app = build_graph()
    init_state = RunState(messages=[Msg(role="user", content=question)])

    final_state = None
    import time
    start_time = time.time()
    async for event in app.astream(
        init_state,
        config={"configurable": {"thread_id": "gaia-baseline"}},
    ):
        for _, state in event.items():
            final_state = state
    elapsed = time.time() - start_time

    if isinstance(final_state, dict):
        answer = final_state.get("answer", "")
        steps = final_state.get("step", 0)
    else:
        answer = getattr(final_state, "answer", "")
        steps = getattr(final_state, "step", 0)

    return {"time": elapsed, "answer": answer, "steps": steps}


async def run_speculation(question: str):
    """Run speculation on a single GAIA question."""
    app = build_speculation_graph()
    init_state = SpeculationState(messages=[Msg(role="user", content=question)])

    final_state = None
    import time
    start_time = time.time()
    async for event in app.astream(
        init_state,
        config={"configurable": {"thread_id": "gaia-speculation"}},
    ):
        for _, state in event.items():
            final_state = state
    elapsed = time.time() - start_time

    if isinstance(final_state, dict):
        answer = final_state.get("answer", "")
        steps = final_state.get("step", 0)
        hits = final_state.get("cache_hits", 0)
        misses = final_state.get("cache_misses", 0)
    else:
        answer = getattr(final_state, "answer", "")
        steps = getattr(final_state, "step", 0)
        hits = getattr(final_state, "cache_hits", 0)
        misses = getattr(final_state, "cache_misses", 0)

    total = hits + misses
    hit_rate = (hits / total * 100) if total else 0.0

    return {
        "time": elapsed,
        "answer": answer,
        "steps": steps,
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate": hit_rate,
    }


def load_gaia_samples(level: str, num_queries: int, seed: int):
    """Load GAIA dataset and sample tasks from the specified level."""
    dataset = GAIADataset()
    dataset.load()
    if level:
        examples = dataset.get_level(level)
    else:
        examples = dataset.get_all()

    if len(examples) < num_queries:
        raise ValueError("Not enough GAIA examples for the requested sample.")

    rng = random.Random(seed)
    return rng.sample(examples, num_queries)


async def compare_question(task_id: str, question: str, ground_truth: str = None):
    """Compare baseline vs speculation for one GAIA example."""
    rprint("\n" + "=" * 80)
    rprint(f"[bold cyan]⚡ GAIA SPECULATION COMPARISON[/bold cyan] - Task {task_id}")
    rprint("=" * 80)
    rprint(f"\n[bold]Question:[/bold] {question}")
    if ground_truth:
        rprint(f"[bold]Ground Truth:[/bold] {ground_truth}")

    baseline_result = await run_baseline(question)
    rprint(
        f"[yellow]Baseline:[/yellow] {baseline_result['time']:.2f}s | "
        f"{baseline_result['steps']} steps | "
        f"Answer: {baseline_result['answer'] or '[no answer]'}"
    )

    speculation_result = await run_speculation(question)
    rprint(
        f"[green]Speculation:[/green] {speculation_result['time']:.2f}s | "
        f"{speculation_result['steps']} steps | "
        f"Answer: {speculation_result['answer'] or '[no answer]'}"
    )
    total_hits = speculation_result["cache_hits"] + speculation_result["cache_misses"]
    rprint(
        f"Cache Hits: {speculation_result['cache_hits']}/{total_hits} "
        f"({speculation_result['hit_rate']:.0f}%)"
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs speculation on GAIA tasks."
    )
    parser.add_argument(
        "--level",
        type=str,
        default="1",
        help="GAIA level to sample from (1, 2, or 3). Default: 1",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=3,
        help="Number of GAIA tasks to sample. Default: 3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    samples = load_gaia_samples(args.level, args.num_queries, args.seed)
    for example in samples:
        if hasattr(example, "question"):
            question = example.question
            ground_truth = getattr(example, "final_answer", None)
            task_id = getattr(example, "task_id", "unknown")
        else:
            question = str(example)
            ground_truth = None
            task_id = "unknown"

        await compare_question(task_id, question, ground_truth)


if __name__ == "__main__":
    asyncio.run(main())

