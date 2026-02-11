#!/usr/bin/env python3
"""
Compare baseline vs speculation on sampled HotPotQA questions.

Usage:
    python compare_hotpot.py --num-queries 5 --seed 42
"""
import argparse
import asyncio
import random
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv()

from spec_tool_call import (
    build_graph,
    build_speculation_graph,
    HotPotQADataset,
    Msg,
)
from spec_tool_call.models import RunState
from spec_tool_call.speculation_state import SpeculationState


async def run_baseline(question: str):
    """Run a single baseline (no speculation) query."""
    app = build_graph()
    init_state = RunState(messages=[Msg(role="user", content=question)])

    final_state = None
    import time
    start_time = time.time()
    async for event in app.astream(
        init_state,
        config={"configurable": {"thread_id": "baseline-compare"}},
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

    return {
        "time": elapsed,
        "answer": answer,
        "steps": steps,
    }


async def run_speculation(question: str):
    """Run a single query with speculation enabled."""
    app = build_speculation_graph()
    init_state = SpeculationState(messages=[Msg(role="user", content=question)])

    final_state = None
    import time
    start_time = time.time()
    async for event in app.astream(
        init_state,
        config={"configurable": {"thread_id": "speculation-compare"}},
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


async def compare_question(question: str, ground_truth: str = None):
    """Run both baseline and speculation on a single question."""
    rprint("\n" + "=" * 80)
    rprint("[bold cyan]⚡ SPECULATION COMPARISON[/bold cyan]")
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
    rprint(
        f"Cache Hits: {speculation_result['cache_hits']}/"
        f"{speculation_result['cache_hits'] + speculation_result['cache_misses']} "
        f"({speculation_result['hit_rate']:.0f}%)"
    )


def sample_hotpot_questions(num_queries: int, seed: int = 42):
    """Load HotPotQA and sample easy/medium questions."""
    dataset = HotPotQADataset()
    dataset.load(max_examples=None, random_seed=seed)

    filtered = [
        ex
        for ex in dataset.examples
        if ex.level.lower() in {"easy", "medium"}
    ]
    if len(filtered) < num_queries:
        raise ValueError("Not enough easy/medium examples in dataset.")

    rng = random.Random(seed)
    samples = rng.sample(filtered, num_queries)
    return samples


async def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs speculation on sampled HotPotQA questions."
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=3,
        help="Number of HotPotQA questions to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    args = parser.parse_args()

    samples = sample_hotpot_questions(args.num_queries, args.seed)
    for ex in samples:
        await compare_question(ex.question, ex.answer)


if __name__ == "__main__":
    asyncio.run(main())

