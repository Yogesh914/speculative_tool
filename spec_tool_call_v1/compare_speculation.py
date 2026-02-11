#!/usr/bin/env python3
"""Simple comparison: Baseline vs Speculation"""
import asyncio
import time
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv()


async def run_baseline(question: str):
    from spec_tool_call import build_graph, Msg
    from spec_tool_call.models import RunState
    
    rprint("\n[yellow]🔹 Running BASELINE (no speculation)...[/yellow]")
    
    app = build_graph()
    init_state = RunState(messages=[Msg(role="user", content=question)])
    
    start_time = time.time()
    
    final_state = None
    async for event in app.astream(
        init_state,
        config={"configurable": {"thread_id": "baseline-test"}}
    ):
        for node_name, state in event.items():
            final_state = state
    
    elapsed = time.time() - start_time
    
    # Extract values (handle both dict and object)
    if isinstance(final_state, dict):
        answer = final_state.get('answer', 'No answer')
        steps = final_state.get('step', 0)
    else:
        answer = final_state.answer if hasattr(final_state, 'answer') else "No answer"
        steps = final_state.step if hasattr(final_state, 'step') else 0
    
    rprint(f"[yellow]✓ Baseline completed in {elapsed:.2f}s ({steps} steps)[/yellow]")
    rprint(f"  Answer: {answer}")
    
    return {
        "time": elapsed,
        "answer": answer,
        "steps": steps
    }


async def run_speculation(question: str):
    from spec_tool_call.speculation_graph import build_speculation_graph
    from spec_tool_call.speculation_state import SpeculationState, Msg
    
    rprint("\n[green]🔹 Running SPECULATION (draft + target parallel)...[/green]")
    
    app = build_speculation_graph()
    init_state = SpeculationState(messages=[Msg(role="user", content=question)])
    
    start_time = time.time()
    
    final_state = None
    async for event in app.astream(
        init_state,
        config={"configurable": {"thread_id": "speculation-test"}}
    ):
        for node_name, state in event.items():
            final_state = state
    
    elapsed = time.time() - start_time
    
    # Extract values (handle both dict and object)
    if isinstance(final_state, dict):
        answer = final_state.get('answer', 'No answer')
        steps = final_state.get('step', 0)
        hits = final_state.get('cache_hits', 0)
        misses = final_state.get('cache_misses', 0)
    else:
        answer = final_state.answer if hasattr(final_state, 'answer') else "No answer"
        steps = final_state.step if hasattr(final_state, 'step') else 0
        hits = final_state.cache_hits if hasattr(final_state, 'cache_hits') else 0
        misses = final_state.cache_misses if hasattr(final_state, 'cache_misses') else 0
    
    total = hits + misses
    hit_rate = (hits / total * 100) if total > 0 else 0
    
    rprint(f"[green]✓ Speculation completed in {elapsed:.2f}s ({steps} steps)[/green]")
    rprint(f"  Answer: {answer}")
    rprint(f"  Cache: {hits}/{total} hits ({hit_rate:.0f}%)")
    
    return {
        "time": elapsed,
        "answer": answer,
        "steps": steps,
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate": hit_rate
    }


async def compare(question: str):
    rprint("\n" + "="*80)
    rprint("[bold cyan]⚡ SPECULATION COMPARISON TEST[/bold cyan]")
    rprint("="*80)
    rprint(f"\n[bold]Question:[/bold] {question}\n")
    
    # Run both modes
    baseline_result = await run_baseline(question)
    speculation_result = await run_speculation(question)
    
    time_saved = baseline_result["time"] - speculation_result["time"]
    speedup = baseline_result["time"] / speculation_result["time"]
    percentage = (time_saved / baseline_result["time"]) * 100
    
    rprint("\n" + "="*80)
    rprint("[bold cyan]📊 COMPARISON RESULTS[/bold cyan]")
    rprint("="*80)
    
    rprint(f"\n[bold]Time Comparison:[/bold]")
    rprint(f"  Baseline:     [yellow]{baseline_result['time']:.2f}s[/yellow]")
    rprint(f"  Speculation:  [green]{speculation_result['time']:.2f}s[/green]")
    rprint(f"  Time Saved:   [magenta]{time_saved:.2f}s[/magenta]")
    
    if speedup > 1:
        rprint(f"\n[bold green]🎉 Speedup: {speedup:.2f}x faster ({percentage:.1f}% reduction)[/bold green]")
    else:
        rprint(f"\n[bold yellow]⚠️  No speedup ({speedup:.2f}x)[/bold yellow]")
    
    if speculation_result.get("cache_hits", 0) > 0:
        rprint(f"\n[bold]Speculation Stats:[/bold]")
        rprint(f"  Cache Hit Rate: [green]{speculation_result['hit_rate']:.0f}%[/green]")
        rprint(f"  Saved {speculation_result['cache_hits']} tool execution(s)!")
    
    rprint("\n" + "="*80)


async def main():
    """Main function"""
    # Calculation test (original)
    # question = "What is 25 * 48? Please calculate this for me and give me the final answer."
    
    # Web search test (using Gensee AI)
    question = "Search for information about 'artificial intelligence' and tell me what it is in one sentence."
    
    await compare(question)


if __name__ == "__main__":
    asyncio.run(main())

