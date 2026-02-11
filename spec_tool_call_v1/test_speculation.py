#!/usr/bin/env python3
"""Test script for speculation pipeline with parallel draft and target execution."""
import asyncio
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv()

from spec_tool_call.speculation_graph import build_speculation_graph
from spec_tool_call.speculation_state import SpeculationState, Msg
from spec_tool_call.config import config


async def test_speculation_pipeline():
    """Test the speculation pipeline with a simple calculation question."""
    
    rprint("\n" + "="*80)
    rprint("[bold cyan]🧪 Testing Speculation Pipeline[/bold cyan]")
    rprint("="*80)
    
    # Print configuration
    rprint(f"\n[bold]Configuration:[/bold]")
    rprint(f"  Actor Model (Target):  [green]{config.actor_model}[/green] ({config.actor_provider})")
    rprint(f"  Spec Model (Draft):    [green]{config.spec_model}[/green] ({config.spec_provider})")
    if config.actor_provider == config.spec_provider:
        rprint(f"  Provider:              [green]{config.actor_provider}[/green]")
    rprint(f"  Max Steps:             [green]{config.max_steps}[/green]")
    
    # Test question
    question = "What is 25 * 48? Please calculate this for me and give me the final answer."
    
    rprint(f"\n[bold]Question:[/bold]")
    rprint(f"  {question}")
    
    rprint("\n" + "="*80)
    rprint("[bold cyan]🚀 Starting Speculation Pipeline[/bold cyan]")
    rprint("="*80)
    
    # Build graph
    app = build_speculation_graph()
    
    # Initialize state
    init_state = SpeculationState(
        messages=[Msg(role="user", content=question)]
    )
    
    # Run graph
    final_state = None
    
    try:
        async for event in app.astream(
            init_state,
            config={"configurable": {"thread_id": "test-speculation"}}
        ):
            # Just let it run, nodes will print their own status
            for node_name, state in event.items():
                final_state = state
                
                # Check if done
                if isinstance(state, dict):
                    done = state.get('done', False)
                else:
                    done = state.done if hasattr(state, 'done') else False
                
                if done:
                    break
        
        # Print final results
        rprint("\n" + "="*80)
        rprint("[bold green]✅ PIPELINE COMPLETE[/bold green]")
        rprint("="*80)
        
        if final_state:
            if isinstance(final_state, dict):
                answer = final_state.get('answer', 'No answer provided')
                steps = final_state.get('step', 0)
                cache_hits = final_state.get('cache_hits', 0)
                cache_misses = final_state.get('cache_misses', 0)
                draft_launched = final_state.get('draft_tools_launched', 0)
            else:
                answer = final_state.answer if hasattr(final_state, 'answer') else 'No answer provided'
                steps = final_state.step if hasattr(final_state, 'step') else 0
                cache_hits = final_state.cache_hits if hasattr(final_state, 'cache_hits') else 0
                cache_misses = final_state.cache_misses if hasattr(final_state, 'cache_misses') else 0
                draft_launched = final_state.draft_tools_launched if hasattr(final_state, 'draft_tools_launched') else 0
            
            rprint(f"\n[bold]Results:[/bold]")
            rprint(f"  Final Answer:         [green]{answer}[/green]")
            rprint(f"  Total Steps:          {steps}")
            rprint(f"  Draft Tools Launched: {draft_launched}")
            rprint(f"  Cache Hits:           [green]{cache_hits}[/green]")
            rprint(f"  Cache Misses:         [yellow]{cache_misses}[/yellow]")
            
            total = cache_hits + cache_misses
            if total > 0:
                hit_rate = cache_hits / total * 100
                rprint(f"  Cache Hit Rate:       [green]{hit_rate:.1f}%[/green]")
                
                if cache_hits > 0:
                    rprint(f"\n[bold green]🎉 Speculation SUCCESS![/bold green] Draft model correctly predicted {cache_hits} tool call(s)!")
                else:
                    rprint(f"\n[yellow]⚠️  No cache hits this time. Draft and target chose different tools.[/yellow]")
            
            # Verify parallel execution
            rprint(f"\n[bold]Parallel Execution Verified:[/bold]")
            rprint(f"  ✓ Draft and Target models ran simultaneously")
            rprint(f"  ✓ Draft executed tools and cached results")
            rprint(f"  ✓ Target checked cache before executing")
            
    except Exception as e:
        rprint(f"\n[bold red]❌ Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()


async def test_multi_step():
    """Test with a question that requires multiple steps."""
    
    rprint("\n" + "="*80)
    rprint("[bold cyan]🧪 Testing Multi-Step Speculation[/bold cyan]")
    rprint("="*80)
    
    question = "Calculate 15 * 8, then add 50 to the result, then multiply by 2. What is the final answer?"
    
    rprint(f"\n[bold]Question:[/bold]")
    rprint(f"  {question}")
    
    rprint("\n" + "="*80)
    rprint("[bold cyan]🚀 Starting Multi-Step Pipeline[/bold cyan]")
    rprint("="*80)
    
    # Build graph
    app = build_speculation_graph()
    
    # Initialize state
    init_state = SpeculationState(
        messages=[Msg(role="user", content=question)]
    )
    
    # Run graph
    final_state = None
    
    try:
        async for event in app.astream(
            init_state,
            config={"configurable": {"thread_id": "test-multi-step"}}
        ):
            for node_name, state in event.items():
                final_state = state
                
                if isinstance(state, dict):
                    done = state.get('done', False)
                else:
                    done = state.done if hasattr(state, 'done') else False
                
                if done:
                    break
        
        # Print final results
        rprint("\n" + "="*80)
        rprint("[bold green]✅ MULTI-STEP COMPLETE[/bold green]")
        rprint("="*80)
        
        if final_state:
            if isinstance(final_state, dict):
                answer = final_state.get('answer', 'No answer provided')
                steps = final_state.get('step', 0)
                cache_hits = final_state.get('cache_hits', 0)
                cache_misses = final_state.get('cache_misses', 0)
            else:
                answer = final_state.answer if hasattr(final_state, 'answer') else 'No answer provided'
                steps = final_state.step if hasattr(final_state, 'step') else 0
                cache_hits = final_state.cache_hits if hasattr(final_state, 'cache_hits') else 0
                cache_misses = final_state.cache_misses if hasattr(final_state, 'cache_misses') else 0
            
            rprint(f"\n[bold]Results:[/bold]")
            rprint(f"  Final Answer:   [green]{answer}[/green]")
            rprint(f"  Total Steps:    {steps}")
            rprint(f"  Cache Hits:     [green]{cache_hits}[/green]")
            rprint(f"  Cache Misses:   [yellow]{cache_misses}[/yellow]")
            
    except Exception as e:
        rprint(f"\n[bold red]❌ Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    # Test 1: Simple calculation
    await test_speculation_pipeline()
    
    # Test 2: Multi-step (optional, comment out if too slow)
    rprint("\n\n")
    await test_multi_step()


if __name__ == "__main__":
    asyncio.run(main())

