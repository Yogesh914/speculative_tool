"""Metrics logging and tracking for speculative execution."""
import time
from typing import Dict, Any, List

from rich import print as rprint
from rich.table import Table

from .models import RunState, Msg


class MetricsLogger:
    """Logger for speculation metrics and results."""

    @staticmethod
    def log_speculation(num_preds: int):
        """Log speculator launching predictions."""
        if num_preds > 0:
            rprint(f"[cyan]Speculator launched {num_preds} future(s)[/cyan]")

    @staticmethod
    def log_step(step: int, node: str):
        """Log graph node execution."""
        if node == "actor_decide":
            rprint(f"[magenta]Step {step} → actor_decide[/magenta]")
        elif node == "speculate":
            rprint(f"[green]Step {step} → speculate[/green]")

    @staticmethod
    def print_messages(messages: List[Msg], max_length: int = 180):
        """Print conversation messages."""
        rprint("\n[bold]Messages:[/bold]")
        for m in messages:
            tag = m.role if m.role != "tool" else f"tool:{m.name}"
            content_str = m.content[:max_length]
            if len(m.content) > max_length:
                content_str += "..."
            rprint(f"[yellow]{tag}[/yellow]: {content_str}")

    @staticmethod
    def print_summary(state: RunState):
        """Print execution summary with metrics."""
        elapsed = time.time() - state.t0

        rprint("\n[bold]Speculation Metrics[/bold]")
        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("Hits")
        tbl.add_column("Misses")
        tbl.add_column("Launched")
        tbl.add_column("Hit Rate")
        tbl.add_column("Elapsed(s)")

        total = state.hits + state.misses
        hit_rate = (state.hits / total * 100) if total > 0 else 0.0

        tbl.add_row(
            str(state.hits),
            str(state.misses),
            str(state.speculative_launched),
            f"{hit_rate:.1f}%",
            f"{elapsed:.2f}"
        )
        rprint(tbl)

        if state.answer:
            rprint(f"[bold green]FINAL ANSWER[/bold green]: {state.answer}")

    @staticmethod
    def get_metrics_dict(state: RunState) -> Dict[str, Any]:
        """Extract metrics as dictionary for evaluation."""
        elapsed = time.time() - state.t0
        total = state.hits + state.misses
        hit_rate = (state.hits / total) if total > 0 else 0.0

        return {
            "hits": state.hits,
            "misses": state.misses,
            "launched": state.speculative_launched,
            "hit_rate": hit_rate,
            "elapsed_seconds": elapsed,
            "steps": state.step,
            "answer": state.answer,
        }
