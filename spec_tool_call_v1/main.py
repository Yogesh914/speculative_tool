"""Main entry point for speculative tool calling experiments."""
import asyncio
import argparse
from rich import print as rprint

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from spec_tool_call import (
    build_graph,
    GAIADataset,
    HotPotQADataset,
    EvaluationResults,
    Msg,
    config
)
from spec_tool_call.metrics import MetricsLogger
from spec_tool_call.models import RunState


# Demo question
DEMO_QUESTION = (
    "You will answer a GAIA-style question exactly and only at the end.\n"
    "Task: Find the city where 'the Vietnamese specimens described by Kuznetzov in "
    "Nedoshivina's 2010 paper were deposited'.\n"
    "Do planning and tool calls (web_get, file_read) as needed. "
    "Emit the final line ONLY as 'FINAL ANSWER: <city>'."
)


async def run_demo():
    """Run a single demo question."""
    rprint("[bold blue]Running Demo Question[/bold blue]\n")

    app = build_graph()

    # Initialize state
    init_state = RunState(messages=[Msg(role="user", content=DEMO_QUESTION)])

    # Run graph
    final_state = None
    async for event in app.astream(init_state, config={"configurable": {"thread_id": "demo"}}):
        for node_name, state in event.items():
            if hasattr(state, 'step'):
                MetricsLogger.log_step(state.step, node_name)
                state.step += 1
                final_state = state

    # Print results
    if final_state:
        MetricsLogger.print_messages(final_state.messages)
        MetricsLogger.print_summary(final_state)


async def run_evaluation(level: str = None, max_examples: int = None):
    """Run evaluation on configured dataset (GAIA or HotPotQA)."""

    # Print configuration
    config.print_config()

    # Load appropriate dataset
    if config.is_hotpot():
        rprint(f"[bold blue]Running HotPotQA Evaluation[/bold blue]\n")

        # Load HotPotQA dataset
        hotpot_dataset = HotPotQADataset(csv_path=config.dataset_path)
        hotpot_dataset.load(
            max_examples=max_examples or config.dataset_size,
            random_seed=config.dataset_random_seed
        )

        # Filter by level if specified
        if level:
            examples = hotpot_dataset.get_by_level(level)
            rprint(f"Evaluating Level '{level}': {len(examples)} examples")
        else:
            examples = hotpot_dataset.get_all()
            rprint(f"Evaluating all levels: {len(examples)} examples")

        # Build graph
        app = build_graph()

        # Run evaluation
        results = EvaluationResults()

        for idx, example in enumerate(examples, 1):
            rprint(f"\n[bold]Example {idx}/{len(examples)}[/bold] - {example.id}")
            rprint(f"Question: {example.question[:100]}...")
            rprint(f"Type: {example.type} | Level: {example.level}")

            # Initialize state
            init_state = RunState(messages=[Msg(role="user", content=example.question)])

            # Run graph
            final_state = None
            try:
                async for event in app.astream(
                    init_state,
                    config={"configurable": {"thread_id": f"hotpot-{example.id}"}}
                ):
                    for node_name, state in event.items():
                        if hasattr(state, 'step'):
                            state.step += 1
                            final_state = state
            except Exception as e:
                rprint(f"[red]Error: {e}[/red]")
                continue

            # Collect metrics
            if final_state:
                metrics = MetricsLogger.get_metrics_dict(final_state)
                predicted = final_state.answer or ""

                # Create a GAIA-style object for compatibility with EvaluationResults
                # Note: We use a simple dataclass-like object here
                class HotPotResult:
                    def __init__(self, ex):
                        self.task_id = ex.id
                        self.question = ex.question
                        self.final_answer = ex.answer
                        self.level = ex.level

                results.add(HotPotResult(example), predicted, metrics)

                rprint(f"Predicted: {predicted}")
                rprint(f"Ground Truth: {example.answer}")
                rprint(f"Correct: {predicted.lower().strip() == example.answer.lower().strip()}")

        # Print summary
        results.print_summary()
        results.save_json(f"results_hotpot.json")

    else:  # GAIA dataset
        rprint("[bold blue]Running GAIA Evaluation[/bold blue]\n")

        # Load dataset
        dataset = GAIADataset()
        dataset.load()

        # Get examples to evaluate
        if level:
            examples = dataset.get_level(level)
            rprint(f"Evaluating Level {level}: {len(examples)} examples")
        else:
            examples = dataset.get_all()
            rprint(f"Evaluating all levels: {len(examples)} examples")

        if max_examples:
            examples = examples[:max_examples]
            rprint(f"Limited to first {max_examples} examples\n")

        # Build graph
        app = build_graph()

        # Run evaluation
        results = EvaluationResults()

        for idx, example in enumerate(examples, 1):
            rprint(f"\n[bold]Example {idx}/{len(examples)}[/bold] - {example.task_id}")
            rprint(f"Question: {example.question[:100]}...")

            # Initialize state
            init_state = RunState(messages=[Msg(role="user", content=example.question)])

            # Run graph
            final_state = None
            try:
                async for event in app.astream(
                    init_state,
                    config={"configurable": {"thread_id": f"eval-{example.task_id}"}}
                ):
                    for node_name, state in event.items():
                        if hasattr(state, 'step'):
                            state.step += 1
                            final_state = state
            except Exception as e:
                rprint(f"[red]Error: {e}[/red]")
                continue

            # Collect metrics
            if final_state:
                metrics = MetricsLogger.get_metrics_dict(final_state)
                predicted = final_state.answer or ""
                results.add(example, predicted, metrics)

                rprint(f"Predicted: {predicted}")
                rprint(f"Ground Truth: {example.final_answer}")
                rprint(f"Correct: {predicted.lower().strip() == example.final_answer.lower().strip()}")

        # Print summary
        results.print_summary()
        results.save_json("results_gaia.json")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Speculative Tool Calling Framework")
    parser.add_argument(
        "--mode",
        choices=["demo", "eval"],
        default="demo",
        help="Run mode: demo or eval"
    )
    parser.add_argument(
        "--level",
        choices=["1", "2", "3"],
        help="GAIA level to evaluate (eval mode only)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Maximum number of examples to evaluate"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        asyncio.run(run_demo())
    elif args.mode == "eval":
        asyncio.run(run_evaluation(level=args.level, max_examples=args.max_examples))


if __name__ == "__main__":
    main()
