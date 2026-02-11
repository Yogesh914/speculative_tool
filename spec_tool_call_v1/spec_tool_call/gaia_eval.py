"""GAIA dataset loading and evaluation."""
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from rich import print as rprint
from rich.table import Table


@dataclass
class GAIAExample:
    """Single GAIA example."""
    task_id: str
    question: str
    level: str
    final_answer: str
    file_name: str = None
    example_dir: str = None

    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any], example_dir: str) -> "GAIAExample":
        """Load from metadata.json."""
        return cls(
            task_id=metadata["task_id"],
            question=metadata["question"],
            level=metadata["level"],
            final_answer=metadata["final_answer"],
            file_name=metadata.get("file_name"),
            example_dir=example_dir,
        )

    def get_file_path(self) -> str:
        """Get full path to attached file if exists."""
        if self.file_name and self.example_dir:
            return os.path.join(self.example_dir, self.file_name)
        return None


class GAIADataset:
    """GAIA dataset loader."""

    def __init__(self, dataset_root: str = "gaia_dataset"):
        self.dataset_root = Path(dataset_root)
        self.examples: Dict[str, List[GAIAExample]] = {
            "1": [],
            "2": [],
            "3": [],
        }

    def load(self):
        """Load all examples from dataset."""
        for level in ["1", "2", "3"]:
            level_dir = self.dataset_root / f"level{level}"
            if not level_dir.exists():
                continue

            for example_dir in sorted(level_dir.iterdir()):
                if not example_dir.is_dir():
                    continue

                metadata_path = example_dir / "metadata.json"
                if not metadata_path.exists():
                    continue

                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                example = GAIAExample.from_metadata(metadata, str(example_dir))
                self.examples[level].append(example)

        rprint(f"[bold]Loaded GAIA dataset:[/bold]")
        for level in ["1", "2", "3"]:
            rprint(f"  Level {level}: {len(self.examples[level])} examples")

    def get_level(self, level: str) -> List[GAIAExample]:
        """Get examples for specific level."""
        return self.examples.get(level, [])

    def get_all(self) -> List[GAIAExample]:
        """Get all examples."""
        return sum(self.examples.values(), [])


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (lowercase, strip)."""
    return answer.strip().lower()


def exact_match(predicted: str, ground_truth: str) -> bool:
    """Check exact match with normalization."""
    return normalize_answer(predicted) == normalize_answer(ground_truth)


class EvaluationResults:
    """Store and display evaluation results."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def add(self, example: GAIAExample, predicted: str, metrics: Dict[str, Any]):
        """Add a single evaluation result."""
        correct = exact_match(predicted, example.final_answer)
        self.results.append({
            "task_id": example.task_id,
            "level": example.level,
            "correct": correct,
            "predicted": predicted,
            "ground_truth": example.final_answer,
            **metrics
        })

    def get_accuracy_by_level(self) -> Dict[str, float]:
        """Calculate accuracy for each level."""
        by_level = {"1": [], "2": [], "3": []}
        for r in self.results:
            by_level[r["level"]].append(r["correct"])

        return {
            level: (sum(correct) / len(correct) * 100 if correct else 0.0)
            for level, correct in by_level.items()
        }

    def print_summary(self):
        """Print evaluation summary."""
        rprint("\n[bold]Evaluation Results[/bold]")

        # Overall stats
        total = len(self.results)
        correct = sum(r["correct"] for r in self.results)
        accuracy = correct / total * 100 if total > 0 else 0.0

        rprint(f"Total: {correct}/{total} ({accuracy:.1f}%)")

        # Level-wise stats
        acc_by_level = self.get_accuracy_by_level()
        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("Level")
        tbl.add_column("Accuracy")
        tbl.add_column("Avg Hit Rate")
        tbl.add_column("Avg Latency(s)")

        for level in ["1", "2", "3"]:
            level_results = [r for r in self.results if r["level"] == level]
            if not level_results:
                continue

            avg_hit_rate = sum(r["hit_rate"] for r in level_results) / len(level_results) * 100
            avg_latency = sum(r["elapsed_seconds"] for r in level_results) / len(level_results)

            tbl.add_row(
                f"Level {level}",
                f"{acc_by_level[level]:.1f}%",
                f"{avg_hit_rate:.1f}%",
                f"{avg_latency:.2f}"
            )

        rprint(tbl)

    def save_json(self, output_path: str):
        """Save results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        rprint(f"[green]Results saved to {output_path}[/green]")
