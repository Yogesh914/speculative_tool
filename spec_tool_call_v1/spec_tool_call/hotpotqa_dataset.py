"""
HotPotQA Dataset Loader

Loads and manages the HotPotQA dataset for multi-hop question answering evaluation.

The HotPotQA dataset contains questions that require reasoning over multiple
supporting documents, making it ideal for testing agent capabilities.

Dataset columns:
- id: Unique identifier
- question: Multi-hop question to answer
- answer: Ground truth answer
- type: Question type ('comparison', 'bridge')
- level: Difficulty level ('easy', 'medium', 'hard')
- supporting_facts: JSON string with supporting evidence
- context_titles: Titles of supporting documents
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HotPotQAExample:
    """Single HotPotQA example."""
    id: str
    question: str
    answer: str
    type: str
    level: str
    supporting_facts: Optional[str] = None
    context_titles: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "type": self.type,
            "level": self.level,
            "supporting_facts": self.supporting_facts,
            "context_titles": self.context_titles,
        }


class HotPotQADataset:
    """HotPotQA dataset loader and manager."""

    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize HotPotQA dataset loader.

        Args:
            csv_path: Path to CSV file. If None, uses default location.
                     Default: ../hotpot_train.csv (relative to project root)
        """
        self.csv_path = csv_path
        self.examples: List[HotPotQAExample] = []
        self.loaded = False

    def _get_default_path(self) -> str:
        """Get default path to hotpot_train.csv."""
        # From spec_tool_call/ directory, go up to project root, then up one more to parent
        current_file = Path(__file__)
        project_root = current_file.parent.parent  # Speculative-Agent/
        parent_dir = project_root.parent  # research/agent_spec/
        return str(parent_dir / "hotpot_train.csv")

    def load(self, max_examples: Optional[int] = None, random_seed: int = 42) -> None:
        """
        Load HotPotQA dataset from CSV.

        Args:
            max_examples: Maximum number of examples to load (None = all)
            random_seed: Random seed for sampling (default: 42 for reproducibility)

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV doesn't have required columns
        """
        # Determine CSV path
        if self.csv_path is None:
            self.csv_path = self._get_default_path()

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"HotPotQA CSV not found at: {self.csv_path}\n"
                f"Expected location: ../hotpot_train.csv (relative to project root)"
            )

        print(f"📖 Loading HotPotQA dataset from: {self.csv_path}")

        # Load CSV
        df = pd.read_csv(self.csv_path)

        # Validate required columns
        required_columns = ["id", "question", "answer", "type", "level"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV missing required columns: {missing_columns}")

        # Normalize difficulty levels and filter for easy/medium only
        if "level" in df.columns:
            df["level"] = df["level"].astype(str).str.lower()
            allowed_levels = {"easy", "medium"}
            before_filter = len(df)
            df = df[df["level"].isin(allowed_levels)]
            print(f"  🎯 Filtering for Easy/Medium only: kept {len(df)} / {before_filter}")
            if df.empty:
                raise ValueError(
                    "No HotPotQA examples remain after filtering for easy/medium difficulty."
                )

        # Sample if max_examples specified
        if max_examples and max_examples < len(df):
            print(f"  🎲 Sampling {max_examples} examples (seed={random_seed})")
            df = df.sample(n=max_examples, random_state=random_seed)
        else:
            print(f"  📊 Loading all {len(df)} examples")

        # Convert to HotPotQAExample objects
        self.examples = []
        for _, row in df.iterrows():
            example = HotPotQAExample(
                id=str(row["id"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
                type=str(row["type"]),
                level=str(row["level"]),
                supporting_facts=str(row.get("supporting_facts", "")) if "supporting_facts" in row else None,
                context_titles=str(row.get("context_titles", "")) if "context_titles" in row else None,
            )
            self.examples.append(example)

        self.loaded = True
        print(f"✅ Loaded {len(self.examples)} HotPotQA examples")

        # Print statistics
        self._print_statistics()

    def _print_statistics(self) -> None:
        """Print dataset statistics."""
        if not self.examples:
            return

        # Count by type
        type_counts = {}
        for ex in self.examples:
            type_counts[ex.type] = type_counts.get(ex.type, 0) + 1

        # Count by level
        level_counts = {}
        for ex in self.examples:
            level_counts[ex.level] = level_counts.get(ex.level, 0) + 1

        print(f"\n📊 Dataset Statistics:")
        print(f"  Total examples: {len(self.examples)}")
        print(f"  By type:")
        for qtype, count in sorted(type_counts.items()):
            print(f"    {qtype}: {count}")
        print(f"  By level:")
        for level, count in sorted(level_counts.items()):
            print(f"    {level}: {count}")

    def get_all(self) -> List[HotPotQAExample]:
        """Get all examples."""
        if not self.loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self.examples

    def get_by_type(self, qtype: str) -> List[HotPotQAExample]:
        """
        Get examples of specific type.

        Args:
            qtype: Question type ('comparison', 'bridge')

        Returns:
            List of examples matching the type.
        """
        if not self.loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return [ex for ex in self.examples if ex.type == qtype]

    def get_by_level(self, level: str) -> List[HotPotQAExample]:
        """
        Get examples of specific difficulty level.

        Args:
            level: Difficulty level ('easy', 'medium', 'hard')

        Returns:
            List of examples matching the level.
        """
        if not self.loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return [ex for ex in self.examples if ex.level == level]

    def get_example(self, idx: int) -> HotPotQAExample:
        """
        Get example by index.

        Args:
            idx: Index (0-based)

        Returns:
            HotPotQAExample at the specified index.
        """
        if not self.loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self.examples[idx]

    def __len__(self) -> int:
        """Get number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> HotPotQAExample:
        """Get example by index (allows indexing)."""
        return self.get_example(idx)


# ==================== Utility Functions ====================

def load_hotpotqa_queries(
    csv_path: Optional[str] = None,
    n_samples: int = 100,
    random_seed: int = 42
) -> List[str]:
    """
    Load HotPotQA queries (questions only).

    This is a convenience function for getting just the questions,
    compatible with the old version's load_test_queries() interface.

    Args:
        csv_path: Path to CSV file (None = use default)
        n_samples: Number of samples to load
        random_seed: Random seed for sampling

    Returns:
        List of question strings.
    """
    dataset = HotPotQADataset(csv_path=csv_path)
    dataset.load(max_examples=n_samples, random_seed=random_seed)
    return [ex.question for ex in dataset.examples]


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("HOTPOTQA DATASET TEST")
    print("="*60)

    # Test loading dataset
    dataset = HotPotQADataset()

    try:
        # Load first 10 examples
        dataset.load(max_examples=10, random_seed=42)

        # Show sample examples
        print("\n" + "="*60)
        print("SAMPLE EXAMPLES")
        print("="*60)

        for i in range(min(3, len(dataset))):
            ex = dataset[i]
            print(f"\nExample {i+1}:")
            print(f"  ID: {ex.id}")
            print(f"  Type: {ex.type} | Level: {ex.level}")
            print(f"  Question: {ex.question}")
            print(f"  Answer: {ex.answer}")

        # Test filtering
        print("\n" + "="*60)
        print("FILTERING TEST")
        print("="*60)

        comparison_qs = dataset.get_by_type("comparison")
        print(f"\nComparison questions: {len(comparison_qs)}")

        easy_qs = dataset.get_by_level("easy")
        print(f"Easy questions: {len(easy_qs)}")

        # Test convenience function
        print("\n" + "="*60)
        print("CONVENIENCE FUNCTION TEST")
        print("="*60)

        queries = load_hotpotqa_queries(n_samples=5)
        print(f"\nLoaded {len(queries)} queries:")
        for i, q in enumerate(queries[:3], 1):
            print(f"  {i}. {q[:80]}...")

        print("\n" + "="*60)
        print("✅ TEST COMPLETE")
        print("="*60)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo use HotPotQA dataset:")
        print("1. Download hotpot_train.csv")
        print("2. Place it in the parent directory (../hotpot_train.csv)")
        print("3. Or specify custom path: HotPotQADataset(csv_path='your/path.csv')")
