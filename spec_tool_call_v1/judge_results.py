#!/usr/bin/env python3
"""
Post-process experiment results with an LLM judge.

Usage:
    python judge_results.py results/baseline_hotpot_20251113_120817.json
    python judge_results.py results/*.json
"""
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from spec_tool_call.accuracy_judge import evaluate_accuracy_with_judge


class JudgeStub:
    """Lightweight object to reuse evaluate_accuracy_with_judge for stored runs."""

    def __init__(self, record: Dict[str, Any]):
        self.record = record
        query_block = record.get("query", {})
        self.query = query_block.get("question", "")
        self.ground_truth = query_block.get("ground_truth")
        self.model_output = query_block.get("model_output") or query_block.get("predicted")
        self.predicted_answer = query_block.get("predicted")
        judge_block = query_block.get("judge", {})
        self.judge_decision = judge_block.get("decision")
        self.judge_reason = judge_block.get("raw")
        self.judge_correct = judge_block.get("correct")


async def judge_file(path: Path) -> None:
    """Run the judge on a saved results JSON and update it in-place."""
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r") as f:
        data = json.load(f)

    queries: List[Dict[str, Any]] = data.get("queries", [])
    stubs: List[JudgeStub] = [JudgeStub(record) for record in queries]

    if not stubs:
        print(f"⚠️  No queries found in {path}")
        return

    await evaluate_accuracy_with_judge(stubs)

    judged = 0
    correct = 0

    for stub in stubs:
        record = stub.record
        query_block = record.setdefault("query", {})
        query_block["judge"] = {
            "decision": stub.judge_decision,
            "raw": stub.judge_reason,
            "correct": stub.judge_correct,
        }
        query_block["correct"] = stub.judge_correct

        if stub.judge_correct is not None:
            judged += 1
            if stub.judge_correct:
                correct += 1

    accuracy = (correct / judged * 100) if judged else 0.0

    summary = data.setdefault("summary", {})
    summary["accuracy"] = accuracy
    summary["accuracy_metrics"] = {
        "judged_queries": judged,
        "judged_correct": correct,
    }

    metadata = data.setdefault("metadata", {})
    metadata["accuracy_judged_at"] = datetime.utcnow().isoformat()

    with path.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Judged {judged} queries ({accuracy:.1f}% correct) in {path}")


async def main(files: List[str]) -> None:
    for file_path in files:
        await judge_file(Path(file_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM-based accuracy judging on saved experiment results."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Result JSON files to update in-place.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.files))

