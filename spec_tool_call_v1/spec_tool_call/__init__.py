"""Speculative tool calling framework for LangGraph agents."""

from .config import SpecConfig, config
from .models import Msg, RunState, ToolSpec
from .graph import build_graph
from .gaia_eval import GAIADataset, EvaluationResults, GAIAExample
from .hotpotqa_dataset import HotPotQADataset, HotPotQAExample, load_hotpotqa_queries
from .speculation_state import SpeculationState
from .speculation_graph import build_speculation_graph
from .tool_verification import speculation_checker, verify_single_tool_call
from .accuracy_judge import evaluate_accuracy_with_judge

__all__ = [
    "SpecConfig",
    "config",
    "Msg",
    "RunState",
    "ToolSpec",
    "build_graph",
    "GAIADataset",
    "EvaluationResults",
    "GAIAExample",
    "HotPotQADataset",
    "HotPotQAExample",
    "load_hotpotqa_queries",
    "SpeculationState",
    "build_speculation_graph",
    "speculation_checker",
    "verify_single_tool_call",
    "evaluate_accuracy_with_judge",
]
