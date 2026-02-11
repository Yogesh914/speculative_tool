"""LLM-based accuracy judging utilities."""

import asyncio
import os
from typing import List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from .enhanced_metrics import ExperimentMetrics


DEFAULT_JUDGE_MODEL = "gemini-2.5-flash-lite"
DEFAULT_JUDGE_PROVIDER = "google-genai"
DEFAULT_JUDGE_TEMPERATURE = "0.0"

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial grader. "
    "Given the original question, the model's full output, and the ground truth answer, "
    "determine if the model output contains a correct and sufficient answer. "
    "Respond with exactly one word: CORRECT or WRONG."
)

JUDGE_USER_TEMPLATE = """Question:
{question}

Model Output:
{model_output}

Ground Truth Answer:
{ground_truth}

Does the model output answer the question correctly? Respond with ONLY CORRECT or WRONG."""


def _extract_content(response) -> str:
    """Normalize LangChain response content to plain string."""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return " ".join(parts)
    return str(content or "")


def _normalize_decision(raw: str) -> str:
    """Convert raw judge response to CORRECT/WRONG."""
    text = (raw or "").strip().upper()
    if text.startswith("CORRECT"):
        return "CORRECT"
    if text.startswith("WRONG"):
        return "WRONG"
    # Fallback: treat anything else as wrong to be conservative
    return "WRONG"


def _run_judge(metrics_list: List[ExperimentMetrics]) -> None:
    """Synchronously evaluate all metrics with the judge model."""
    if not metrics_list:
        return

    model_name = os.getenv("GAIA_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    provider = os.getenv("GAIA_JUDGE_PROVIDER", DEFAULT_JUDGE_PROVIDER)
    temperature = float(os.getenv("GAIA_JUDGE_TEMPERATURE", DEFAULT_JUDGE_TEMPERATURE))

    judge_model = init_chat_model(
        model_name,
        model_provider=provider,
        temperature=temperature,
    )

    system_msg = SystemMessage(content=JUDGE_SYSTEM_PROMPT)

    for metrics in metrics_list:
        candidate = metrics.model_output or metrics.predicted_answer
        if not candidate or not metrics.ground_truth:
            continue

        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=metrics.query,
            model_output=candidate,
            ground_truth=metrics.ground_truth,
        )
        response = judge_model.invoke([system_msg, HumanMessage(content=user_prompt)])
        raw_text = _extract_content(response)
        decision = _normalize_decision(raw_text)

        metrics.judge_decision = decision
        metrics.judge_reason = raw_text
        metrics.judge_correct = decision == "CORRECT"


async def evaluate_accuracy_with_judge(metrics_list: List[ExperimentMetrics]) -> None:
    """Run the Gemini judge asynchronously without blocking main timings."""
    if not metrics_list:
        return
    await asyncio.to_thread(_run_judge, metrics_list)

