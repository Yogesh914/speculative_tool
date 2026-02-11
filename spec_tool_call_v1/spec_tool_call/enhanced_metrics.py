"""
Enhanced Metrics System

Combines metrics from both old and new implementations:
- Old version: Verification metrics (strict/fallback acceptance rates)
- New version: Speculation cache metrics (hits/misses/timing)

This module provides comprehensive tracking for speculation experiments,
including tool usage statistics, verification methods, and performance metrics.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ToolCallInfo:
    """Information about a single tool call."""
    tool_name: str
    args: Dict[str, Any]
    result: str
    success: bool
    execution_time: float = 0.0


@dataclass
class StepMetrics:
    """Metrics for a single reasoning step."""
    step_number: int

    # Draft model metrics
    draft_tools: List[ToolCallInfo] = field(default_factory=list)
    draft_plan_time: float = 0.0
    draft_exec_time: float = 0.0

    # Target model metrics
    target_tools: List[ToolCallInfo] = field(default_factory=list)
    target_plan_time: float = 0.0
    target_exec_time: float = 0.0

    # Verification metrics
    cache_hits: int = 0
    cache_misses: int = 0
    verified_by_ast: int = 0
    verified_by_embedding: int = 0
    verification_time: float = 0.0

    # Per-tool verification details
    verification_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExperimentMetrics:
    """Comprehensive metrics for an entire experiment run."""
    
    # Experiment metadata
    experiment_id: str
    timestamp: str
    dataset: str
    dataset_size: int
    
    # Model configuration
    actor_model: str
    spec_model: str
    model_provider: str
    actor_provider: str
    spec_provider: str
    speculation_enabled: bool
    
    # Query information
    query: str
    ground_truth: Optional[str] = None
    predicted_answer: Optional[str] = None  # Deprecated: retained for backward compatibility
    model_output: Optional[str] = None      # Raw assistant response
    
    # Judge evaluation
    judge_decision: Optional[str] = None    # "CORRECT" or "WRONG"
    judge_reason: Optional[str] = None      # Raw text returned by judge
    judge_correct: Optional[bool] = None
    
    # Overall execution metrics
    total_steps: int = 0
    total_time: float = 0.0
    success: bool = False
    
    # Step-by-step metrics
    steps: List[StepMetrics] = field(default_factory=list)
    
    # Aggregate cache metrics
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    total_draft_tools_launched: int = 0
    
    # Aggregate verification metrics
    total_verified_by_ast: int = 0
    total_verified_by_embedding: int = 0
    total_both_failed: int = 0
    
    # Tool usage statistics
    draft_tool_names: List[str] = field(default_factory=list)
    target_tool_names: List[str] = field(default_factory=list)
    
    # Time breakdown
    total_draft_plan_time: float = 0.0
    total_draft_exec_time: float = 0.0
    total_target_plan_time: float = 0.0
    total_target_exec_time: float = 0.0
    total_verification_time: float = 0.0

    def add_step(self, step: StepMetrics) -> None:
        """Add metrics for a step and update aggregates."""
        self.steps.append(step)

        # Update cache metrics
        self.total_cache_hits += step.cache_hits
        self.total_cache_misses += step.cache_misses
        self.total_draft_tools_launched += len(step.draft_tools)

        # Update verification metrics
        self.total_verified_by_ast += step.verified_by_ast
        self.total_verified_by_embedding += step.verified_by_embedding

        # Update tool names
        self.draft_tool_names.extend([t.tool_name for t in step.draft_tools])
        self.target_tool_names.extend([t.tool_name for t in step.target_tools])

        # Update time breakdown
        self.total_draft_plan_time += step.draft_plan_time
        self.total_draft_exec_time += step.draft_exec_time
        self.total_target_plan_time += step.target_plan_time
        self.total_target_exec_time += step.target_exec_time
        self.total_verification_time += step.verification_time

    def calculate_rates(self) -> Dict[str, float]:
        """Calculate acceptance and hit rates."""
        total_target_tools = self.total_cache_hits + self.total_cache_misses

        # Cache hit rate
        cache_hit_rate = (
            (self.total_cache_hits / total_target_tools * 100)
            if total_target_tools > 0 else 0.0
        )

        # Verification rates
        total_verified = self.total_verified_by_ast + self.total_verified_by_embedding
        total_verification_attempts = total_verified + self.total_both_failed

        strict_only_rate = (
            (self.total_verified_by_ast / total_verification_attempts * 100)
            if total_verification_attempts > 0 else 0.0
        )

        with_fallback_rate = (
            (total_verified / total_verification_attempts * 100)
            if total_verification_attempts > 0 else 0.0
        )

        improvement = with_fallback_rate - strict_only_rate

        return {
            "cache_hit_rate": cache_hit_rate,
            "strict_only_rate": strict_only_rate,
            "with_fallback_rate": with_fallback_rate,
            "fallback_improvement": improvement,
        }

    def calculate_time_savings(self) -> Dict[str, float]:
        """Calculate time savings from speculation."""
        # Target-only time (if no speculation)
        target_only_time = (
            self.total_target_plan_time +
            self.total_target_exec_time +
            (self.total_cache_misses * 0.5)  # Estimate for missed tools
        )

        # With-speculation time (actual)
        with_speculation_time = self.total_time

        # Time saved
        time_saved = max(0, target_only_time - with_speculation_time)
        percent_saved = (
            (time_saved / target_only_time * 100)
            if target_only_time > 0 else 0.0
        )

        return {
            "target_only_time": target_only_time,
            "with_speculation_time": with_speculation_time,
            "time_saved": time_saved,
            "percent_saved": percent_saved,
        }

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get statistics about tool usage."""
        from collections import Counter

        draft_counts = Counter(self.draft_tool_names)
        target_counts = Counter(self.target_tool_names)

        # Tool diversity (number of unique tools)
        draft_unique = len(set(self.draft_tool_names))
        target_unique = len(set(self.target_tool_names))

        # Tool overlap (how many draft tools were also used by target)
        draft_set = set(self.draft_tool_names)
        target_set = set(self.target_tool_names)
        overlap = len(draft_set & target_set)

        return {
            "draft_tool_counts": dict(draft_counts),
            "target_tool_counts": dict(target_counts),
            "draft_unique_tools": draft_unique,
            "target_unique_tools": target_unique,
            "tool_overlap": overlap,
            "total_draft_calls": len(self.draft_tool_names),
            "total_target_calls": len(self.target_tool_names),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        rates = self.calculate_rates()
        time_savings = self.calculate_time_savings()
        tool_stats = self.get_tool_usage_stats()

        return {
            "metadata": {
                "experiment_id": self.experiment_id,
                "timestamp": self.timestamp,
                "dataset": self.dataset,
                "dataset_size": self.dataset_size,
                "actor_model": self.actor_model,
                "spec_model": self.spec_model,
                "model_provider": self.model_provider,
                "actor_provider": self.actor_provider,
                "spec_provider": self.spec_provider,
                "speculation_enabled": self.speculation_enabled,
            },
            "query": {
                "question": self.query,
                "ground_truth": self.ground_truth,
                "predicted": self.predicted_answer,
                "model_output": self.model_output or self.predicted_answer,
                "judge": {
                    "decision": self.judge_decision,
                    "raw": self.judge_reason,
                    "correct": self.judge_correct,
                },
                "correct": self.judge_correct,
            },
            "execution": {
                "total_steps": self.total_steps,
                "total_time": self.total_time,
                "success": self.success,
            },
            "cache_metrics": {
                "hits": self.total_cache_hits,
                "misses": self.total_cache_misses,
                "launched": self.total_draft_tools_launched,
                "hit_rate": rates["cache_hit_rate"],
            },
            "verification_metrics": {
                "strict_ast_verified": self.total_verified_by_ast,
                "embedding_fallback_verified": self.total_verified_by_embedding,
                "both_failed": self.total_both_failed,
                "strict_only_rate": rates["strict_only_rate"],
                "with_fallback_rate": rates["with_fallback_rate"],
                "fallback_improvement": rates["fallback_improvement"],
            },
            "time_breakdown": {
                "draft_plan_time": self.total_draft_plan_time,
                "draft_exec_time": self.total_draft_exec_time,
                "target_plan_time": self.total_target_plan_time,
                "target_exec_time": self.total_target_exec_time,
                "verification_time": self.total_verification_time,
                "total": self.total_time,
            },
            "time_savings": time_savings,
            "tool_statistics": tool_stats,
            "steps": [
                {
                    "step": step.step_number,
                    "draft_tools": [
                        {
                            "tool": t.tool_name,
                            "args": t.args,
                            "success": t.success,
                            "time": t.execution_time,
                        }
                        for t in step.draft_tools
                    ],
                    "target_tools": [
                        {
                            "tool": t.tool_name,
                            "args": t.args,
                            "success": t.success,
                            "time": t.execution_time,
                        }
                        for t in step.target_tools
                    ],
                    "cache_hits": step.cache_hits,
                    "cache_misses": step.cache_misses,
                    "verification_details": step.verification_details,
                    "timings": {
                        "draft_plan": step.draft_plan_time,
                        "draft_exec": step.draft_exec_time,
                        "target_plan": step.target_plan_time,
                        "target_exec": step.target_exec_time,
                        "verification": step.verification_time,
                    },
                }
                for step in self.steps
            ],
        }

    def print_summary(self) -> None:
        """Print a human-readable summary."""
        rates = self.calculate_rates()
        time_savings = self.calculate_time_savings()
        tool_stats = self.get_tool_usage_stats()

        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)

        print(f"\n📋 Metadata:")
        print(f"  Experiment ID: {self.experiment_id}")
        print(f"  Dataset: {self.dataset}")
        print(f"  Actor: {self.actor_model} ({self.actor_provider})")
        print(f"  Spec: {self.spec_model} ({self.spec_provider})")
        print(f"  Speculation: {'ENABLED' if self.speculation_enabled else 'DISABLED'}")

        print(f"\n❓ Query:")
        print(f"  Question: {self.query[:100]}...")
        if self.ground_truth:
            print(f"  Ground Truth: {self.ground_truth}")
        if self.model_output or self.predicted_answer:
            output = self.model_output or self.predicted_answer
            print(f"  Model Output: {output}")
        if self.judge_decision:
            icon = "✅" if self.judge_decision.upper().startswith("CORRECT") else "❌"
            print(f"  Judge: {icon} {self.judge_decision} ({self.judge_reason})")

        print(f"\n⚡ Execution:")
        print(f"  Total Steps: {self.total_steps}")
        print(f"  Total Time: {self.total_time:.2f}s")
        print(f"  Success: {'✅' if self.success else '❌'}")

        print(f"\n🎯 Cache Performance:")
        print(f"  Hits: {self.total_cache_hits}")
        print(f"  Misses: {self.total_cache_misses}")
        print(f"  Launched: {self.total_draft_tools_launched}")
        print(f"  Hit Rate: {rates['cache_hit_rate']:.1f}%")

        print(f"\n✓ Verification:")
        print(f"  Strict AST: {self.total_verified_by_ast}")
        print(f"  Embedding Fallback: {self.total_verified_by_embedding}")
        print(f"  Both Failed: {self.total_both_failed}")
        print(f"  Strict Only Rate: {rates['strict_only_rate']:.1f}%")
        print(f"  With Fallback Rate: {rates['with_fallback_rate']:.1f}%")
        print(f"  Improvement: +{rates['fallback_improvement']:.1f}%")

        print(f"\n⏱️  Time Savings:")
        print(f"  Target-Only Time: {time_savings['target_only_time']:.2f}s")
        print(f"  With Speculation: {time_savings['with_speculation_time']:.2f}s")
        print(f"  Time Saved: {time_savings['time_saved']:.2f}s ({time_savings['percent_saved']:.1f}%)")

        print(f"\n🔧 Tool Usage:")
        print(f"  Draft Calls: {tool_stats['total_draft_calls']} ({tool_stats['draft_unique_tools']} unique)")
        print(f"  Target Calls: {tool_stats['total_target_calls']} ({tool_stats['target_unique_tools']} unique)")
        print(f"  Tool Overlap: {tool_stats['tool_overlap']}")

        if tool_stats['draft_tool_counts']:
            print(f"\n  Draft Tools Used:")
            for tool, count in sorted(tool_stats['draft_tool_counts'].items(), key=lambda x: -x[1])[:5]:
                print(f"    {tool}: {count}")

        if tool_stats['target_tool_counts']:
            print(f"\n  Target Tools Used:")
            for tool, count in sorted(tool_stats['target_tool_counts'].items(), key=lambda x: -x[1])[:5]:
                print(f"    {tool}: {count}")

        print("\n" + "="*80)


def create_experiment_metrics(
    experiment_id: str,
    query: str,
    ground_truth: Optional[str] = None,
) -> ExperimentMetrics:
    """Create a new ExperimentMetrics instance with metadata from config."""
    from .config import config
    from datetime import datetime

    return ExperimentMetrics(
        experiment_id=experiment_id,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        dataset=config.dataset,
        dataset_size=config.dataset_size,
        actor_model=config.actor_model,
        spec_model=config.spec_model,
        model_provider=config.model_provider,
        actor_provider=config.actor_provider,
        spec_provider=config.spec_provider,
        speculation_enabled=config.enable_speculation,
        query=query,
        ground_truth=ground_truth,
    )
