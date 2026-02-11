"""State models for speculation pipeline with parallel draft and target execution."""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


class Msg(BaseModel):
    """Message in conversation history."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None


@dataclass
class SpeculationState:
    """
    State for speculation pipeline with parallel draft and target models.
    
    Flow:
    1. kickoff → parallel: draft_plan + target_plan
    2. draft_plan → draft_exec (execute tools immediately)
    3. target_plan + draft_exec → verify_or_exec (check cache and decide)
    4. Loop or end
    """
    # Conversation history
    messages: List[Msg] = field(default_factory=list)
    
    # Control flow
    step: int = 0
    done: bool = False
    answer: Optional[str] = None
    
    # Draft model predictions
    draft_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    draft_ready: bool = False  # Flag: draft planning完成
    draft_exec_ready: bool = False  # Flag: draft execution完成
    
    # Target model decisions
    target_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    target_ready: bool = False  # Flag: target planning完成
    
    # Tool execution results cache
    # Key: (tool_name, normalized_args) -> Result
    draft_cache: Dict[Tuple[str, str], Any] = field(default_factory=dict)
    
    # Final results after verification
    verified_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics
    cache_hits: int = 0
    cache_misses: int = 0
    draft_tools_launched: int = 0

    # Verification metrics (for enhanced tracking)
    verified_by_ast: int = 0
    verified_by_embedding: int = 0
    both_failed: int = 0

    # Tool usage tracking (for statistics)
    draft_tool_names: List[str] = field(default_factory=list)
    target_tool_names: List[str] = field(default_factory=list)

    # Per-step verification details (for detailed results)
    verification_details: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    draft_plan_time: float = 0.0
    draft_exec_time: float = 0.0
    target_plan_time: float = 0.0
    verify_time: float = 0.0
    t0: float = field(default_factory=time.time)
    
    def reset_flags(self):
        """Reset flags and cache for next iteration.
        
        This clears the draft cache to ensure fresh predictions each step.
        Only metrics (cache_hits, cache_misses) are preserved across steps.
        """
        self.draft_ready = False
        self.draft_exec_ready = False
        self.target_ready = False
        self.draft_tool_calls = []
        self.target_tool_calls = []
        self.verified_results = []
        self.draft_cache = {}  # Clear cache for fresh predictions each step
    
    def can_verify(self) -> bool:
        """Check if both draft execution and target planning are ready."""
        return self.draft_exec_ready and self.target_ready
    
    def normalize_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Normalize tool arguments to create cache key."""
        import json
        try:
            return json.dumps(args, sort_keys=True)
        except:
            return str(args)
    
    def get_cache_key(self, tool_name: str, args: Dict[str, Any]) -> Tuple[str, str]:
        """Generate cache key for tool call."""
        return (tool_name, self.normalize_tool_call(tool_name, args))

