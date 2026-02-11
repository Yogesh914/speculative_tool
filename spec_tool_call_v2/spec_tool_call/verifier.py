"""
Tool call verification strategies for speculative execution.

This module defines different strategies for matching actor tool calls
with speculative predictions to determine if cached results can be reused.
"""

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import json


class ToolVerifier(ABC):
    """Base class for tool verification strategies."""
    
    @abstractmethod
    def verify(
        self,
        actor_tool_name: str,
        actor_tool_args: Dict[str, Any],
        spec_tool_name: str,
        spec_tool_args: Dict[str, Any]
    ) -> tuple[bool, float, str]:
        """
        Verify if a speculative prediction matches the actor's request.
        
        Args:
            actor_tool_name: Tool name requested by actor
            actor_tool_args: Arguments requested by actor
            spec_tool_name: Tool name in speculative prediction
            spec_tool_args: Arguments in speculative prediction
            
        Returns:
            Tuple of (is_match, similarity_score, reason)
            - is_match: Whether the prediction can be used
            - similarity_score: Confidence score (0.0 to 1.0)
            - reason: Human-readable explanation
        """
        pass


class ExactMatchVerifier(ToolVerifier):
    """
    Exact match verifier - requires perfect match of tool name and all arguments.
    
    This is the most conservative strategy:
    - Tool names must match exactly
    - All argument keys must match
    - All argument values must match exactly (after normalization)
    
    Pros: No false positives, always correct results
    Cons: Low cache hit rate, may miss semantically equivalent queries
    """
    
    def verify(
        self,
        actor_tool_name: str,
        actor_tool_args: Dict[str, Any],
        spec_tool_name: str,
        spec_tool_args: Dict[str, Any]
    ) -> tuple[bool, float, str]:
        # Tool names must match
        if actor_tool_name != spec_tool_name:
            return False, 0.0, f"Tool mismatch: {spec_tool_name} != {actor_tool_name}"
        
        # Normalize and compare arguments
        actor_normalized = self._normalize_args(actor_tool_args)
        spec_normalized = self._normalize_args(spec_tool_args)
        
        # Check if arguments match exactly
        if actor_normalized == spec_normalized:
            return True, 1.0, "Exact match"
        
        # Find differences
        diff_keys = set(actor_normalized.keys()) ^ set(spec_normalized.keys())
        if diff_keys:
            return False, 0.0, f"Different argument keys: {diff_keys}"
        
        # Find mismatched values
        mismatches = []
        for key in actor_normalized:
            if actor_normalized[key] != spec_normalized[key]:
                mismatches.append(key)
        
        if mismatches:
            return False, 0.0, f"Mismatched arguments: {', '.join(mismatches)}"
        
        return True, 1.0, "Exact match"
    
    def _normalize_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize arguments for comparison."""
        normalized = {}
        for key, value in args.items():
            if isinstance(value, str):
                # Strip whitespace, lowercase for case-insensitive comparison
                normalized[key] = value.strip()
            elif isinstance(value, (list, dict)):
                # Convert to JSON string for comparison
                normalized[key] = json.dumps(value, sort_keys=True)
            else:
                normalized[key] = value
        return normalized


class ToolNameOnlyVerifier(ToolVerifier):
    """
    Tool name only verifier - only checks if tool names match.
    
    This is the current implementation (broken):
    - Only verifies tool name matches
    - Ignores all arguments
    
    Pros: Highest cache hit rate
    Cons: Returns wrong results, confuses actor, degrades performance
    
    ⚠️ WARNING: This strategy is known to degrade performance!
    Use only for comparison/debugging purposes.
    """
    
    def verify(
        self,
        actor_tool_name: str,
        actor_tool_args: Dict[str, Any],
        spec_tool_name: str,
        spec_tool_args: Dict[str, Any]
    ) -> tuple[bool, float, str]:
        if actor_tool_name == spec_tool_name:
            return True, 1.0, "Tool name match (args ignored)"
        return False, 0.0, f"Tool mismatch: {spec_tool_name} != {actor_tool_name}"


class JaccardSimilarityVerifier(ToolVerifier):
    """
    Jaccard similarity verifier - uses normalized token-based Jaccard similarity.
    
    This is a balanced strategy that handles word order and minor variations:
    - Tool names must match exactly
    - Arguments are normalized (lowercase, remove stopwords)
    - Jaccard similarity computed on token sets
    - Accepts if similarity >= threshold
    
    Pros: Handles word order, typos, and minor variations; Fast (~1ms)
    Cons: May accept semantically different queries with similar words
    
    Recommended for search queries where word order doesn't matter much.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize verifier with similarity threshold.
        
        Args:
            threshold: Minimum Jaccard similarity to accept (0.0 to 1.0)
                      Recommended: 0.7 for balanced flexibility
        """
        self.threshold = threshold
        # Common stopwords to remove (expand as needed)
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are'
        }
    
    def verify(
        self,
        actor_tool_name: str,
        actor_tool_args: Dict[str, Any],
        spec_tool_name: str,
        spec_tool_args: Dict[str, Any]
    ) -> tuple[bool, float, str]:
        # Tool names must match exactly
        if actor_tool_name != spec_tool_name:
            return False, 0.0, f"Tool mismatch: {spec_tool_name} != {actor_tool_name}"
        
        # For non-string arguments, require exact match
        if not self._has_string_args(actor_tool_args) or not self._has_string_args(spec_tool_args):
            # Fall back to exact comparison for non-string args
            if actor_tool_args == spec_tool_args:
                return True, 1.0, "Exact match (non-string args)"
            return False, 0.0, "Non-string args don't match exactly"
        
        # Compare arguments using Jaccard similarity
        similarity = self._compute_args_similarity(actor_tool_args, spec_tool_args)
        
        if similarity >= self.threshold:
            return True, similarity, f"Jaccard similarity {similarity:.3f} >= {self.threshold}"
        else:
            return False, similarity, f"Jaccard similarity {similarity:.3f} < {self.threshold}"
    
    def _has_string_args(self, args: Dict[str, Any]) -> bool:
        """Check if arguments contain string values."""
        return any(isinstance(v, str) for v in args.values())
    
    def _normalize_string(self, text: str) -> set:
        """
        Normalize a string and return set of tokens.
        
        Steps:
        1. Lowercase
        2. Split into words
        3. Remove stopwords
        4. Remove punctuation
        5. Return set of tokens
        """
        if not isinstance(text, str):
            return {str(text)}
        
        # Lowercase and split
        text = text.lower()
        
        # Simple tokenization (split on whitespace and common punctuation)
        import re
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stopwords]
        
        return set(tokens)
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """
        Compute Jaccard similarity between two sets.
        
        Jaccard = |intersection| / |union|
        
        Returns:
            Float between 0.0 (no overlap) and 1.0 (identical)
        """
        if not set1 and not set2:
            return 1.0  # Both empty = identical
        
        if not set1 or not set2:
            return 0.0  # One empty = no similarity
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_args_similarity(
        self, 
        args1: Dict[str, Any], 
        args2: Dict[str, Any]
    ) -> float:
        """
        Compute overall similarity between two argument dictionaries.
        
        Strategy:
        1. Check if keys match (if not, return 0)
        2. For each key, compute Jaccard similarity of values
        3. Return average similarity across all keys
        """
        # Keys must match
        keys1 = set(args1.keys())
        keys2 = set(args2.keys())
        
        if keys1 != keys2:
            # Different keys = different arguments
            return 0.0
        
        # Compute similarity for each key
        similarities = []
        
        for key in keys1:
            val1 = args1[key]
            val2 = args2[key]
            
            # Handle different types
            if type(val1) != type(val2):
                similarities.append(0.0)
                continue
            
            if isinstance(val1, str):
                # String comparison using Jaccard
                tokens1 = self._normalize_string(val1)
                tokens2 = self._normalize_string(val2)
                sim = self._jaccard_similarity(tokens1, tokens2)
                similarities.append(sim)
            
            elif isinstance(val1, (int, float, bool)):
                # Exact match for numbers/booleans
                similarities.append(1.0 if val1 == val2 else 0.0)
            
            elif isinstance(val1, (list, dict)):
                # Exact match for complex types (could be improved)
                similarities.append(1.0 if val1 == val2 else 0.0)
            
            else:
                # Unknown type - exact match
                similarities.append(1.0 if val1 == val2 else 0.0)
        
        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0


class NoSpeculationVerifier(ToolVerifier):
    """
    No speculation verifier - always returns False to disable speculation.
    
    Use this to run baseline comparisons without speculation overhead.
    """
    
    def verify(
        self,
        actor_tool_name: str,
        actor_tool_args: Dict[str, Any],
        spec_tool_name: str,
        spec_tool_args: Dict[str, Any]
    ) -> tuple[bool, float, str]:
        return False, 0.0, "Speculation disabled"


# Factory for creating verifiers
def create_verifier(strategy: str = "exact", threshold: float = 0.7) -> ToolVerifier:
    """
    Create a verifier instance based on strategy name.
    
    Args:
        strategy: One of "exact", "jaccard", "tool_name_only", "none"
        threshold: Similarity threshold for jaccard strategy (0.0 to 1.0, default 0.5)
        
    Returns:
        ToolVerifier instance
    """
    strategies = {
        "exact": ExactMatchVerifier,
        "jaccard": JaccardSimilarityVerifier,
        "tool_name_only": ToolNameOnlyVerifier,
        "none": NoSpeculationVerifier,
    }
    
    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {', '.join(strategies.keys())}"
        )
    
    # Jaccard verifier takes a threshold parameter
    if strategy == "jaccard":
        return strategies[strategy](threshold=threshold)
    else:
        return strategies[strategy]()

