"""
Embedding Similarity Module - Computes semantic similarity between tool calls.

This module provides a fallback mechanism for the strict AST verification by using
embedding-based cosine similarity to check if tool calls are semantically equivalent.

Supports two embedding methods:
1. Gemini API (gemini-embedding-001) - Cloud-based
2. Gemma local model (google/embeddinggemma-300m) - Local via sentence-transformers
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple, Literal
import warnings

warnings.filterwarnings('ignore')

# Try to import dependencies
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: langchain-google-genai not available. Gemini embeddings will not work.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Local embeddings will not work.")


# Global model cache to avoid reloading
_gemma_model = None
_gemini_client = None


def get_gemini_embeddings():
    """Get or create Gemini embeddings model."""
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_AVAILABLE:
            raise ImportError("langchain-google-genai package not installed")
        
        # Get API key from config or environment
        try:
            from .config import config
            api_key = (
                config.embedding_api_key
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
            )
        except:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY not found in environment. "
                "Please set it in your .env file or environment variables."
            )
        
        _gemini_client = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
    return _gemini_client


def get_gemma_model():
    """Get or create Gemma model (cached for performance)."""
    global _gemma_model
    if _gemma_model is None:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers package not installed")
        print("Loading Gemma model (google/embeddinggemma-300m)... This may take a moment.")
        _gemma_model = SentenceTransformer("google/embeddinggemma-300m")
    return _gemma_model


def tool_call_to_string(tool_call: Dict[str, Any]) -> str:
    """
    Convert a tool call to a string representation for embedding.

    Format: "tool_name(param1=value1, param2=value2, ...)"

    Args:
        tool_call: Dictionary with 'name' and 'args' keys

    Returns:
        String representation of the tool call
    """
    name = tool_call.get("name", "unknown")
    args = tool_call.get("args", {})

    # Sort parameters for consistency
    param_strings = []
    for key in sorted(args.keys()):
        value = args[key]
        # Truncate long values
        value_str = str(value)
        if len(value_str) > 100:
            value_str = value_str[:97] + "..."
        param_strings.append(f"{key}={value_str}")

    params = ", ".join(param_strings)
    return f"{name}({params})"


def tool_sequence_to_string(tool_calls: List[Dict[str, Any]]) -> str:
    """
    Convert a sequence of tool calls to a string representation.

    Args:
        tool_calls: List of tool call dictionaries

    Returns:
        String representation of the sequence
    """
    call_strings = [tool_call_to_string(tc) for tc in tool_calls]
    return " -> ".join(call_strings)


def get_embedding_gemini(text: str) -> np.ndarray:
    """
    Get embedding using Gemini API via LangChain.

    Args:
        text: Text to embed

    Returns:
        Embedding vector as numpy array
    """
    if not GEMINI_AVAILABLE:
        raise ImportError(
            "langchain-google-genai package not installed. "
            "Install with: pip install langchain-google-genai"
        )

    embeddings = get_gemini_embeddings()

    try:
        embedding = embeddings.embed_query(text)
        return np.array(embedding)
    except Exception as e:
        raise RuntimeError(f"Failed to get Gemini embedding: {e}")


def get_embedding_gemma(text: str) -> np.ndarray:
    """
    Get embedding using local Gemma model.

    Args:
        text: Text to embed

    Returns:
        Embedding vector as numpy array
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "sentence-transformers package not installed. "
            "Install with: pip install sentence-transformers"
        )

    model = get_gemma_model()

    try:
        # Use encode_query for better semantic understanding
        embedding = model.encode(text)
        return embedding
    except Exception as e:
        raise RuntimeError(f"Failed to get Gemma embedding: {e}")


def get_embedding(
    text: str,
    method: Literal["gemini", "gemma"] = "gemini"
) -> np.ndarray:
    """
    Get embedding using specified method.

    Args:
        text: Text to embed
        method: Either "gemini" (API) or "gemma" (local)

    Returns:
        Embedding vector as numpy array
    """
    if method == "gemini":
        return get_embedding_gemini(text)
    elif method == "gemma":
        return get_embedding_gemma(text)
    else:
        raise ValueError(f"Unknown embedding method: {method}. Use 'gemini' or 'gemma'")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    # Normalize vectors
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)

    # Compute dot product
    similarity = np.dot(vec1_norm, vec2_norm)

    return float(similarity)


def compute_similarity(
    draft_calls: List[Dict[str, Any]],
    target_call: Dict[str, Any],
    threshold: float = 0.5,
    method: Literal["gemini", "gemma"] = "gemini",
    verbose: bool = False
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Check if draft tool sequence is semantically similar to target tool call.

    Args:
        draft_calls: List of tool calls from draft model
        target_call: Single tool call from target model
        threshold: Similarity threshold (0-1, default 0.5)
        method: Embedding method to use ("gemini" or "gemma")
        verbose: If True, print detailed information

    Returns:
        Tuple of (is_similar, similarity_score, details)
        - is_similar: True if similarity >= threshold
        - similarity_score: Cosine similarity between 0 and 1
        - details: Dictionary with additional information
    """
    # Convert to strings
    draft_str = tool_sequence_to_string(draft_calls)
    target_str = tool_call_to_string(target_call)

    if verbose:
        print(f"\n{'='*80}")
        print("EMBEDDING SIMILARITY COMPUTATION")
        print(f"{'='*80}")
        print(f"Method: {method.upper()}")
        print(f"Threshold: {threshold}")
        print(f"\nDraft tool sequence:")
        print(f"  {draft_str}")
        print(f"\nTarget tool call:")
        print(f"  {target_str}")

    # Get embeddings
    try:
        if verbose:
            print(f"\nGenerating embeddings using {method}...")

        draft_embedding = get_embedding(draft_str, method=method)
        target_embedding = get_embedding(target_str, method=method)

        if verbose:
            print(f"✅ Draft embedding shape: {draft_embedding.shape}")
            print(f"   First 5 dims: [{', '.join([f'{x:.4f}' for x in draft_embedding[:5]])}...]")
            print(f"✅ Target embedding shape: {target_embedding.shape}")
            print(f"   First 5 dims: [{', '.join([f'{x:.4f}' for x in target_embedding[:5]])}...]")

    except Exception as e:
        if verbose:
            print(f"❌ Error generating embeddings: {e}")
        return False, 0.0, {
            "error": str(e),
            "draft_string": draft_str,
            "target_string": target_str
        }

    # Compute similarity
    similarity = cosine_similarity(draft_embedding, target_embedding)
    is_similar = similarity >= threshold

    if verbose:
        print(f"\n{'='*40}")
        print(f"Cosine Similarity: {similarity:.4f}")
        print(f"Threshold:         {threshold:.4f}")
        print(f"Result: {'✅ PASS' if is_similar else '❌ FAIL'} (similarity {'≥' if is_similar else '<'} threshold)")
        print(f"{'='*40}")

    details = {
        "draft_string": draft_str,
        "target_string": target_str,
        "similarity_score": similarity,
        "threshold": threshold,
        "method": method,
        "embedding_dims": len(draft_embedding),
        "draft_embedding_sample": draft_embedding[:10].tolist(),  # First 10 dims
        "target_embedding_sample": target_embedding[:10].tolist()
    }

    return is_similar, similarity, details


def check_embedding_equivalence(
    draft_sequence: List[Dict[str, Any]],
    target_call: Dict[str, Any],
    threshold: float = 0.5,
    method: Literal["gemini", "gemma"] = "gemini",
    verbose: bool = False
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check if draft sequence is equivalent to target using embeddings.

    This is the main function to use as a fallback for AST verification.

    Args:
        draft_sequence: List of tool calls from draft model
        target_call: Single tool call from target model
        threshold: Similarity threshold (default 0.5)
        method: Embedding method ("gemini" or "gemma")
        verbose: If True, print detailed information

    Returns:
        Tuple of (is_equivalent, errors, details)
    """
    errors = []

    # Validate inputs
    if not draft_sequence or not isinstance(draft_sequence, list):
        errors.append("Invalid draft sequence: must be non-empty list")
        return False, errors, {}

    if not target_call or not isinstance(target_call, dict):
        errors.append("Invalid target call: must be dict")
        return False, errors, {}

    # Check similarity
    try:
        is_similar, similarity, details = compute_similarity(
            draft_sequence, target_call, threshold, method, verbose=verbose
        )

        if not is_similar:
            errors.append(
                f"Semantic similarity too low: {similarity:.3f} < {threshold} "
                f"(draft: '{details['draft_string']}', target: '{details['target_string']}')"
            )

        return is_similar, errors, details

    except Exception as e:
        if verbose:
            print(f"❌ Embedding similarity check failed: {e}")
        errors.append(f"Embedding similarity check failed: {e}")
        return False, errors, {"error": str(e)}


if __name__ == "__main__":
    # Example usage
    print("Testing Embedding Similarity Module")
    print("=" * 80)

    # Test cases
    draft1 = [
        {"name": "search_web", "args": {"query": "AI agents"}},
        {"name": "extract_key_points", "args": {"text": "results"}},
    ]

    target1 = {
        "name": "research_topic",
        "args": {"topic": "AI agents"}
    }

    draft2 = [
        {"name": "calculate", "args": {"expression": "2 + 2"}},
    ]

    target2 = {
        "name": "add_numbers",
        "args": {"a": 2, "b": 2}
    }

    # Test with Gemini (if available)
    if GEMINI_AVAILABLE and os.environ.get("GOOGLE_API_KEY"):
        print("\n1. Testing with Gemini embeddings:")
        is_sim, score, details = compute_similarity(draft1, target1, threshold=0.5, method="gemini")
        print(f"   Draft: {details['draft_string']}")
        print(f"   Target: {details['target_string']}")
        print(f"   Similarity: {score:.3f}, Is similar? {is_sim}")

        is_sim2, score2, details2 = compute_similarity(draft2, target2, threshold=0.5, method="gemini")
        print(f"\n   Draft: {details2['draft_string']}")
        print(f"   Target: {details2['target_string']}")
        print(f"   Similarity: {score2:.3f}, Is similar? {is_sim2}")
    else:
        print("\n1. Gemini embeddings not available (missing API key or package)")

    # Test with Gemma (if available)
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n2. Testing with Gemma local embeddings:")
        is_sim, score, details = compute_similarity(draft1, target1, threshold=0.5, method="gemma")
        print(f"   Draft: {details['draft_string']}")
        print(f"   Target: {details['target_string']}")
        print(f"   Similarity: {score:.3f}, Is similar? {is_sim}")
    else:
        print("\n2. Gemma embeddings not available (missing package)")

    print("\n" + "=" * 80)
    print("Test complete!")
