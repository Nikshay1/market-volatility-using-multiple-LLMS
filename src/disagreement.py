"""
Disagreement metrics for measuring agent divergence.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

from . import config


# Global embedding model (lazy loaded)
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Get the sentence transformer model for embeddings.
    Uses lazy loading to avoid loading the model until needed.
    
    Returns:
        SentenceTransformer model
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedding_model


def calculate_scalar_disagreement(scores: List[float]) -> float:
    """
    Calculate scalar disagreement as the standard deviation of scores.
    
    This measures how far apart the agents' numerical scores are.
    Higher values indicate more disagreement.
    
    Args:
        scores: List of scores from each agent (typically 2 scores)
        
    Returns:
        Standard deviation of the scores
    """
    if not scores or len(scores) < 2:
        return 0.0
    
    scores_array = np.array(scores, dtype=np.float64)
    return float(np.std(scores_array))


def calculate_semantic_divergence(reasonings: List[str]) -> float:
    """
    Calculate semantic divergence as 1 - cosine_similarity of reasoning embeddings.
    
    This measures how semantically different the agents' reasoning is.
    Higher values indicate more divergent thinking.
    
    Args:
        reasonings: List of reasoning strings from each agent
        
    Returns:
        Semantic divergence (0 = identical, 2 = opposite)
    """
    if not reasonings or len(reasonings) < 2:
        return 0.0
    
    # Filter out empty reasonings
    valid_reasonings = [r for r in reasonings if r and r.strip()]
    if len(valid_reasonings) < 2:
        return 0.0
    
    # Get embeddings
    model = get_embedding_model()
    embeddings = model.encode(valid_reasonings[:2])  # Use first two reasonings
    
    # Calculate cosine similarity
    emb1 = embeddings[0]
    emb2 = embeddings[1]
    
    # Normalize
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
    
    # Divergence = 1 - similarity
    divergence = 1.0 - cosine_sim
    
    return float(divergence)


def calculate_combined_disagreement(
    scores: List[float],
    reasonings: List[str],
    scalar_weight: float = 0.5
) -> float:
    """
    Calculate a combined disagreement score.
    
    Args:
        scores: List of scores from each agent
        reasonings: List of reasoning strings from each agent
        scalar_weight: Weight for scalar component (0-1)
        
    Returns:
        Combined disagreement score
    """
    scalar = calculate_scalar_disagreement(scores)
    semantic = calculate_semantic_divergence(reasonings)
    
    # Normalize scalar to [0, 1] range (max std for [-1,1] is 1)
    scalar_normalized = min(scalar, 1.0)
    
    # Normalize semantic to [0, 1] range (max is 2)
    semantic_normalized = min(semantic / 2.0, 1.0)
    
    combined = scalar_weight * scalar_normalized + (1 - scalar_weight) * semantic_normalized
    
    return float(combined)


def get_embedding(text: str) -> np.ndarray:
    """
    Get embedding vector for a single text.
    
    Args:
        text: Input text string
        
    Returns:
        Embedding vector as numpy array
    """
    model = get_embedding_model()
    return model.encode(text)


if __name__ == "__main__":
    # Test the disagreement metrics
    print("Testing scalar disagreement...")
    test_scores = [0.5, -0.3]
    scalar_d = calculate_scalar_disagreement(test_scores)
    print(f"Scores: {test_scores}")
    print(f"Scalar Disagreement: {scalar_d:.4f}")
    
    print("\nTesting semantic divergence...")
    test_reasonings = [
        "The market fundamentals look strong with solid earnings growth and reasonable valuations.",
        "Headlines are causing excessive fear, creating a contrarian buying opportunity."
    ]
    semantic_d = calculate_semantic_divergence(test_reasonings)
    print(f"Reasoning 1: {test_reasonings[0][:50]}...")
    print(f"Reasoning 2: {test_reasonings[1][:50]}...")
    print(f"Semantic Divergence: {semantic_d:.4f}")
    
    print("\nTesting combined disagreement...")
    combined_d = calculate_combined_disagreement(test_scores, test_reasonings)
    print(f"Combined Disagreement: {combined_d:.4f}")
