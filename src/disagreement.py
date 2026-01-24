"""
Disagreement metrics for Agentic Dissonance v2.

Measures belief dispersion using:
- Confidence-weighted variance
- Semantic divergence across agents
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from . import config
from .aggregator import compute_confidence_weighted_variance, compute_confidence_weighted_mean


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


def calculate_disagreement_conf(
    agent_outputs: List[Dict[str, Any]]
) -> float:
    """
    Calculate confidence-weighted disagreement (variance).
    
    D_conf = Σ(cᵢ × (sᵢ - μ)²) / Σ(cᵢ)
    
    Args:
        agent_outputs: List of agent output dicts with 'score' and 'confidence'
        
    Returns:
        Confidence-weighted variance (disagreement signal)
    """
    if not agent_outputs:
        return 0.0
    
    scores = [o.get('score', 0.0) for o in agent_outputs]
    confidences = [o.get('confidence', 0.5) for o in agent_outputs]
    
    return compute_confidence_weighted_variance(scores, confidences)


def calculate_mean_score(
    agent_outputs: List[Dict[str, Any]]
) -> float:
    """
    Calculate confidence-weighted mean score.
    
    μ = Σ(cᵢ × sᵢ) / Σ(cᵢ)
    
    Args:
        agent_outputs: List of agent output dicts
        
    Returns:
        Confidence-weighted mean score
    """
    if not agent_outputs:
        return 0.0
    
    scores = [o.get('score', 0.0) for o in agent_outputs]
    confidences = [o.get('confidence', 0.5) for o in agent_outputs]
    
    return compute_confidence_weighted_mean(scores, confidences)


def calculate_avg_confidence(
    agent_outputs: List[Dict[str, Any]]
) -> float:
    """
    Calculate average confidence across agents.
    
    Args:
        agent_outputs: List of agent output dicts
        
    Returns:
        Average confidence
    """
    if not agent_outputs:
        return 0.0
    
    confidences = [o.get('confidence', 0.5) for o in agent_outputs]
    return float(np.mean(confidences))


def calculate_semantic_divergence(
    reasonings: List[str]
) -> float:
    """
    Calculate semantic divergence as average pairwise (1 - cosine_similarity).
    
    For 4 agents, computes all pairwise divergences and averages.
    
    Args:
        reasonings: List of reasoning strings from each agent
        
    Returns:
        Average semantic divergence (0 = identical, 2 = opposite)
    """
    if not reasonings or len(reasonings) < 2:
        return 0.0
    
    # Filter out empty reasonings
    valid_reasonings = [r for r in reasonings if r and r.strip()]
    if len(valid_reasonings) < 2:
        return 0.0
    
    # Get embeddings
    model = get_embedding_model()
    embeddings = model.encode(valid_reasonings)
    
    # Compute all pairwise divergences
    divergences = []
    n = len(embeddings)
    
    for i in range(n):
        for j in range(i + 1, n):
            emb_i = embeddings[i]
            emb_j = embeddings[j]
            
            # Normalize
            norm_i = np.linalg.norm(emb_i)
            norm_j = np.linalg.norm(emb_j)
            
            if norm_i == 0 or norm_j == 0:
                continue
            
            cosine_sim = np.dot(emb_i, emb_j) / (norm_i * norm_j)
            divergence = 1.0 - cosine_sim
            divergences.append(divergence)
    
    return float(np.mean(divergences)) if divergences else 0.0


def calculate_all_metrics(
    agent_outputs: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate all disagreement metrics from agent outputs.
    
    Args:
        agent_outputs: List of agent output dicts
        
    Returns:
        Dictionary with all metrics
    """
    # Extract reasonings for semantic analysis
    reasonings = [o.get('reasoning', '') for o in agent_outputs]
    
    # Core metrics
    disagreement_conf = calculate_disagreement_conf(agent_outputs)
    mean_score = calculate_mean_score(agent_outputs)
    avg_confidence = calculate_avg_confidence(agent_outputs)
    
    # Semantic metric (optional, can be slow)
    try:
        semantic_divergence = calculate_semantic_divergence(reasonings)
    except Exception as e:
        print(f"Warning: Could not compute semantic divergence: {e}")
        semantic_divergence = 0.0
    
    # Individual agent scores
    metrics = {
        "disagreement_conf": disagreement_conf,
        "mean_score": mean_score,
        "avg_confidence": avg_confidence,
        "semantic_divergence": semantic_divergence
    }
    
    # Add individual agent data
    agent_names = ["fundamental", "sentiment", "technical", "macro"]
    for i, output in enumerate(agent_outputs):
        if i < len(agent_names):
            name = agent_names[i]
            metrics[f"score_{name}"] = output.get("score", 0.0)
            metrics[f"confidence_{name}"] = output.get("confidence", 0.5)
    
    return metrics


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


# Legacy compatibility functions

def calculate_scalar_disagreement(scores: List[float]) -> float:
    """
    Calculate scalar disagreement as standard deviation of scores.
    
    DEPRECATED: Use calculate_disagreement_conf for confidence-weighted version.
    
    Args:
        scores: List of scores from each agent
        
    Returns:
        Standard deviation of the scores
    """
    if not scores or len(scores) < 2:
        return 0.0
    
    scores_array = np.array(scores, dtype=np.float64)
    return float(np.std(scores_array))


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


if __name__ == "__main__":
    # Test the disagreement metrics
    print("Testing disagreement metrics...")
    
    # Mock agent outputs
    test_outputs = [
        {"score": 0.3, "confidence": 0.8, "reasoning": "Strong fundamentals with solid earnings growth."},
        {"score": -0.2, "confidence": 0.6, "reasoning": "Negative news sentiment creating selling pressure."},
        {"score": 0.5, "confidence": 0.9, "reasoning": "Bullish trend confirmed by momentum indicators."},
        {"score": -0.1, "confidence": 0.7, "reasoning": "Mixed macro environment with rate uncertainty."}
    ]
    
    print("\nTest outputs:")
    for o in test_outputs:
        print(f"  Score: {o['score']:.2f}, Confidence: {o['confidence']:.2f}")
    
    # Calculate all metrics
    metrics = calculate_all_metrics(test_outputs)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nDisagreement metrics test complete!")
