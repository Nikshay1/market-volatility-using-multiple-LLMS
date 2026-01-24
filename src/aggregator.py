"""
Aggregator for computing social feedback statistics.

Implements confidence-weighted mean and variance for the multi-agent
belief formation system.
"""

import numpy as np
from typing import Dict, List, Any, Optional


class Aggregator:
    """
    Computes social feedback statistics from agent beliefs.
    
    After each round, computes:
    - Confidence-weighted mean: μ = Σ(cᵢ × sᵢ) / Σ(cᵢ)
    - Confidence-weighted variance: D = Σ(cᵢ × (sᵢ - μ)²) / Σ(cᵢ)
    """
    
    def __init__(self):
        """Initialize the Aggregator."""
        self.history: List[Dict[str, Any]] = []
    
    def compute_statistics(
        self,
        agent_outputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute confidence-weighted statistics from agent outputs.
        
        Args:
            agent_outputs: List of dicts with 'score', 'confidence', 'reasoning'
            
        Returns:
            Dictionary with:
            - 'mean_score': Confidence-weighted mean
            - 'variance': Confidence-weighted variance (disagreement)
            - 'avg_confidence': Average confidence
            - 'num_agents': Number of agents
        """
        if not agent_outputs:
            return {
                "mean_score": 0.0,
                "variance": 0.0,
                "avg_confidence": 0.0,
                "num_agents": 0
            }
        
        # Extract scores and confidences
        scores = []
        confidences = []
        
        for output in agent_outputs:
            score = output.get('score', 0.0)
            confidence = output.get('confidence', 0.5)
            
            # Validate and clamp values
            score = max(-1.0, min(1.0, float(score)))
            confidence = max(0.0, min(1.0, float(confidence)))
            
            scores.append(score)
            confidences.append(confidence)
        
        scores = np.array(scores, dtype=np.float64)
        confidences = np.array(confidences, dtype=np.float64)
        
        # Handle edge case where all confidences are zero
        total_confidence = np.sum(confidences)
        if total_confidence == 0:
            total_confidence = len(confidences)
            confidences = np.ones_like(confidences) / len(confidences)
        
        # Confidence-weighted mean: μ = Σ(cᵢ × sᵢ) / Σ(cᵢ)
        weighted_mean = np.sum(confidences * scores) / total_confidence
        
        # Confidence-weighted variance: D = Σ(cᵢ × (sᵢ - μ)²) / Σ(cᵢ)
        weighted_variance = np.sum(confidences * (scores - weighted_mean) ** 2) / total_confidence
        
        # Average confidence
        avg_confidence = np.mean(confidences)
        
        result = {
            "mean_score": float(weighted_mean),
            "variance": float(weighted_variance),
            "avg_confidence": float(avg_confidence),
            "num_agents": len(agent_outputs)
        }
        
        # Store in history
        self.history.append(result)
        
        return result
    
    def format_group_summary(
        self,
        statistics: Dict[str, float],
        agent_outputs: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Format group belief summary for agents.
        
        Args:
            statistics: Output from compute_statistics()
            agent_outputs: Optional list of agent outputs for detailed summary
            
        Returns:
            Formatted string for agent context
        """
        summary = f"""
GROUP BELIEF SUMMARY:
- Mean Score: {statistics['mean_score']:.3f}
- Variance (Disagreement): {statistics['variance']:.4f}
- Average Confidence: {statistics['avg_confidence']:.3f}
- Number of Agents: {statistics['num_agents']}
"""
        
        if agent_outputs:
            summary += "\nOTHER AGENTS' POSITIONS:\n"
            for output in agent_outputs:
                agent_name = output.get('agent_name', 'Unknown')
                score = output.get('score', 0.0)
                confidence = output.get('confidence', 0.5)
                reasoning = output.get('reasoning', 'No reasoning provided')
                
                # Truncate reasoning if too long
                if len(reasoning) > 300:
                    reasoning = reasoning[:300] + "..."
                
                summary += f"""
--- {agent_name} ---
Score: {score:.3f} (Confidence: {confidence:.3f})
Reasoning: {reasoning}
"""
        
        return summary
    
    def get_disagreement_signal(
        self,
        agent_outputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Get the final disagreement signal for storage.
        
        Args:
            agent_outputs: Final round agent outputs
            
        Returns:
            Dictionary with disagreement metrics for CSV storage
        """
        stats = self.compute_statistics(agent_outputs)
        
        # Build output with individual agent data
        result = {
            "disagreement_conf": stats["variance"],
            "mean_score": stats["mean_score"],
            "avg_confidence": stats["avg_confidence"]
        }
        
        # Add individual agent scores and confidences
        agent_names = ["fundamental", "sentiment", "technical", "macro"]
        for i, output in enumerate(agent_outputs):
            if i < len(agent_names):
                name = agent_names[i]
                result[f"score_{name}"] = output.get("score", 0.0)
                result[f"confidence_{name}"] = output.get("confidence", 0.5)
        
        return result
    
    def reset(self):
        """Reset the aggregator history."""
        self.history = []


def compute_confidence_weighted_variance(
    scores: List[float],
    confidences: List[float]
) -> float:
    """
    Standalone function to compute confidence-weighted variance.
    
    Args:
        scores: List of agent scores [-1, 1]
        confidences: List of agent confidences [0, 1]
        
    Returns:
        Confidence-weighted variance (disagreement signal)
    """
    if not scores or not confidences or len(scores) != len(confidences):
        return 0.0
    
    scores = np.array(scores, dtype=np.float64)
    confidences = np.array(confidences, dtype=np.float64)
    
    total_conf = np.sum(confidences)
    if total_conf == 0:
        return float(np.var(scores))
    
    weighted_mean = np.sum(confidences * scores) / total_conf
    weighted_var = np.sum(confidences * (scores - weighted_mean) ** 2) / total_conf
    
    return float(weighted_var)


def compute_confidence_weighted_mean(
    scores: List[float],
    confidences: List[float]
) -> float:
    """
    Standalone function to compute confidence-weighted mean.
    
    Args:
        scores: List of agent scores [-1, 1]
        confidences: List of agent confidences [0, 1]
        
    Returns:
        Confidence-weighted mean score
    """
    if not scores or not confidences or len(scores) != len(confidences):
        return 0.0
    
    scores = np.array(scores, dtype=np.float64)
    confidences = np.array(confidences, dtype=np.float64)
    
    total_conf = np.sum(confidences)
    if total_conf == 0:
        return float(np.mean(scores))
    
    weighted_mean = np.sum(confidences * scores) / total_conf
    
    return float(weighted_mean)


if __name__ == "__main__":
    # Test the aggregator
    print("Testing Aggregator...")
    
    # Mock agent outputs
    test_outputs = [
        {"agent_name": "Fundamental", "score": 0.3, "confidence": 0.8, "reasoning": "Strong fundamentals"},
        {"agent_name": "Sentiment", "score": -0.2, "confidence": 0.6, "reasoning": "Negative news sentiment"},
        {"agent_name": "Technical", "score": 0.5, "confidence": 0.9, "reasoning": "Bullish trend"},
        {"agent_name": "Macro", "score": -0.1, "confidence": 0.7, "reasoning": "Mixed macro conditions"}
    ]
    
    agg = Aggregator()
    stats = agg.compute_statistics(test_outputs)
    
    print(f"Statistics: {stats}")
    print(agg.format_group_summary(stats, test_outputs))
    
    # Test standalone functions
    scores = [0.3, -0.2, 0.5, -0.1]
    confidences = [0.8, 0.6, 0.9, 0.7]
    
    print(f"\nStandalone variance: {compute_confidence_weighted_variance(scores, confidences):.4f}")
    print(f"Standalone mean: {compute_confidence_weighted_mean(scores, confidences):.4f}")
