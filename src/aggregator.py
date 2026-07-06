"""
Aggregator for computing social feedback statistics.

Implements confidence-weighted mean and variance for the multi-agent
belief formation system.
"""

import numpy as np
from itertools import combinations
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
        self.score_history: List[Dict[str, float]] = []
        self.pair_variance_history: List[Dict[str, float]] = []
    
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
                "mean_volatility_risk": 0.0,
                "volatility_risk_disagreement": 0.0,
                "num_agents": 0
            }
        
        # Extract directional scores, volatility-risk scores, and confidences
        scores = []
        volatility_risks = []
        confidences = []
        
        for output in agent_outputs:
            score = output.get('score', 0.0)
            confidence = output.get('confidence', 0.5)
            volatility_risk = output.get('volatility_risk', abs(float(score)))
            
            # Validate and clamp values
            score = max(-1.0, min(1.0, float(score)))
            confidence = max(0.0, min(1.0, float(confidence)))
            volatility_risk = max(0.0, min(1.0, float(volatility_risk)))
            
            scores.append(score)
            volatility_risks.append(volatility_risk)
            confidences.append(confidence)
        
        scores = np.array(scores, dtype=np.float64)
        volatility_risks = np.array(volatility_risks, dtype=np.float64)
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
        
        # Confidence-weighted volatility-risk level and disagreement
        weighted_volatility_risk = np.sum(confidences * volatility_risks) / total_confidence
        volatility_risk_variance = np.sum(confidences * (volatility_risks - weighted_volatility_risk) ** 2) / total_confidence

        # Average confidence
        avg_confidence = np.mean(confidences)
        
        agent_scores = {
            output.get("agent_name", f"agent_{i}").lower(): float(scores[i])
            for i, output in enumerate(agent_outputs)
        }
        pair_variance = self._compute_pairwise_variance(agent_outputs)

        result = {
            "mean_score": float(weighted_mean),
            "variance": float(weighted_variance),
            "avg_confidence": float(avg_confidence),
            "mean_volatility_risk": float(weighted_volatility_risk),
            "volatility_risk_disagreement": float(volatility_risk_variance),
            "num_agents": len(agent_outputs),
            "pairwise_variance": pair_variance
        }
        
        # Store in history
        self.history.append(result)
        self.score_history.append(agent_scores)
        self.pair_variance_history.append(pair_variance)
        
        return result
    
    def _compute_pairwise_variance(self, agent_outputs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute between-agent disagreement decomposition by pair."""
        pair_variance: Dict[str, float] = {}
        normalized = []
        for output in agent_outputs:
            name = output.get("agent_name", "unknown").lower()
            score = max(-1.0, min(1.0, float(output.get("score", 0.0))))
            normalized.append((name, score))

        for (left_name, left_score), (right_name, right_score) in combinations(normalized, 2):
            key = f"pair_var_{left_name}_{right_name}"
            pair_variance[key] = float(((left_score - right_score) ** 2) / 2.0)

        return pair_variance

    def get_pairwise_correlation(self) -> Dict[str, float]:
        """Compute pairwise score correlation across historical days."""
        if len(self.score_history) < 2:
            return {}

        agent_names = sorted({name for day in self.score_history for name in day.keys()})
        correlations: Dict[str, float] = {}

        for left_name, right_name in combinations(agent_names, 2):
            left_series = []
            right_series = []
            for day in self.score_history:
                if left_name in day and right_name in day:
                    left_series.append(day[left_name])
                    right_series.append(day[right_name])
            if len(left_series) < 2:
                continue
            corr = float(np.corrcoef(left_series, right_series)[0, 1])
            if np.isnan(corr):
                continue
            correlations[f"{left_name}__{right_name}"] = corr

        return correlations

    def get_calibration_signals(self, threshold: float, min_periods: int) -> Dict[str, Any]:
        """Return persistent high-correlation pairs that should be diversified."""
        if len(self.score_history) < min_periods:
            return {"high_pairs": [], "pair_correlations": {}}

        recent_scores = self.score_history[-min_periods:]
        agent_names = sorted({name for day in recent_scores for name in day.keys()})
        high_pairs = []
        pair_correlations = {}

        for left_name, right_name in combinations(agent_names, 2):
            left_series = []
            right_series = []
            for day in recent_scores:
                if left_name in day and right_name in day:
                    left_series.append(day[left_name])
                    right_series.append(day[right_name])
            if len(left_series) < min_periods:
                continue

            corr = float(np.corrcoef(left_series, right_series)[0, 1])
            if np.isnan(corr):
                continue
            key = f"{left_name}__{right_name}"
            pair_correlations[key] = corr
            if corr > threshold:
                high_pairs.append({"pair": key, "correlation": corr})

        return {"high_pairs": high_pairs, "pair_correlations": pair_correlations}

    def build_diversity_report(self) -> Dict[str, Any]:
        """Build a report that demonstrates whether agents are non-redundant."""
        return {
            "history_days": len(self.score_history),
            "pairwise_correlation": self.get_pairwise_correlation(),
            "average_pair_variance": self._average_pairwise_variance()
        }

    def _average_pairwise_variance(self) -> Dict[str, float]:
        if not self.pair_variance_history:
            return {}
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for day in self.pair_variance_history:
            for key, value in day.items():
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
        return {key: sums[key] / counts[key] for key in sums}

    def format_opposing_argument(
        self,
        current_agent_output: Dict[str, Any],
        all_outputs: List[Dict[str, Any]]
    ) -> str:
        """
        Format the OPPOSING argument for the current agent (Blind & Battle Protocol).
        
        Instead of showing the mean (which causes herding), we show the agent
        the most opposing viewpoint and ask them to critique it specifically.
        
        Args:
            current_agent_output: The current agent's Round 1 output
            all_outputs: All agents' Round 1 outputs
            
        Returns:
            Formatted critique prompt with the opposing argument
        """
        current_score = current_agent_output.get('score', 0.0)
        current_name = current_agent_output.get('agent_name', 'You')
        
        # Find the most opposing agent (maximum score difference)
        max_diff = -1
        opponent = None
        
        for output in all_outputs:
            opponent_name = output.get('agent_name', 'Unknown')
            if opponent_name == current_name:
                continue
            
            opponent_score = output.get('score', 0.0)
            diff = abs(current_score - opponent_score)
            
            if diff > max_diff:
                max_diff = diff
                opponent = output
        
        if opponent is None:
            return ""
        
        opponent_name = opponent.get('agent_name', 'Unknown')
        opponent_score = opponent.get('score', 0.0)
        opponent_reasoning = opponent.get('reasoning', 'No reasoning provided')
        
        # Determine if we're Bull vs Bear
        current_direction = "BULLISH" if current_score > 0 else "BEARISH" if current_score < 0 else "NEUTRAL"
        opponent_direction = "BULLISH" if opponent_score > 0 else "BEARISH" if opponent_score < 0 else "NEUTRAL"
        
        return f"""
================================================================================
                          🔥 CRITIQUE ROUND - BATTLE MODE 🔥
================================================================================

YOUR POSITION: {current_direction} (Score: {current_score:.3f})

OPPOSING ARGUMENT FROM {opponent_name.upper()} ({opponent_direction}):
Score: {opponent_score:.3f}
Reasoning: "{opponent_reasoning}"

================================================================================
                          YOUR MISSION (CRITIQUE PROTOCOL)
================================================================================

1. Point out the SPECIFIC FLAW in {opponent_name}'s logic.
2. Explain WHY your data/analysis is more relevant TODAY.
3. DO NOT agree just for harmony. If they are wrong, say so.
4. Update your score ONLY if their logic is undeniably correct.

RESPOND WITH YOUR UPDATED ASSESSMENT:
{{"score": <your directional score>, "volatility_risk": <your volatility risk>, "confidence": <your confidence>, "reasoning": "<your critique of their argument + your defense>"}}
"""
    
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
            "avg_confidence": stats["avg_confidence"],
            "mean_volatility_risk": stats["mean_volatility_risk"],
            "volatility_risk_disagreement": stats["volatility_risk_disagreement"]
        }
        result.update(stats.get("pairwise_variance", {}))
        
        # Add individual agent scores and confidences
        # Use agent_name from output, not position index (handles missing agents)
        for output in agent_outputs:
            agent_name = output.get("agent_name", "").lower()
            if agent_name:
                result[f"score_{agent_name}"] = output.get("score", 0.0)
                result[f"volatility_risk_{agent_name}"] = output.get("volatility_risk", abs(output.get("score", 0.0)))
                result[f"confidence_{agent_name}"] = output.get("confidence", 0.5)
        
        return result
    
    def reset(self):
        """Reset the aggregator history."""
        self.history = []
        self.score_history = []
        self.pair_variance_history = []


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
