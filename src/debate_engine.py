"""
Debate engine orchestrating multi-round debates between agents.
"""

import time
from typing import Dict, List, Any, Tuple
from datetime import datetime

from .agents import Agent, FundamentalAgent, SentimentAgent, create_agents
from .disagreement import calculate_scalar_disagreement, calculate_semantic_divergence
from . import config


class DebateRoom:
    """
    Orchestrates multi-round debates between Fundamental and Sentiment agents.
    Implements the debate protocol from Du et al., 2023.
    """
    
    def __init__(
        self,
        fundamental_agent: Agent = None,
        sentiment_agent: Agent = None,
        num_rounds: int = None
    ):
        """
        Initialize the debate room.
        
        Args:
            fundamental_agent: The fundamental analysis agent
            sentiment_agent: The sentiment analysis agent
            num_rounds: Number of debate rounds (default from config)
        """
        if fundamental_agent is None or sentiment_agent is None:
            self.fundamental_agent, self.sentiment_agent = create_agents()
        else:
            self.fundamental_agent = fundamental_agent
            self.sentiment_agent = sentiment_agent
        
        self.num_rounds = num_rounds if num_rounds is not None else config.DEBATE_ROUNDS
    
    def run_daily_debate(
        self,
        market_context: str,
        date: datetime = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run a full debate for a single day.
        
        Protocol:
        - Round 1: Independent generation
        - Rounds 2-N: Each agent sees other's previous response and updates
        
        Args:
            market_context: Formatted market and news context string
            date: The date being analyzed (for logging)
            verbose: Whether to print debug information
            
        Returns:
            Dictionary containing:
            - 'date': The analysis date
            - 'rounds': List of round results
            - 'final_outputs': Final responses from each agent
            - 'metrics': Disagreement metrics
        """
        date_str = date.strftime("%Y-%m-%d") if date else "unknown"
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Debate for {date_str}")
            print(f"{'='*60}")
        
        rounds_data = []
        fund_response = None
        sent_response = None
        
        for round_num in range(1, self.num_rounds + 1):
            if verbose:
                print(f"\n--- Round {round_num}/{self.num_rounds} ---")
            
            # Determine debate context for each agent
            if round_num == 1:
                # Round 1: Independent generation
                fund_debate_ctx = None
                sent_debate_ctx = None
            else:
                # Subsequent rounds: Each agent sees the other's previous response
                fund_debate_ctx = self._format_debate_context(
                    "SentimentAgent", sent_response
                )
                sent_debate_ctx = self._format_debate_context(
                    "FundamentalAgent", fund_response
                )
            
            # Get responses from both agents
            fund_response = self.fundamental_agent.generate_response(
                market_context, fund_debate_ctx
            )
            
            # Rate limit between calls
            time.sleep(config.RATE_LIMIT_DELAY)
            
            sent_response = self.sentiment_agent.generate_response(
                market_context, sent_debate_ctx
            )
            
            if verbose:
                print(f"Fundamental: score={fund_response.get('score', 'N/A'):.2f}")
                print(f"Sentiment:   score={sent_response.get('score', 'N/A'):.2f}")
            
            # Store round data
            rounds_data.append({
                "round": round_num,
                "fundamental": fund_response.copy(),
                "sentiment": sent_response.copy()
            })
            
            # Rate limit between rounds
            if round_num < self.num_rounds:
                time.sleep(config.RATE_LIMIT_DELAY)
        
        # Calculate metrics on final outputs
        metrics = self.calculate_metrics(fund_response, sent_response)
        
        if verbose:
            print(f"\n--- Final Metrics ---")
            print(f"Scalar Disagreement: {metrics['disagreement_scalar']:.4f}")
            print(f"Semantic Divergence: {metrics['disagreement_semantic']:.4f}")
        
        return {
            "date": date_str,
            "rounds": rounds_data,
            "final_outputs": {
                "fundamental": fund_response,
                "sentiment": sent_response
            },
            "metrics": metrics
        }
    
    def _format_debate_context(
        self,
        agent_name: str,
        response: Dict[str, Any]
    ) -> str:
        """
        Format an agent's response as debate context for the other agent.
        
        Args:
            agent_name: Name of the agent who produced the response
            response: The agent's response dictionary
            
        Returns:
            Formatted debate context string
        """
        return f"""Agent: {agent_name}
Score: {response.get('score', 0.0)}
Reasoning: {response.get('reasoning', 'No reasoning provided')}"""
    
    def calculate_metrics(
        self,
        fund_response: Dict[str, Any],
        sent_response: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate disagreement metrics from final agent outputs.
        
        Args:
            fund_response: Final fundamental agent response
            sent_response: Final sentiment agent response
            
        Returns:
            Dictionary with disagreement metrics
        """
        # Extract scores
        scores = [
            fund_response.get('score', 0.0),
            sent_response.get('score', 0.0)
        ]
        
        # Extract reasonings
        reasonings = [
            fund_response.get('reasoning', ''),
            sent_response.get('reasoning', '')
        ]
        
        # Calculate metrics
        scalar_disagreement = calculate_scalar_disagreement(scores)
        semantic_divergence = calculate_semantic_divergence(reasonings)
        
        return {
            "disagreement_scalar": scalar_disagreement,
            "disagreement_semantic": semantic_divergence,
            "score_fundamental": scores[0],
            "score_sentiment": scores[1]
        }


def run_single_debate_test():
    """Test function to run a single debate."""
    from .data_loader import (
        fetch_market_data, load_market_data,
        get_market_context_for_date, format_context_for_agent, fetch_news
    )
    
    # Load market data
    try:
        market_df = load_market_data()
    except FileNotFoundError:
        market_df = fetch_market_data()
    
    # Pick a test date
    test_date = datetime(2024, 6, 15)
    
    # Get context
    market_context = get_market_context_for_date(market_df, test_date)
    news = fetch_news(config.TICKER, test_date)
    context_str = format_context_for_agent(market_context, news)
    
    print("Market Context:")
    print(context_str)
    print("\n")
    
    # Run debate
    room = DebateRoom()
    results = room.run_daily_debate(context_str, test_date, verbose=True)
    
    return results


if __name__ == "__main__":
    results = run_single_debate_test()
    print("\n\nFinal Results:")
    print(f"Date: {results['date']}")
    print(f"Scalar Disagreement: {results['metrics']['disagreement_scalar']:.4f}")
    print(f"Semantic Divergence: {results['metrics']['disagreement_semantic']:.4f}")
