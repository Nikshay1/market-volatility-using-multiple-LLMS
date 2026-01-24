"""
Debate engine for Agentic Dissonance v2.

Implements 2-round debate protocol with social feedback:
1. Round 1: All 4 agents produce independent beliefs
2. Aggregator computes confidence-weighted mean and variance
3. Round 2: Agents see other agents' reasoning + group stats, update beliefs
"""

import time
import json
import os
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime

from .agents import Agent, create_agents
from .aggregator import Aggregator
from .infobots import FundamentalInfobot, MacroInfobot, create_infobots
from . import config


class DebateRoom:
    """
    Orchestrates multi-round debates between 4 belief agents.
    
    Protocol:
    1. Infobots inject data into context
    2. Round 1: Independent belief generation
    3. Aggregator computes mean + variance
    4. Round 2: Agents see group feedback, update beliefs
    5. Final metrics computed
    """
    
    def __init__(
        self,
        agents: List[Agent] = None,
        num_rounds: int = None,
        ticker: str = None,
        use_cache: bool = True
    ):
        """
        Initialize the debate room.
        
        Args:
            agents: List of belief agents (default: create all 4)
            num_rounds: Number of debate rounds (default from config)
            ticker: Stock ticker for infobots
            use_cache: Whether to cache LLM outputs
        """
        self.agents = agents if agents is not None else create_agents()
        self.num_rounds = num_rounds if num_rounds is not None else config.DEBATE_ROUNDS
        self.ticker = ticker or config.DEFAULT_TICKER
        self.use_cache = use_cache and config.ENABLE_LLM_CACHE
        
        # Initialize infobots
        self.fund_infobot, self.macro_infobot = create_infobots(self.ticker)
        
        # Initialize aggregator
        self.aggregator = Aggregator()
    
    def _get_cache_path(self, date: datetime, round_num: int, agent_name: str) -> str:
        """Get cache file path for an agent response."""
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{self.ticker}_{date_str}_r{round_num}_{agent_name}.json"
        return os.path.join(config.CACHE_DIR, filename)
    
    def _load_cached_response(self, date: datetime, round_num: int, agent_name: str) -> Optional[Dict]:
        """Load cached response if available."""
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(date, round_num, agent_name)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                    # Check cache validity
                    cached_time = datetime.fromisoformat(cached.get('cached_at', '2000-01-01'))
                    if (datetime.now() - cached_time).days < config.CACHE_EXPIRY_DAYS:
                        return cached.get('response')
            except Exception as e:
                print(f"Cache read error: {e}")
        return None
    
    def _save_cached_response(self, date: datetime, round_num: int, agent_name: str, response: Dict):
        """Save response to cache."""
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path(date, round_num, agent_name)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'cached_at': datetime.now().isoformat(),
                    'response': response
                }, f)
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def run_daily_debate(
        self,
        market_context: str,
        date: datetime = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run a full debate for a single day.
        
        Args:
            market_context: Formatted market and data context string
            date: The date being analyzed
            verbose: Whether to print debug information
            
        Returns:
            Dictionary containing:
            - 'date': The analysis date
            - 'ticker': Stock ticker
            - 'rounds': List of round results
            - 'final_outputs': Final responses from each agent
            - 'metrics': Disagreement metrics
        """
        date = date or datetime.now()
        date_str = date.strftime("%Y-%m-%d")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Debate for {self.ticker} on {date_str}")
            print(f"Rounds: {self.num_rounds}")
            print(f"{'='*60}")
        
        # Inject infobot data into context
        full_context = self._build_full_context(market_context, date)
        
        rounds_data = []
        agent_outputs = []
        
        for round_num in range(1, self.num_rounds + 1):
            if verbose:
                print(f"\n--- Round {round_num}/{self.num_rounds} ---")
            
            # Prepare debate context for round 2+
            debate_context = None
            if round_num > 1 and agent_outputs:
                # Compute aggregator stats from previous round
                stats = self.aggregator.compute_statistics(agent_outputs)
                debate_context = self.aggregator.format_group_summary(stats, agent_outputs)
            
            # Get responses from all agents
            round_outputs = []
            for agent in self.agents:
                # Check cache first
                cached = self._load_cached_response(date, round_num, agent.name)
                if cached:
                    response = cached
                    if verbose:
                        print(f"  {agent.name}: score={response.get('score', 0):.3f} (cached)")
                else:
                    response = agent.generate_response(full_context, debate_context)
                    self._save_cached_response(date, round_num, agent.name, response)
                    if verbose:
                        print(f"  {agent.name}: score={response.get('score', 0):.3f}, conf={response.get('confidence', 0):.3f}")
                
                round_outputs.append(response)
                
                # Rate limit between agent calls
                time.sleep(config.RATE_LIMIT_DELAY)
            
            # Store round data
            rounds_data.append({
                "round": round_num,
                "outputs": [o.copy() for o in round_outputs]
            })
            
            agent_outputs = round_outputs
        
        # Compute final metrics
        final_stats = self.aggregator.compute_statistics(agent_outputs)
        disagreement_signal = self.aggregator.get_disagreement_signal(agent_outputs)
        
        if verbose:
            print(f"\n--- Final Metrics ---")
            print(f"Mean Score: {final_stats['mean_score']:.4f}")
            print(f"Variance (Disagreement): {final_stats['variance']:.4f}")
            print(f"Avg Confidence: {final_stats['avg_confidence']:.4f}")
        
        return {
            "date": date_str,
            "ticker": self.ticker,
            "rounds": rounds_data,
            "final_outputs": {agent.name.lower(): output for agent, output in zip(self.agents, agent_outputs)},
            "metrics": {
                "disagreement_conf": final_stats["variance"],
                "mean_score": final_stats["mean_score"],
                "avg_confidence": final_stats["avg_confidence"]
            },
            "disagreement_signal": disagreement_signal
        }
    
    def _build_full_context(self, market_context: str, date: datetime) -> str:
        """
        Build full context including infobot data.
        
        Args:
            market_context: Base market context
            date: Reference date
            
        Returns:
            Full context string with all data
        """
        fundamental_data = self.fund_infobot.format_for_context(date)
        macro_data = self.macro_infobot.format_for_context(date)
        
        return f"""{market_context}
{fundamental_data}
{macro_data}
"""


def run_debate_experiment(
    ticker: str = None,
    date: datetime = None,
    rounds_list: List[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run debate experiment comparing different round counts.
    
    Args:
        ticker: Stock ticker
        date: Reference date
        rounds_list: List of round counts to test (default: [2, 3, 4])
        verbose: Whether to print results
        
    Returns:
        Dictionary with results for each round count
    """
    from .data_loader import (
        fetch_market_data, load_market_data,
        get_market_context_for_date, format_context_for_agent, fetch_news
    )
    
    ticker = ticker or config.DEFAULT_TICKER
    date = date or datetime(2024, 6, 15)
    rounds_list = rounds_list or [2, 3, 4]
    
    # Load market data
    try:
        market_df = load_market_data()
    except FileNotFoundError:
        market_df = fetch_market_data()
    
    # Get base context
    market_context = get_market_context_for_date(market_df, date)
    news = fetch_news(ticker, date)
    context_str = format_context_for_agent(market_context, news, ticker)
    
    results = {}
    
    for num_rounds in rounds_list:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing {num_rounds} debate rounds")
            print(f"{'='*60}")
        
        start_time = time.time()
        
        room = DebateRoom(num_rounds=num_rounds, ticker=ticker, use_cache=False)
        debate_result = room.run_daily_debate(context_str, date, verbose=verbose)
        
        elapsed = time.time() - start_time
        
        results[num_rounds] = {
            "disagreement": debate_result["metrics"]["disagreement_conf"],
            "mean_score": debate_result["metrics"]["mean_score"],
            "avg_confidence": debate_result["metrics"]["avg_confidence"],
            "runtime_seconds": elapsed
        }
    
    if verbose:
        print(f"\n{'='*60}")
        print("EXPERIMENT RESULTS")
        print(f"{'='*60}")
        print(f"{'Rounds':<10} {'Disagreement':<15} {'Mean Score':<12} {'Confidence':<12} {'Runtime':<10}")
        print("-" * 60)
        for num_rounds, data in results.items():
            print(f"{num_rounds:<10} {data['disagreement']:<15.4f} {data['mean_score']:<12.4f} "
                  f"{data['avg_confidence']:<12.4f} {data['runtime_seconds']:<10.2f}s")
    
    return results


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
    news = fetch_news(config.DEFAULT_TICKER, test_date)
    context_str = format_context_for_agent(market_context, news)
    
    print("Market Context:")
    print(context_str[:500] + "...")
    print("\n")
    
    # Run debate
    room = DebateRoom()
    results = room.run_daily_debate(context_str, test_date, verbose=True)
    
    return results


if __name__ == "__main__":
    results = run_single_debate_test()
    print("\n\nFinal Results:")
    print(f"Date: {results['date']}")
    print(f"Ticker: {results['ticker']}")
    print(f"Disagreement (D_conf): {results['metrics']['disagreement_conf']:.4f}")
    print(f"Mean Score: {results['metrics']['mean_score']:.4f}")
    print(f"Avg Confidence: {results['metrics']['avg_confidence']:.4f}")
