"""
LLM Agent classes for the multi-agent debate framework.
Uses Groq API with llama-3.3-70b-versatile model.
"""

import json
import time
import re
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod

from groq import Groq

from . import config


class Agent(ABC):
    """
    Base class for debate agents.
    Handles Groq API calls with retry logic and rate limiting.
    """
    
    def __init__(self, name: str, system_prompt: str):
        """
        Initialize the agent.
        
        Args:
            name: Agent identifier name
            system_prompt: System prompt defining agent personality
        """
        self.name = name
        self.system_prompt = system_prompt
        self._client = None
    
    @property
    def client(self) -> Groq:
        """Lazy initialization of Groq client."""
        if self._client is None:
            api_key = config.get_groq_api_key()
            self._client = Groq(api_key=api_key)
        return self._client
    
    def call_groq(self, prompt: str, max_retries: int = None) -> str:
        """
        Call the Groq API with retry logic and rate limiting.
        
        Args:
            prompt: User prompt to send
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response text from the model
        """
        if max_retries is None:
            max_retries = config.API_RETRY_ATTEMPTS
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Rate limiting delay
                if attempt > 0:
                    time.sleep(config.API_RETRY_DELAY * (attempt + 1))
                
                response = self.client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.GROQ_TEMPERATURE,
                    max_tokens=config.GROQ_MAX_TOKENS,
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Check for rate limiting
                if "rate" in error_msg or "limit" in error_msg or "429" in error_msg:
                    print(f"[{self.name}] Rate limited. Waiting before retry...")
                    time.sleep(config.RATE_LIMIT_DELAY * (attempt + 1) * 2)
                else:
                    print(f"[{self.name}] API error (attempt {attempt + 1}/{max_retries}): {e}")
        
        raise RuntimeError(f"[{self.name}] Failed after {max_retries} attempts: {last_error}")
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from the model response.
        Handles various edge cases in LLM output.
        
        Args:
            response: Raw response text from the model
            
        Returns:
            Parsed dictionary with 'score' and 'reasoning'
        """
        # Try to extract JSON from the response
        try:
            # First, try direct parsing
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON block in markdown code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object
        json_pattern = r'\{[^{}]*"score"[^{}]*"reasoning"[^{}]*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[-1])
            except json.JSONDecodeError:
                pass
        
        # Try a more flexible pattern
        json_pattern = r'\{.*?\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if 'score' in parsed and 'reasoning' in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # Fallback: Try to extract score and reasoning manually
        score_match = re.search(r'"?score"?\s*[:\s]\s*(-?[\d.]+)', response)
        reasoning_match = re.search(r'"?reasoning"?\s*[:\s]\s*["\'](.+?)["\']', response, re.DOTALL)
        
        if score_match:
            score = float(score_match.group(1))
            # Clamp to [-1, 1]
            score = max(-1.0, min(1.0, score))
            reasoning = reasoning_match.group(1) if reasoning_match else response[:500]
            return {
                "score": score,
                "reasoning": reasoning
            }
        
        # Last resort: return a default structure
        print(f"[{self.name}] Warning: Could not parse JSON response. Using fallback.")
        return {
            "score": 0.0,
            "reasoning": response[:500] if len(response) > 500 else response
        }
    
    @abstractmethod
    def get_analysis_prompt(self, context: str) -> str:
        """
        Get the analysis prompt for this agent type.
        
        Args:
            context: Market and news context
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def generate_response(
        self,
        context: str,
        prev_debate_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate agent response for the given context.
        
        Args:
            context: Market and news context
            prev_debate_context: Other agent's previous response (for debate rounds)
            
        Returns:
            Dictionary with 'score' and 'reasoning'
        """
        # Build the prompt
        prompt = self.get_analysis_prompt(context)
        
        # Add debate context if this is a subsequent round
        if prev_debate_context:
            prompt += f"""

These are the solutions to the problem from other agents:
{prev_debate_context}

Using the opinions of other agents as additional advice, can you give an updated response?

Remember to respond with a JSON object containing:
- "score": a float between -1 (very bearish) and 1 (very bullish)
- "reasoning": your updated analysis as a string
"""
        
        # Call the model
        response = self.call_groq(prompt)
        
        # Parse the response
        parsed = self.parse_json_response(response)
        
        # Ensure score is in valid range
        if 'score' in parsed:
            parsed['score'] = max(-1.0, min(1.0, float(parsed['score'])))
        
        return parsed


class FundamentalAgent(Agent):
    """
    Agent representing a conservative value investor.
    Focuses on earnings, balance sheet strength, and valuation ratios.
    """
    
    SYSTEM_PROMPT = """You are a conservative value investor focusing on earnings, balance sheet strength, valuation ratios and long-term risk.

Your role is to analyze market conditions from a fundamental perspective:
- Focus on financial metrics, earnings quality, and intrinsic value
- Consider P/E ratios, debt levels, cash flows, and margin of safety
- Be skeptical of momentum and short-term price movements
- Prioritize downside protection and long-term wealth preservation

Always provide your analysis in a structured JSON format."""

    def __init__(self):
        super().__init__(
            name="FundamentalAgent",
            system_prompt=self.SYSTEM_PROMPT
        )
    
    def get_analysis_prompt(self, context: str) -> str:
        return f"""Analyze the following market data from a fundamental, value investing perspective.

{context}

Based on your analysis, provide a market outlook as a JSON object with:
- "score": a float between -1 (very bearish, high risk) and 1 (very bullish, low risk)
  - Score reflects your view on forward risk and return potential
  - Negative scores indicate concern about valuation or fundamentals
  - Positive scores indicate favorable risk/reward
- "reasoning": a detailed explanation of your fundamental analysis (2-4 sentences)

Focus on:
1. What do price movements suggest about underlying value?
2. Is current volatility justifying a margin of safety?
3. What risks are not priced into the market?

Respond ONLY with the JSON object, no additional text."""


class SentimentAgent(Agent):
    """
    Agent representing a short-term contrarian news trader.
    Focuses on headlines, narratives, fear, and crowd psychology.
    """
    
    SYSTEM_PROMPT = """You are a short-term contrarian news trader focusing on headlines, narratives, fear, and crowd psychology.

Your role is to analyze market sentiment and psychology:
- Focus on news flow, narrative shifts, and crowd behavior
- Look for signs of excessive fear or greed
- Consider how headlines might move prices in the short term
- Be contrarian when sentiment reaches extremes

Always provide your analysis in a structured JSON format."""

    def __init__(self):
        super().__init__(
            name="SentimentAgent",
            system_prompt=self.SYSTEM_PROMPT
        )
    
    def get_analysis_prompt(self, context: str) -> str:
        return f"""Analyze the following market data and headlines from a sentiment and contrarian perspective.

{context}

Based on your analysis, provide a market outlook as a JSON object with:
- "score": a float between -1 (very bearish, expect decline) and 1 (very bullish, expect rise)
  - Score reflects your near-term directional view based on sentiment
  - Negative scores: crowd is complacent, contrarian bearish signal
  - Positive scores: crowd is fearful, contrarian bullish signal
- "reasoning": a detailed explanation of your sentiment analysis (2-4 sentences)

Focus on:
1. What emotions are driving the headlines?
2. Is the crowd positioned on one side?
3. What contrarian opportunities exist?

Respond ONLY with the JSON object, no additional text."""


def create_agents() -> tuple:
    """
    Factory function to create the debate agents.
    
    Returns:
        Tuple of (FundamentalAgent, SentimentAgent)
    """
    return FundamentalAgent(), SentimentAgent()


if __name__ == "__main__":
    # Test agent initialization
    fund_agent, sent_agent = create_agents()
    
    test_context = """
    === Market Analysis for AAPL on 2024-06-15 ===
    
    MARKET DATA:
    - Current Price: $195.50
    - Daily Return: 0.75%
    - Trading Volume: 45,000,000
    
    5-DAY LOOKBACK:
    - Average Daily Return: 0.25%
    - Volatility (Std Dev): 1.2%
    - Cumulative Price Change: 2.5%
    
    RECENT NEWS HEADLINES:
    1. Apple announces new AI features for iPhone
    2. Morgan Stanley raises AAPL price target
    3. Tech stocks rally on Fed rate hopes
    """
    
    print("Testing FundamentalAgent...")
    print(f"System prompt: {fund_agent.system_prompt[:100]}...")
    
    print("\nTesting SentimentAgent...")
    print(f"System prompt: {sent_agent.system_prompt[:100]}...")
    
    print("\nAgents initialized successfully!")
