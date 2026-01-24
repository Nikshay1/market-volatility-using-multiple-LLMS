"""
LLM Belief Agents for Agentic Dissonance v2.

Four heterogeneous agents that produce score + confidence + reasoning:
- FundamentalAgent: Long-term valuation and risk
- SentimentAgent: Short-term crowd psychology
- TechnicalAgent: Trend and momentum signals
- MacroAgent: Macroeconomic conditions

Supports both Groq API (cloud) and Ollama (local) backends.
"""

import json
import time
import re
from typing import Dict, Optional, Any, List
from abc import ABC, abstractmethod

# Conditional imports based on backend
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import ollama
except ImportError:
    ollama = None

from . import config


class Agent(ABC):
    """
    Base class for belief agents.
    
    All agents output:
    - score: float in [-1, 1]
    - confidence: float in [0, 1]
    - reasoning: string
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
        self._groq_client = None
    
    @property
    def groq_client(self):
        """Lazy initialization of Groq client."""
        if self._groq_client is None and Groq is not None:
            self._groq_client = Groq(api_key=config.get_groq_api_key())
        return self._groq_client
    
    def call_llm(self, prompt: str, max_retries: int = None) -> str:
        """
        Call the LLM (either Groq or Ollama based on config).
        
        Args:
            prompt: User prompt to send
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response text from the model
        """
        if config.LLM_BACKEND == "ollama":
            return self._call_ollama(prompt, max_retries)
        else:
            return self._call_groq(prompt, max_retries)
    
    def _call_ollama(self, prompt: str, max_retries: int = None) -> str:
        """Call the Ollama local LLM."""
        if ollama is None:
            raise ImportError("ollama package not installed")
        
        max_retries = max_retries or config.API_RETRY_ATTEMPTS
        
        for attempt in range(max_retries):
            try:
                response = ollama.chat(
                    model=config.OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    options={
                        "temperature": config.OLLAMA_TEMPERATURE,
                        "num_predict": config.OLLAMA_MAX_TOKENS
                    }
                )
                return response["message"]["content"]
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Ollama error (attempt {attempt + 1}): {e}")
                    time.sleep(config.API_RETRY_DELAY)
                else:
                    raise
        
        return ""
    
    def _call_groq(self, prompt: str, max_retries: int = None) -> str:
        """Call the Groq API with retry logic."""
        if Groq is None:
            raise ImportError("groq package not installed")
        
        max_retries = max_retries or config.API_RETRY_ATTEMPTS
        
        for attempt in range(max_retries):
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=config.GROQ_MODEL,
                    temperature=config.GROQ_TEMPERATURE,
                    max_tokens=config.GROQ_MAX_TOKENS
                )
                return chat_completion.choices[0].message.content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Groq API error (attempt {attempt + 1}): {e}")
                    time.sleep(config.API_RETRY_DELAY)
                else:
                    raise
        
        return ""
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from the model response.
        
        Expected format:
        {
            "score": float in [-1, 1],
            "confidence": float in [0, 1],
            "reasoning": string
        }
        """
        try:
            # Sanitize the response - escape control characters in strings
            sanitized = self._sanitize_json_string(response)
            
            # Try direct JSON parse
            if sanitized.strip().startswith('{'):
                try:
                    return self._validate_response(json.loads(sanitized))
                except json.JSONDecodeError:
                    pass
            
            # Try to find and extract JSON object
            json_str = self._extract_json_object(response)
            if json_str:
                sanitized_json = self._sanitize_json_string(json_str)
                try:
                    return self._validate_response(json.loads(sanitized_json))
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON block in markdown
            json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_block:
                sanitized_block = self._sanitize_json_string(json_block.group(1))
                try:
                    return self._validate_response(json.loads(sanitized_block))
                except json.JSONDecodeError:
                    pass
            
            # Fallback: extract values manually using regex
            return self._extract_values_manually(response)
            
        except Exception as e:
            print(f"Warning: Could not parse response from {self.name}: {e}")
            return self._extract_values_manually(response)
    
    def _sanitize_json_string(self, text: str) -> str:
        """
        Sanitize a string to make it valid JSON by escaping control characters.
        """
        # First, let's handle the case where we have valid JSON with unescaped newlines
        # We need to escape control characters inside string values only
        
        result = []
        in_string = False
        escape_next = False
        
        for char in text:
            if escape_next:
                result.append(char)
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                result.append(char)
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                result.append(char)
                continue
            
            if in_string:
                # Escape control characters inside strings
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                elif ord(char) < 32:
                    result.append(f'\\u{ord(char):04x}')
                else:
                    result.append(char)
            else:
                # Outside strings, keep whitespace for formatting
                result.append(char)
        
        return ''.join(result)
    
    def _extract_json_object(self, text: str) -> Optional[str]:
        """
        Extract a JSON object from text, handling nested braces.
        """
        # Find the first opening brace
        start = text.find('{')
        if start == -1:
            return None
        
        # Count braces to find matching close
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        
        return None
    
    def _extract_values_manually(self, response: str) -> Dict[str, Any]:
        """
        Extract score, confidence, and reasoning using regex patterns.
        """
        # Try to find score
        score_match = re.search(r'"?score"?\s*[:=]\s*([-+]?\d*\.?\d+)', response)
        score = float(score_match.group(1)) if score_match else 0.0
        
        # Try to find confidence
        conf_match = re.search(r'"?confidence"?\s*[:=]\s*(\d*\.?\d+)', response)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        
        # Try to find reasoning - get text between "reasoning": and the next comma or brace
        reason_match = re.search(
            r'"?reasoning"?\s*[:=]\s*"([^"]*(?:"[^"]*)*)"',
            response, re.DOTALL
        )
        if reason_match:
            reasoning = reason_match.group(1)
            # Unescape any escaped characters
            reasoning = reasoning.replace('\\n', ' ').replace('\\t', ' ')
        else:
            # Just take a snippet of the response as reasoning
            reasoning = response[:200].replace('\n', ' ').replace('\r', ' ')
        
        return self._validate_response({
            "score": score,
            "confidence": confidence,
            "reasoning": reasoning
        })
    
    def _validate_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clamp response values."""
        score = float(data.get('score', 0.0))
        confidence = float(data.get('confidence', 0.5))
        reasoning = str(data.get('reasoning', ''))
        
        # Clamp values to valid ranges
        score = max(-1.0, min(1.0, score))
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "score": score,
            "confidence": confidence,
            "reasoning": reasoning,
            "agent_name": self.name
        }
    
    @abstractmethod
    def get_analysis_prompt(self, context: str, debate_context: Optional[str] = None) -> str:
        """Get the analysis prompt for this agent type."""
        pass
    
    def generate_response(
        self,
        context: str,
        debate_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate agent response for the given context.
        
        Args:
            context: Market and data context
            debate_context: Optional group feedback and other agents' reasoning
            
        Returns:
            Dictionary with 'score', 'confidence', 'reasoning', 'agent_name'
        """
        prompt = self.get_analysis_prompt(context, debate_context)
        response = self.call_llm(prompt)
        return self.parse_json_response(response)


class FundamentalAgent(Agent):
    """
    Agent focusing on long-term valuation and fundamental risk.
    
    Analyzes: P/E ratios, revenue growth, balance sheet, margins.
    Score meaning: Fundamental valuation attractiveness
    """
    
    SYSTEM_PROMPT = """You are a fundamental analyst focused on long-term valuation and risk.

Your role is to analyze:
- Earnings quality and growth
- Balance sheet strength (debt levels, cash)
- Valuation ratios (P/E, P/B, P/S)
- Profit margins and efficiency
- Business model sustainability

Your SCORE represents LONG-TERM VALUATION:
- Score = +1: Extremely undervalued, strong fundamentals, low risk
- Score = 0: Fair value, neutral outlook
- Score = -1: Extremely overvalued, weak fundamentals, high risk

Your CONFIDENCE represents how certain you are (0.0 = pure guess, 1.0 = very certain).

IMPORTANT: You must respond ONLY with a valid JSON object in this exact format:
{"score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<your analysis>"}"""
    
    def __init__(self):
        super().__init__(name="Fundamental", system_prompt=self.SYSTEM_PROMPT)
    
    def get_analysis_prompt(self, context: str, debate_context: Optional[str] = None) -> str:
        base_prompt = f"""Analyze the following data and provide your fundamental assessment:

{context}

Based on the fundamental data, evaluate the long-term valuation and risk."""

        if debate_context:
            base_prompt += f"""

{debate_context}

INSTRUCTION: Critique other agents' positions and defend your own analysis.
Update your score and confidence ONLY if the group information reveals something important you missed.
Maintain your independence - do not simply follow the crowd."""
        
        base_prompt += """

Respond with a JSON object: {"score": <float>, "confidence": <float>, "reasoning": "<analysis>"}"""
        
        return base_prompt


class SentimentAgent(Agent):
    """
    Agent focusing on short-term crowd psychology and news sentiment.
    
    Analyzes: News headlines, market narratives, fear/greed signals.
    Score meaning: Short-term sentiment direction
    """
    
    SYSTEM_PROMPT = """You are a sentiment analyst focused on crowd psychology and news impact.

Your role is to analyze:
- News headlines and narratives
- Market fear/greed indicators
- Crowd behavior and momentum
- Contrarian opportunities
- Short-term psychological factors

Your SCORE represents SHORT-TERM SENTIMENT:
- Score = +1: Extreme bullish sentiment, possible euphoria
- Score = 0: Neutral sentiment, balanced views
- Score = -1: Extreme bearish sentiment, possible fear

Your CONFIDENCE represents how certain you are (0.0 = pure guess, 1.0 = very certain).

IMPORTANT: You must respond ONLY with a valid JSON object in this exact format:
{"score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<your analysis>"}"""
    
    def __init__(self):
        super().__init__(name="Sentiment", system_prompt=self.SYSTEM_PROMPT)
    
    def get_analysis_prompt(self, context: str, debate_context: Optional[str] = None) -> str:
        base_prompt = f"""Analyze the following data and provide your sentiment assessment:

{context}

Based on the news and market psychology, evaluate the short-term sentiment direction."""

        if debate_context:
            base_prompt += f"""

{debate_context}

INSTRUCTION: Critique other agents' positions and defend your own analysis.
Update your score and confidence ONLY if the group information reveals something important you missed.
Maintain your independence - do not simply follow the crowd."""
        
        base_prompt += """

Respond with a JSON object: {"score": <float>, "confidence": <float>, "reasoning": "<analysis>"}"""
        
        return base_prompt


class TechnicalAgent(Agent):
    """
    Agent focusing on trend and momentum signals.
    
    Analyzes: Price action, trends, support/resistance, momentum.
    Score meaning: Technical trend direction
    """
    
    SYSTEM_PROMPT = """You are a technical analyst focused on price trends and momentum.

Your role is to analyze:
- Price trends and patterns
- Support and resistance levels
- Momentum indicators (implied from price action)
- Volume and volatility patterns
- Chart formations

Your SCORE represents TREND DIRECTION:
- Score = +1: Strong bullish trend, high momentum
- Score = 0: Sideways/consolidation, no clear trend
- Score = -1: Strong bearish trend, negative momentum

Your CONFIDENCE represents how certain you are (0.0 = pure guess, 1.0 = very certain).

IMPORTANT: You must respond ONLY with a valid JSON object in this exact format:
{"score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<your analysis>"}"""
    
    def __init__(self):
        super().__init__(name="Technical", system_prompt=self.SYSTEM_PROMPT)
    
    def get_analysis_prompt(self, context: str, debate_context: Optional[str] = None) -> str:
        base_prompt = f"""Analyze the following data and provide your technical assessment:

{context}

Based on the price action and technical indicators, evaluate the trend direction."""

        if debate_context:
            base_prompt += f"""

{debate_context}

INSTRUCTION: Critique other agents' positions and defend your own analysis.
Update your score and confidence ONLY if the group information reveals something important you missed.
Maintain your independence - do not simply follow the crowd."""
        
        base_prompt += """

Respond with a JSON object: {"score": <float>, "confidence": <float>, "reasoning": "<analysis>"}"""
        
        return base_prompt


class MacroAgent(Agent):
    """
    Agent focusing on macroeconomic conditions.
    
    Analyzes: Interest rates, inflation, GDP, policy stance.
    Score meaning: Macroeconomic risk environment
    """
    
    SYSTEM_PROMPT = """You are a macro analyst focused on economic conditions and policy.

Your role is to analyze:
- Interest rate environment
- Inflation trends
- Economic growth (GDP)
- Central bank policy stance
- Yield curve signals
- Global macro risks

Your SCORE represents MACRO RISK ENVIRONMENT:
- Score = +1: Very favorable macro conditions, low risk
- Score = 0: Neutral conditions, balanced risks
- Score = -1: Unfavorable macro conditions, high risk

Your CONFIDENCE represents how certain you are (0.0 = pure guess, 1.0 = very certain).

IMPORTANT: You must respond ONLY with a valid JSON object in this exact format:
{"score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<your analysis>"}"""
    
    def __init__(self):
        super().__init__(name="Macro", system_prompt=self.SYSTEM_PROMPT)
    
    def get_analysis_prompt(self, context: str, debate_context: Optional[str] = None) -> str:
        base_prompt = f"""Analyze the following data and provide your macroeconomic assessment:

{context}

Based on the macro indicators, evaluate the risk environment for equities."""

        if debate_context:
            base_prompt += f"""

{debate_context}

INSTRUCTION: Critique other agents' positions and defend your own analysis.
Update your score and confidence ONLY if the group information reveals something important you missed.
Maintain your independence - do not simply follow the crowd."""
        
        base_prompt += """

Respond with a JSON object: {"score": <float>, "confidence": <float>, "reasoning": "<analysis>"}"""
        
        return base_prompt


def create_agents() -> List[Agent]:
    """
    Factory function to create all belief agents.
    
    Returns:
        List of [FundamentalAgent, SentimentAgent, TechnicalAgent, MacroAgent]
    """
    return [
        FundamentalAgent(),
        SentimentAgent(),
        TechnicalAgent(),
        MacroAgent()
    ]


def create_agents_dict() -> Dict[str, Agent]:
    """
    Factory function to create agents as a dictionary.
    
    Returns:
        Dictionary mapping agent names to agent instances
    """
    agents = create_agents()
    return {agent.name.lower(): agent for agent in agents}


if __name__ == "__main__":
    # Test agent initialization
    print("Testing agent creation...")
    agents = create_agents()
    
    for agent in agents:
        print(f"\n{agent.name}Agent:")
        print(f"  System prompt: {agent.system_prompt[:100]}...")
    
    print(f"\nCreated {len(agents)} agents successfully!")
    
    # Test with mock context
    test_context = """
    === Market Analysis for AAPL on 2024-06-15 ===
    
    MARKET DATA:
    - Current Price: $195.50
    - 5-day Return: +2.3%
    - 20-day Volatility: 18.5%
    
    FUNDAMENTAL DATA:
    - P/E Ratio: 28.5
    - Revenue Growth: 8.2%
    - Profit Margin: 25.3%
    
    NEWS HEADLINES:
    - "Apple announces new AI features at WWDC"
    - "iPhone sales beat expectations in Q2"
    """
    
    print("\nTest context created. Ready for debate!")
