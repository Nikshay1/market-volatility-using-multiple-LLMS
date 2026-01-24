# Agentic Dissonance v2

> **Confidence-Weighted Multi-Agent Disagreement for Volatility Modeling**

A Python research framework that models financial market volatility using belief dispersion among heterogeneous LLM agents. Four specialized AI agents (Fundamental, Sentiment, Technical, Macro) debate market conditions, and their **confidence-weighted disagreement** is used as an exogenous variable in a GARCH-X volatility model.

---

## 🧠 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Daily Analysis Flow                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   📊 Infobots inject data (fundamentals + macro)             │
│              ↓                                               │
│   🔵 ROUND 1: 4 agents produce independent beliefs           │
│   ┌──────────┬──────────┬──────────┬──────────┐             │
│   │Fundamental│Sentiment │Technical │  Macro   │             │
│   │ score    │ score    │ score    │ score    │             │
│   │ confidence│ confidence│ confidence│ confidence│           │
│   └──────────┴──────────┴──────────┴──────────┘             │
│              ↓                                               │
│   📈 Aggregator computes: μ (mean) + D (variance)            │
│              ↓                                               │
│   🔴 ROUND 2: Agents see group feedback, update beliefs      │
│              ↓                                               │
│   📉 Final disagreement signal D_conf → GARCH-X              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Configure API Keys

#### FRED API (Macroeconomic Data)
Get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) and set it:

```powershell
# Windows PowerShell
$env:FRED_API_KEY = "your-fred-api-key"
```

> **Note**: Without FRED_API_KEY, the system uses mock macro data.

### 3. Choose Your LLM Backend

| Backend | Best For | Rate Limits |
|---------|----------|-------------|
| **Ollama** (default) | Long runs (500+ days) | ✅ Unlimited |
| **Groq** | Quick tests | ⚠️ Limited |

### 4. Run a Quick Test

```powershell
# Activate virtual environment (Windows)
venv\Scripts\activate

# Run 3-day test
python -m src.backtest --test --days 3
```

### 5. Run Analysis

```powershell
python -m src.analysis
```

---

## 🏃 Running the Full Pipeline

### With Ollama (Recommended)

```powershell
# 1. Install Ollama from https://ollama.ai
# 2. Pull the model
ollama pull mistral

# 3. Run backtest (default: 4 tickers - AAPL, MSFT, TSLA, SPY)
python -m src.backtest

# 4. Run analysis
python -m src.analysis
```

### With Groq API

```powershell
# Set environment variables
$env:LLM_BACKEND = "groq"
$env:GROQ_API_KEY = "your-api-key"
$env:FRED_API_KEY = "your-fred-api-key"  # Optional: for real macro data

# Run test
python -m src.backtest --test --days 5
```

---

## 📁 Project Structure

```
project/
├── src/
│   ├── config.py          # Configuration (tickers, dates, LLM backend)
│   ├── infobots.py         # Data agents (fundamentals + macro data)
│   ├── agents.py           # 4 belief agents with confidence output
│   ├── aggregator.py       # Confidence-weighted mean & variance
│   ├── debate_engine.py    # 2-round debate protocol
│   ├── disagreement.py     # Disagreement metrics computation
│   ├── data_loader.py      # Market data & news fetching
│   ├── backtest.py         # Main backtest runner with caching
│   └── analysis.py         # GARCH-X modeling & visualization
├── data/
│   ├── raw_market_data.csv        # OHLCV data
│   ├── disagreement_signals.csv   # Daily disagreement metrics
│   └── cache/                     # LLM output cache
├── output/
│   └── results.png         # Analysis visualization
├── requirements.txt
├── README.md               # This file
├── commands.md             # Detailed command reference
└── locallyLLM.md           # Ollama setup guide
```

---

## 🤖 The Four Agents

| Agent | Analyzes | Score Meaning |
|-------|----------|---------------|
| **Fundamental** | P/E, margins, debt | Long-term valuation risk |
| **Sentiment** | News, headlines | Short-term crowd psychology |
| **Technical** | Price action, trends | Momentum direction |
| **Macro** | Rates, inflation, policy | Economic risk environment |

Each agent outputs:
```json
{
  "score": [-1.0, 1.0],
  "confidence": [0.0, 1.0],
  "reasoning": "..."
}
```

---

## 📊 Key Metrics

### Confidence-Weighted Disagreement (D_conf)

```
μ = Σ(cᵢ × sᵢ) / Σ(cᵢ)           # Weighted mean
D = Σ(cᵢ × (sᵢ - μ)²) / Σ(cᵢ)    # Weighted variance
```

### GARCH-X Model

```
σ²_t = ω + α×ε²_{t-1} + β×σ²_{t-1} + γ×D_{t-1}
```

Where `D_{t-1}` is the lagged disagreement signal.

---

## 🔧 CLI Commands

| Command | Description |
|---------|-------------|
| `python -m src.backtest --test --days N` | Quick test with N days |
| `python -m src.backtest` | Full backtest (all tickers) |
| `python -m src.backtest --ticker AAPL` | Single ticker backtest |
| `python -m src.backtest --resume 2024-06-01` | Resume from date |
| `python -m src.backtest --compare-rounds` | Compare 2/3/4 round configs |
| `python -m src.analysis` | Run GARCH analysis + plot |

---

## 📚 Documentation

- **[commands.md](commands.md)** - Detailed command reference
- **[locallyLLM.md](locallyLLM.md)** - Ollama setup guide

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ollama error: connection refused` | Run `ollama serve` |
| `Model not found` | Run `ollama pull llama3.1` |
| `GROQ_API_KEY not set` | Set environment variable |
| `FRED_API_KEY not set` (warning) | Set `$env:FRED_API_KEY` or use mock data |
| Rate limit errors (Groq) | Switch to Ollama |
| `FileNotFoundError: disagreement_signals.csv` | Run backtest first |

---

## 📄 License

MIT