# Market Volatility Using Multiple LLMs

A multi-agent debate framework for volatility forecasting using LLM-based belief agents. This system measures residual disagreement between agents and tests whether it can serve as a proxy for market risk.

## 📋 Current Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Target Asset** | TSLA | High volatility 2018-2020 (Funding Secured, Model 3, COVID) |
| **Time Window** | 2018-01-01 to 2020-12-31 | Covers Trade War, Fed Pivot, COVID Crash |
| **Temperature** | 0.7 | Forces heterogeneous agent responses |
| **Debate Rounds** | 2 | Round 1 = individual bias, Round 2 = social friction |
| **Agents** | 3 | Sentiment, Technical, Macro (Fundamental removed) |

---

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
cd market-volatility-using-multiple-LLMS
pip install -r requirements.txt
```

### 2. Set Up LLM Backend

#### Option A: Ollama (Local - Recommended)
```powershell
# Install Ollama from https://ollama.ai
ollama pull mistral

# Set environment variable (optional, defaults to Ollama)
$env:LLM_BACKEND = "ollama"
```

#### Option B: Groq Cloud API
```powershell
# Get API key from https://console.groq.com
$env:GROQ_API_KEY = "your-api-key-here"
$env:LLM_BACKEND = "groq"
```

### 3. Set Up Historical News Data

Download the Kaggle dataset and convert it:

```powershell
# 1. Download from: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
# 2. Place "analyst_ratings_processed.csv" in the project root
# 3. Run the import script:

python scripts/import_kaggle.py
```

This creates `data/historical_news.csv` with TSLA headlines for 2018-2020.

### 4. (Optional) Set Up FRED API for Macro Data

```powershell
# Get free API key from: https://fred.stlouisfed.org/docs/api/api_key.html
$env:FRED_API_KEY = "your-fred-api-key"
```

---

## 🏃 Running the Project

### Run Full Backtest
```powershell
python -m src.backtest
```
This runs weekly debates (Fridays only) across the 2018-2020 period.

### Run Quick Test (3-5 days)
```powershell
python -m src.backtest --test --days 5
```

### Run Analysis After Backtest
```powershell
python -m src.analysis
```
Outputs:
- Correlation analysis (Disagreement ↔ Forward Volatility)
- GARCH vs GARCH-X model comparison
- Executive Summary with PASSED/MIXED/FAILED verdict
- Visualization saved to `output/results.png`

---

## 📁 Project Structure

```
market-volatility-using-multiple-LLMS/
├── src/
│   ├── config.py         # Configuration (TSLA, dates, temperatures)
│   ├── agents.py         # LLM Belief Agents (Sentiment, Technical, Macro)
│   ├── infobots.py       # Data providers (MacroInfobot with VIX/TNX/Oil/DXY)
│   ├── data_loader.py    # Market data & historical news loader
│   ├── debate_engine.py  # Multi-round debate orchestration
│   ├── backtest.py       # Backtest runner (weekly Friday sampling)
│   ├── analysis.py       # GARCH modeling & executive summary
│   ├── disagreement.py   # Disagreement metrics calculation
│   └── aggregator.py     # Score aggregation
├── scripts/
│   └── import_kaggle.py  # Converts Kaggle news to required format
├── data/
│   ├── historical_news.csv    # TSLA headlines (created by import script)
│   ├── raw_market_data.csv    # Price data (created during backtest)
│   └── disagreement_signals.csv # Output signals
├── output/
│   └── results.png       # Analysis visualization
└── requirements.txt
```

---

## 🔧 Key Commands Reference

| Command | Description |
|---------|-------------|
| `python -m src.backtest` | Full backtest (weekly) |
| `python -m src.backtest --test --days 5` | Quick test mode |
| `python -m src.backtest --ticker TSLA` | Single ticker |
| `python -m src.backtest --rounds 2` | Set debate rounds |
| `python -m src.backtest --resume 2019-01-01` | Resume from date |
| `python -m src.backtest --compare-rounds` | Compare 2/3/4 rounds |
| `python -m src.analysis` | Run statistical analysis |

---

## 📊 Understanding the Output

### Disagreement Metrics
- **D_conf**: Confidence-weighted disagreement between agents
- **D_conf_change**: Rate of change (spikes predict volatility)

### Executive Summary Verdict
- **PASSED**: Strong evidence (Lower AIC + p < 0.05)
- **MIXED**: Weak evidence (Lower AIC but p > 0.05)
- **FAILED**: Baseline GARCH performed better

---

## ⚠️ Important Notes

1. **FundamentalAgent Disabled**: Removed to prevent look-ahead bias (fetching 2026 P/E ratios for 2018 predictions)

2. **Historical News Required**: The system uses `data/historical_news.csv` instead of live RSS feeds to ensure proper backtesting

3. **Weekly Sampling**: Debates run only on Fridays to reduce runtime by 5x while capturing volatility trends

4. **VIX/TNX/Oil/DXY**: MacroInfobot fetches these from yfinance for the historical date

---

## 📝 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_BACKEND` | No | `ollama` (default) or `groq` |
| `GROQ_API_KEY` | If using Groq | Groq Cloud API key |
| `OLLAMA_BASE_URL` | No | Default: `http://localhost:11434` |
| `OLLAMA_MODEL` | No | Default: `mistral` |
| `FRED_API_KEY` | No | Optional for enhanced macro data |
