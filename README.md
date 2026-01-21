# Market Volatility Analysis using Multi-Agent LLM Debates

This project uses a multi-agent debate framework to analyze market volatility by having LLM agents debate market conditions and measuring their disagreement as a proxy for market uncertainty.

---

## Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Choose Your LLM Backend

You can run this project using either:
- **Ollama (Local)** - No rate limits, unlimited runs ✅ *(Default)*
- **Groq API (Cloud)** - Faster but has rate limits

---

## Option A: Run with Ollama (Local LLM) - Recommended for Long Runs

This is the **default** configuration. Perfect for analyzing 500+ days without rate limits.

### Setup Ollama

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)

2. **Pull the Mistral model**:
   ```powershell
   ollama pull mistral
   ```

3. **Verify Ollama is running**:
   ```powershell
   ollama list
   ```

### Run with Ollama

No environment variables needed - Ollama is the default:

```powershell
# Quick test (5 days)
python -m src.backtest --test --days 5

# Full backtest
python -m src.backtest

# Run analysis
python -m src.analysis
```

---

## Option B: Run with Groq API (Cloud LLM)

Use this for faster processing, but be aware of rate limits (~20-30 days max).

### Set Environment Variables

```powershell
# Windows PowerShell
$env:LLM_BACKEND = "groq"
$env:GROQ_API_KEY = "your-groq-api-key"

# Windows CMD
set LLM_BACKEND=groq
set GROQ_API_KEY=your-groq-api-key

# Linux/Mac
export LLM_BACKEND="groq"
export GROQ_API_KEY="your-groq-api-key"
```

### Run with Groq

```powershell
python -m src.backtest --test --days 5
```

---

## Switching Between Backends

| Backend | Environment Variable | When to Use |
|---------|---------------------|-------------|
| **Ollama** (default) | `$env:LLM_BACKEND = "ollama"` | Long runs (500+ days), no rate limits |
| **Groq** | `$env:LLM_BACKEND = "groq"` | Quick tests, faster inference |

### Changing the Default

To permanently change the default backend, edit `src/config.py`:

```python
# Options: "groq" or "ollama"
LLM_BACKEND: str = os.environ.get("LLM_BACKEND", "ollama")  # Change "ollama" to "groq"
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `python -m src.backtest --test --days 5` | Quick test with 5 days |
| `python -m src.backtest` | Full backtest |
| `python -m src.backtest --resume 2024-06-15` | Resume from a specific date |
| `python -m src.analysis` | Run statistical analysis |

For detailed command documentation, see [commands.md](commands.md).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ollama error: connection refused` | Make sure Ollama is running: `ollama serve` |
| `Model not found` | Pull the model: `ollama pull mistral` |
| `GROQ_API_KEY not set` | Set the environment variable (see Option B) |
| Rate limit errors (Groq) | Switch to Ollama or wait and retry |

---

## Project Structure

```
├── src/
│   ├── config.py         # Configuration (LLM backend, dates, etc.)
│   ├── agents.py         # LLM agents with Groq/Ollama support
│   ├── backtest.py       # Main backtest runner
│   ├── analysis.py       # Statistical analysis
│   └── ...
├── data/                 # Generated data files
├── output/               # Analysis results and plots
├── requirements.txt      # Python dependencies
├── commands.md           # Detailed command reference
└── locallyLLM.md         # Detailed Ollama setup guide
```