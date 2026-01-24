# Running with Local LLM (Ollama)

This guide explains how to run Agentic Dissonance v2 using Ollama for unlimited local inference.

---

## Why Use Ollama?

| Aspect | Groq (Cloud) | Ollama (Local) |
|--------|--------------|----------------|
| Rate Limits | ~30 req/min | **Unlimited** ✅ |
| Cost | Free tier limited | **Free** ✅ |
| Privacy | Data sent to cloud | **Local** ✅ |
| Speed | Fast (cloud GPUs) | Depends on hardware |

---

## Quick Setup

### 1. Install Ollama

**Windows**: Download from [ollama.ai/download](https://ollama.ai/download)

**macOS**:
```bash
brew install ollama
```

**Linux**:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull a Model

```powershell
ollama pull mistral
```

### 3. Verify Installation

```powershell
ollama list
```

Expected output:
```
NAME              ID              SIZE
llama3.1:latest   <hash>          4.7 GB
```

---

## Running the Project

### Ensure Ollama is Running

Ollama runs in background after installation. If not:

```powershell
ollama serve
```

### (Optional) Set FRED API Key for Real Macro Data

Get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html):

```powershell
$env:FRED_API_KEY = "your-fred-api-key"
```

> Without this key, the macro agent uses simulated data.

### Run Quick Test

```powershell
# Activate venv
venv\Scripts\activate

# Test with 3 days
python -m src.backtest --test --days 3
```

### Run Full Backtest

```powershell
python -m src.backtest
```

### Run Analysis

```powershell
python -m src.analysis
```

---

## Available Models

| Model | Command | Size | Quality |
|-------|---------|------|---------|
| **Mistral 7B** *(default)* | `ollama pull mistral` | 4.1 GB | Good |
| Llama 3.1 8B | `ollama pull llama3.1` | 4.7 GB | Good |
| Llama 3.1 70B | `ollama pull llama3.1:70b` | 40 GB | Best |
| Phi-3 | `ollama pull phi3` | 2.2 GB | Fast |

To use a different model:
```powershell
$env:OLLAMA_MODEL = "mistral"
python -m src.backtest --test --days 3
```

---

## Switching Between Backends

### Use Ollama (default)

```powershell
$env:LLM_BACKEND = "ollama"
```

### Use Groq

```powershell
$env:LLM_BACKEND = "groq"
$env:GROQ_API_KEY = "your-api-key"
```

---

## Configuration

Default settings in `src/config.py`:

```python
LLM_BACKEND = "ollama"              # Default backend
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"
OLLAMA_TEMPERATURE = 0.0            # Deterministic

# FRED API (Macroeconomic Data)
FRED_API_KEY = os.environ.get("FRED_API_KEY")  # Set via environment
```

### Environment Variables

```powershell
# LLM Backend
$env:LLM_BACKEND = "ollama"          # or "groq"
$env:OLLAMA_MODEL = "mistral"        # or "llama3.1"

# API Keys
$env:FRED_API_KEY = "your-key"       # For real macro data
$env:GROQ_API_KEY = "your-key"       # If using Groq backend
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Connection refused` | Run `ollama serve` |
| `Model not found` | Run `ollama pull mistral` |
| Slow performance | Use smaller model like `phi3` |
| Out of memory | Close other apps, use smaller model |

### Check GPU Usage

```powershell
# NVIDIA
nvidia-smi

# Check Ollama
ollama ps
```

---

## Performance Tips

1. **Use GPU**: Ollama auto-detects NVIDIA/Apple Silicon
2. **Close other apps**: Free up RAM
3. **Run overnight**: For 500+ days of analysis
4. **Use resume**: `python -m src.backtest --resume 2024-06-01`

---

## Estimated Runtime

With Llama 3.1 8B on modern hardware:

| Days | CPU Time | GPU Time |
|------|----------|----------|
| 5 | ~10 min | ~3 min |
| 50 | ~1.5 hr | ~30 min |
| 250 | ~8 hr | ~2.5 hr |

---

## Summary

| Step | Command |
|------|---------|
| Install Ollama | [ollama.ai](https://ollama.ai) |
| Pull model | `ollama pull mistral` |
| Verify | `ollama list` |
| Set FRED key (optional) | `$env:FRED_API_KEY = "your-key"` |
| Test | `python -m src.backtest --test --days 3` |
| Full run | `python -m src.backtest` |
| Analysis | `python -m src.analysis` |
