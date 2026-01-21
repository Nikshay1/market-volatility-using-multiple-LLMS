# Running the Project with a Local LLM (Ollama)

This guide explains how to run the Market Volatility Analysis project using a locally installed LLM via Ollama. This approach eliminates API rate limits, allowing you to analyze **500+ days** of market data without interruption.

---

## Table of Contents

1. [Why Use a Local LLM?](#why-use-a-local-llm)
2. [Prerequisites](#prerequisites)
3. [Installing Ollama](#installing-ollama)
4. [Downloading a Model](#downloading-a-model)
5. [Configuring the Project](#configuring-the-project)
6. [Running the Project](#running-the-project)
7. [Code Changes Made](#code-changes-made)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tips](#performance-tips)

---

## Why Use a Local LLM?

| Aspect | Cloud API (Groq) | Local LLM (Ollama) |
|--------|------------------|-------------------|
| **Rate Limits** | ~30 requests/minute | **Unlimited** ✅ |
| **Max Days Analyzable** | ~20-30 days | **500+ days** ✅ |
| **Cost** | Free tier limited | **Completely free** ✅ |
| **Speed** | Fast (cloud GPUs) | Depends on your hardware |
| **Privacy** | Data sent to cloud | **Data stays local** ✅ |
| **Internet Required** | Yes | No (after model download) |

---

## Prerequisites

- **Python 3.9+** installed
- **8GB+ RAM** recommended (Mistral 7B uses ~4.1GB)
- **Storage**: ~5GB for the Mistral model
- **CPU or GPU**: Works on CPU, faster with GPU (NVIDIA/Apple Silicon)

---

## Installing Ollama

### Windows

1. Download the installer from: [https://ollama.ai/download](https://ollama.ai/download)
2. Run the installer and follow the prompts
3. Ollama will start automatically in the background

### macOS

```bash
brew install ollama
```

Or download from [https://ollama.ai/download](https://ollama.ai/download)

### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Verify Installation

Open a terminal and run:

```powershell
ollama --version
```

You should see the version number (e.g., `ollama version 0.1.x`).

---

## Downloading a Model

The project is configured to use the **Mistral 7B** model by default.

### Pull the Mistral Model

```powershell
ollama pull mistral
```

This downloads ~4.1GB. Wait for it to complete.

### Verify the Model is Available

```powershell
ollama list
```

Expected output:
```
NAME              ID              SIZE    MODIFIED
mistral:latest    <hash>          4.1 GB  <date>
```

### Alternative Models

You can use other models by changing the `OLLAMA_MODEL` setting:

| Model | Command | Size | Quality | Speed |
|-------|---------|------|---------|-------|
| **Mistral 7B** *(default)* | `ollama pull mistral` | 4.1 GB | Good | Fast |
| Llama 3 8B | `ollama pull llama3:8b` | 4.7 GB | Good | Fast |
| Llama 3 70B | `ollama pull llama3:70b` | 40 GB | Best | Slow |
| Phi-3 | `ollama pull phi3` | 2.2 GB | Okay | Fastest |

To use a different model, set the environment variable:

```powershell
$env:OLLAMA_MODEL = "llama3:8b"
```

---

## Configuring the Project

### Default Configuration

The project now defaults to using Ollama. The configuration in `src/config.py`:

```python
# LLM Backend: "groq" or "ollama"
LLM_BACKEND: str = os.environ.get("LLM_BACKEND", "ollama")  # Default: ollama

# Ollama Configuration
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "mistral"
OLLAMA_TEMPERATURE: float = 0.0
OLLAMA_MAX_TOKENS: int = 1024
```

### Switching to Groq (Cloud API)

If you want to use Groq instead:

```powershell
# Windows PowerShell
$env:LLM_BACKEND = "groq"
$env:GROQ_API_KEY = "your-api-key"

# Then run your command
python -m src.backtest --test --days 5
```

### Switching Back to Ollama

```powershell
$env:LLM_BACKEND = "ollama"
```

Or simply don't set the `LLM_BACKEND` variable (Ollama is the default).

---

## Running the Project

### Step 1: Ensure Ollama is Running

Ollama usually runs in the background after installation. If not:

```powershell
ollama serve
```

### Step 2: Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Run a Quick Test

```powershell
python -m src.backtest --test --days 5
```

You should see output like:
```
[FundamentalAgent] Using OLLAMA backend
[SentimentAgent] Using OLLAMA backend
Running quick test with 5 days...
============================================================
Starting Debate for 2024-01-02
============================================================
...
```

### Step 4: Run the Full Backtest (500+ Days)

Update the date range in `src/config.py`:

```python
START_DATE: str = "2023-01-01"  # Start date
END_DATE: str = "2024-12-31"    # End date (~500 trading days)
```

Then run:

```powershell
python -m src.backtest
```

**Estimated time**: With Mistral on a modern CPU, expect ~30-60 seconds per day, so 500 days ≈ 4-8 hours.

### Step 5: Run Analysis

After the backtest completes:

```powershell
python -m src.analysis
```

---

## Code Changes Made

This section documents the modifications made to support local LLM inference.

### 1. src/config.py

**Added LLM Backend Configuration:**

```python
# ============================================================
# LLM Backend Configuration
# ============================================================

# Options: "groq" or "ollama"
LLM_BACKEND: str = os.environ.get("LLM_BACKEND", "ollama")

# ============================================================
# Ollama Configuration (Local LLM)
# ============================================================

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_TEMPERATURE: float = 0.0
OLLAMA_MAX_TOKENS: int = 1024
```

### 2. src/agents.py

**Key Changes:**

1. **Added conditional imports for both backends:**
   ```python
   try:
       from groq import Groq
   except ImportError:
       Groq = None

   try:
       import ollama
   except ImportError:
       ollama = None
   ```

2. **Added `call_llm()` method that routes to the appropriate backend:**
   ```python
   def call_llm(self, prompt: str, max_retries: int = None) -> str:
       if self._backend == "ollama":
           return self._call_ollama(prompt, max_retries)
       else:
           return self._call_groq(prompt, max_retries)
   ```

3. **Added `_call_ollama()` method for local inference:**
   ```python
   def _call_ollama(self, prompt: str, max_retries: int = None) -> str:
       response = ollama.chat(
           model=config.OLLAMA_MODEL,
           messages=[
               {"role": "system", "content": self.system_prompt},
               {"role": "user", "content": prompt}
           ],
           options={
               "temperature": config.OLLAMA_TEMPERATURE,
               "num_predict": config.OLLAMA_MAX_TOKENS,
           }
       )
       return response['message']['content']
   ```

4. **Removed rate limiting delays for Ollama** (not needed for local inference)

### 3. requirements.txt

**Added:**
```
ollama
```

---

## Troubleshooting

### Error: "Ollama error: connection refused"

**Cause**: Ollama service is not running.

**Fix**:
```powershell
ollama serve
```

Or restart Ollama from the system tray (Windows).

---

### Error: "Model not found"

**Cause**: The specified model hasn't been downloaded.

**Fix**:
```powershell
ollama pull mistral
```

---

### Error: "Ollama library not installed"

**Cause**: Missing Python package.

**Fix**:
```powershell
pip install ollama
```

---

### Slow Performance

**Causes & Fixes**:

1. **Not enough RAM**: Close other applications
2. **Using CPU instead of GPU**: Ollama auto-detects GPU, but check:
   ```powershell
   ollama ps
   ```
3. **Large model**: Use a smaller model like `phi3`

---

### Inconsistent Agent Responses

**Cause**: Different models may produce different quality outputs.

**Fix**: Ensure `temperature` is set to `0.0` for deterministic outputs:
```python
OLLAMA_TEMPERATURE: float = 0.0
```

---

## Performance Tips

### 1. Use GPU Acceleration

If you have an NVIDIA GPU:
- Ollama automatically uses CUDA if available
- Verify GPU usage with `nvidia-smi`

For Apple Silicon Macs:
- GPU acceleration is automatic

### 2. Close Other Applications

Free up RAM for better performance:
```powershell
# Check RAM usage
Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 10 Name, @{N='RAM(MB)';E={[math]::Round($_.WorkingSet/1MB,2)}}
```

### 3. Run Overnight

For 500+ days of analysis:
- Start the backtest before leaving
- Use `--resume` if interrupted:
  ```powershell
  python -m src.backtest --resume 2024-06-15
  ```

### 4. Monitor Progress

The backtest saves checkpoints every 5 days to `data/disagreement_signals.csv`. You can check progress by viewing this file.

---

## Summary

| To Do | Command |
|-------|---------|
| Install Ollama | Download from [ollama.ai](https://ollama.ai) |
| Download model | `ollama pull mistral` |
| Verify model | `ollama list` |
| Install dependencies | `pip install -r requirements.txt` |
| Quick test | `python -m src.backtest --test --days 5` |
| Full backtest | `python -m src.backtest` |
| Analysis | `python -m src.analysis` |

**You're all set!** 🎉 You can now run the analysis on 500+ days without hitting any rate limits.
