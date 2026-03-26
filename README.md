# SO Error Resolver

Automatically resolve programming errors by scraping Stack Overflow and analyzing results with Ollama AI models — cloud or local.

## How It Works

```
Error Input → Clean Query → Scrape SO → AI Analysis → [Low Confidence?] → Refine Query → Re-scrape → ... → Fix Found
```

The tool runs an **iterative loop**:

1. **Scrape** Stack Overflow for threads matching your error.
2. **Analyze** scraped questions + answers with an Ollama model.
3. **Evaluate** the AI's confidence level.
4. **Refine** — if confidence is Low, the AI suggests better search queries.
5. **Repeat** until a High-confidence fix is found or max iterations are reached.

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running — [https://ollama.ai](https://ollama.ai)

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/so-error-resolver.git
cd so-error-resolver

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running
ollama serve

# Pull the default model (cloud-hosted, no GPU required)
ollama pull minimax/m2:5
```

## Usage

### Interactive Mode

```bash
python main.py
```

Paste your error when prompted. Press Enter on an empty line to submit.

### Direct Error Input

```bash
python main.py -e "TypeError: 'NoneType' object is not subscriptable"
```

### Use a Local Model

```bash
python main.py -m llama3
python main.py -m mistral
```

### Skip Proxy Rotation

```bash
python main.py --no-proxy
```

### Increase Iterations

```bash
python main.py --max-iterations 8
```

### Full CLI Reference

```
usage: so-error-resolver [-h] [-m MODEL] [--host HOST] [--no-proxy] [-e ERROR]
                         [--max-iterations MAX_ITERATIONS]

Resolve programming errors by iteratively scraping Stack Overflow and analyzing
results with Ollama AI models (cloud or local).

options:
  -m, --model MODEL           Ollama model identifier (default: minimax/m2:5)
  --host HOST                 Ollama API endpoint (default: http://localhost:11434)
  --no-proxy                  Skip proxy rotation — use direct connections
  -e, --error ERROR           Error message / traceback
  --max-iterations N          Maximum scrape-analyze-refine iterations (default: 5)
```

## Architecture

```
so-error-resolver/
├── main.py            # CLI entry point + iterative orchestration loop
├── proxy_manager.py   # Proxy rotation, health checks, UA spoofing
├── scraper.py         # SO search + DuckDuckGo/Google fallbacks + extraction
├── ai_engine.py       # Ollama connector, prompt engineering, response parser
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable reference
└── .gitignore
```

### Module Details

#### `main.py`
- CLI interface using `rich` for formatted output.
- Orchestrates the iterative scrape → analyze → refine loop.
- Handles keyboard interrupts gracefully.

#### `scraper.py`
- Searches Stack Overflow directly, falls back to DuckDuckGo, then Google.
- Extracts question titles, bodies, code blocks, answers (accepted + top-voted).
- Tracks seen URLs across iterations to avoid duplicate scraping.
- Accepts refined queries from the AI engine for broadened search.

#### `ai_engine.py`
- Connects to Ollama (supports both cloud and local models).
- Builds structured prompts with error + scraped data.
- Parses AI responses into `AIResolution` objects with root cause, fix, confidence, and refined queries.
- Supports cloud model naming: `minimax/m2:5`, `anthropic/claude-3.5-sonnet`, etc.

#### `proxy_manager.py`
- Round-robin proxy rotation with async health checks.
- Marks proxies unhealthy after 3 consecutive failures.
- Re-checks unhealthy proxies every 5 minutes.
- Randomizes User-Agent headers to mimic real browsers.

## Models

### Cloud Models (Recommended)

Cloud models run on Ollama's infrastructure — no GPU required.

| Model | Command | Notes |
|-------|---------|-------|
| MiniMax M2 | `ollama pull minimax/m2:5` | Default, fast, good for code |
| Claude 3.5 Sonnet | `ollama pull anthropic/claude-3.5-sonnet` | High quality reasoning |
| GPT-4o | `ollama pull openai/gpt-4o` | OpenAI via Ollama cloud |

### Local Models

Local models require sufficient RAM/VRAM.

| Model | Command | RAM Needed |
|-------|---------|------------|
| Llama 3 8B | `ollama pull llama3` | ~8 GB |
| Mistral 7B | `ollama pull mistral` | ~6 GB |
| CodeLlama | `ollama pull codellama` | ~8 GB |

## Configuration

Environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `minimax/m2:5` | Model to use |
| `MAX_ITERATIONS` | `5` | Max scrape-analyze loops |
| `MAX_RETRIES` | `3` | HTTP request retries |
| `MAX_THREADS` | `5` | Concurrent scraper threads |
| `REQUEST_TIMEOUT` | `180` | LLM timeout (seconds) |

## Error Handling

- **403 Forbidden**: Rotates proxy, retries with exponential back-off.
- **429 Rate Limit**: Waits 5s × attempt number before retry.
- **Ollama Timeout**: Retries up to 3 times, shows troubleshooting tips.
- **No Results**: Falls back from SO search → DuckDuckGo → Google.
- **Low Confidence**: Automatically refines search and retries.

## License

MIT
