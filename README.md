# SO Error Resolver

Automatically resolve programming errors by scraping Stack Overflow and analyzing results with Ollama **cloud** models — no GPU, no model download.

## How It Works

```
Error Input → Clean Query → Scrape SO → Cloud AI Analysis → [Low Confidence?] → Refine Query → Re-scrape → ... → Fix Found
```

The tool runs an **iterative loop**:

1. **Scrape** Stack Overflow for threads matching your error.
2. **Analyze** scraped questions + answers with an Ollama cloud model.
3. **Evaluate** the AI's confidence level.
4. **Refine** — if confidence is Low, the AI suggests better search queries.
5. **Repeat** until a High-confidence fix is found or max iterations are reached.

## Prerequisites

- **Python 3.10+**
- **Ollama** installed — [https://ollama.ai](https://ollama.ai)
- **Internet connection** — cloud models run on Ollama's infrastructure

## Installation

```bash
git clone https://github.com/aashed2408/so-error-resolver.git
cd so-error-resolver
pip install -r requirements.txt
```

That's it. No `ollama pull` needed — cloud models are accessed remotely.

## Usage

### Start Ollama

```bash
ollama serve
```

### Run the Resolver

```bash
python main.py
```

Paste your error when prompted. Press Enter on an empty line to submit.

### Direct Error Input

```bash
python main.py -e "TypeError: 'NoneType' object is not subscriptable"
```

### Skip Proxy Rotation

```bash
python main.py --no-proxy
```

### Increase Iterations

```bash
python main.py --max-iterations 8
```

### Use a Different Cloud Model

```bash
python main.py -m minimax/m2:5
```

### Full CLI Reference

```
usage: so-error-resolver [-h] [-m MODEL] [--host HOST] [--no-proxy]
                         [-e ERROR] [--max-iterations MAX_ITERATIONS]

options:
  -m, --model MODEL           Ollama cloud model (default: minimax/m2:5)
  --host HOST                 Ollama API endpoint (default: http://localhost:11434)
  --no-proxy                  Skip proxy rotation — use direct connections
  -e, --error ERROR           Error message / traceback
  --max-iterations N          Maximum scrape-analyze-refine iterations (default: 5)
```

## Architecture

```
so-error-resolver/
├── main.py            # CLI + iterative orchestration loop
├── proxy_manager.py   # Proxy rotation, health checks, UA spoofing
├── scraper.py         # SO search + DuckDuckGo/Google fallbacks + extraction
├── ai_engine.py       # Ollama cloud connector, prompt engineering, parser
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable reference
└── .gitignore
```

### Module Details

#### `main.py`
- CLI interface using `rich` for formatted output.
- Orchestrates the iterative scrape → analyze → refine loop.
- Generates fallback search queries if the AI doesn't provide any.

#### `scraper.py`
- Searches Stack Overflow directly, falls back to DuckDuckGo, then Google.
- Extracts question titles, bodies, code blocks, answers (accepted + top-voted).
- Tracks seen URLs across iterations to avoid duplicate scraping.

#### `ai_engine.py`
- Connects to Ollama cloud models — no local model download required.
- Builds structured prompts with error + scraped data.
- Parses AI responses into structured `AIResolution` objects.
- Signals when more scraping is needed via `needs_more_data` + `refined_queries`.

#### `proxy_manager.py`
- Round-robin proxy rotation with async health checks.
- Marks proxies unhealthy after 3 consecutive failures.
- Randomizes User-Agent headers to mimic real browsers.

## Cloud Models

Cloud models run on Ollama's infrastructure — no GPU or local download required.

| Model | Default | Notes |
|-------|---------|-------|
| `minimax/m2:5` | Yes | Fast, good for code analysis |

Any Ollama-compatible cloud model works — just pass it with `-m`.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `minimax/m2:5` | Cloud model identifier |
| `MAX_ITERATIONS` | `5` | Max scrape-analyze loops |

## Error Handling

- **403 Forbidden**: Rotates proxy, retries with exponential back-off.
- **429 Rate Limit**: Waits 5s × attempt number before retry.
- **Ollama Timeout**: Retries up to 3 times, shows troubleshooting tips.
- **No Results**: Falls back from SO search → DuckDuckGo → Google.
- **Low Confidence**: Automatically refines search and retries.

## License

MIT
