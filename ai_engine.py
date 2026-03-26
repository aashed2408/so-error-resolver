"""AI Engine — Ollama cloud/local connector with prompt engineering for error resolution.

Supports both local Ollama models and Ollama cloud-hosted models.
Cloud models are specified in ``provider/model:tag`` format, e.g. ``minimax/m2:5``.

The engine can signal that more scraping is needed by setting
:attr:`AIResolution.needs_more_data` to ``True`` and providing refined search
queries in :attr:`AIResolution.refined_queries`.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional

import ollama

from scraper import ScrapedQuestion


# ──────────────────────────── data models ─────────────────────────────────────

@dataclass
class AIResolution:
    """Structured output from the LLM analysis."""

    root_cause: str
    fix_recommendation: str
    confidence: str
    relevant_threads: list[str]
    raw_response: str
    needs_more_data: bool = False
    refined_queries: list[str] = field(default_factory=list)


# ──────────────────────────── prompt template ────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior software engineer and debugging specialist with deep expertise \
in diagnosing runtime errors, exceptions, and stack traces across all major \
programming languages and frameworks.

You will receive:
  1. A user's error message / traceback.
  2. Scraped Stack Overflow data (questions, answers, code snippets) — \
possibly from multiple scraping rounds.

Your task:
  A. Identify the **ROOT CAUSE** of the error in clear, precise language.
  B. Provide a **FIX RECOMMENDATION** — a step-by-step resolution with concrete \
code examples when applicable. This must be a real, working fix — not a guess.
  C. Rate your **CONFIDENCE** as exactly one of: High, Medium, or Low.
  D. List the **RELEVANT THREADS** (Stack Overflow URLs) that were most useful.
  E. If the scraped data is INSUFFICIENT to give a High-confidence answer, set \
**NEEDS_MORE_DATA: Yes** and provide 1-3 **REFINED_QUERIES** — alternative \
search terms that would find better Stack Overflow threads.

Rules:
  - Be direct and actionable. Zero filler.
  - Always wrap code suggestions in fenced code blocks with the language tag.
  - If multiple causes are possible, list all of them with probabilities.
  - Never invent Stack Overflow URLs — only cite URLs from the provided data.
  - If confidence is Low, you MUST provide refined queries to improve the answer.
  - A fix recommendation must include the actual code change, not just a description.
"""

_ITERATION_SUFFIX = """\
NOTE: This is iteration {iteration} of {max_iterations}. The scraper will keep \
looking for better data until confidence is High or max iterations are reached. \
If you need different search terms, provide them in REFINED_QUERIES.
"""


def _build_user_prompt(
    error_snippet: str,
    questions: list[ScrapedQuestion],
    *,
    iteration: int = 1,
    max_iterations: int = 5,
    previous_attempts: list[str] | None = None,
) -> str:
    """Craft the user-facing prompt from raw error + scraped data."""
    parts: list[str] = []

    parts.append("=" * 60)
    parts.append("USER ERROR / TRACEBACK:")
    parts.append("=" * 60)
    parts.append(error_snippet.strip())
    parts.append("")

    if previous_attempts:
        parts.append("=" * 60)
        parts.append("PREVIOUS SEARCH QUERIES TRIED (avoid repeating these):")
        parts.append("=" * 60)
        for q in previous_attempts:
            parts.append(f"  - {q}")
        parts.append("")

    parts.append("=" * 60)
    parts.append("SCRAPED STACK OVERFLOW DATA:")
    parts.append("=" * 60)

    if not questions:
        parts.append(
            "(No Stack Overflow threads were found for this error. "
            "Infer the root cause from the traceback alone, and suggest "
            "refined search queries.)"
        )
    else:
        for i, q in enumerate(questions, 1):
            parts.append(f"\n--- Thread {i} ---")
            parts.append(f"URL: {q.url}")
            parts.append(f"Title: {q.title}")
            parts.append(f"Votes: {q.votes}  |  Tags: {', '.join(q.tags)}")
            parts.append(f"Question Body:\n{q.body_text[:800]}")

            if q.code_blocks:
                parts.append("Question Code Snippet(s):")
                for cb in q.code_blocks[:2]:
                    parts.append(f"```python\n{cb[:500]}\n```")

            if q.answers:
                # Include accepted + top-voted answers
                for a_idx, ans in enumerate(q.answers[:2]):
                    label = "Accepted Answer" if ans.is_accepted else f"Answer #{a_idx + 1}"
                    parts.append(f"\n{label} (votes={ans.vote_count}, by {ans.author or 'unknown'}):")
                    parts.append(ans.body_text[:1000])
                    if ans.code_blocks:
                        parts.append("Answer Code:")
                        for cb in ans.code_blocks[:3]:
                            parts.append(f"```python\n{cb[:600]}\n```")
            parts.append("")

    parts.append(_ITERATION_SUFFIX.format(iteration=iteration, max_iterations=max_iterations))
    parts.append("")
    parts.append("=" * 60)
    parts.append("Respond in this EXACT format:")
    parts.append("")
    parts.append("ROOT CAUSE:")
    parts.append("<your precise diagnosis>")
    parts.append("")
    parts.append("FIX RECOMMENDATION:")
    parts.append("<step-by-step fix with code blocks>")
    parts.append("")
    parts.append("CONFIDENCE: <High | Medium | Low>")
    parts.append("")
    parts.append("RELEVANT THREADS:")
    parts.append("<bullet list of URLs that were most useful>")
    parts.append("")
    parts.append("NEEDS_MORE_DATA: <Yes | No>")
    parts.append("")
    parts.append("REFINED_QUERIES:")
    parts.append("<if NEEDS_MORE_DATA is Yes, list 1-3 alternative search queries>")
    parts.append("  - query 1")
    parts.append("  - query 2")
    parts.append("")
    return "\n".join(parts)


# ──────────────────────────── engine class ───────────────────────────────────


class AIEngine:
    """Wraps the Ollama Python client — works with both local and cloud models.

    Cloud models use the ``provider/model:tag`` naming convention
    (e.g. ``minimax/m2:5``). The Ollama server handles routing automatically.

    Parameters
    ----------
    model:
        Ollama model identifier. Use ``minimax/m2:5`` for the MiniMax cloud model,
        or ``llama3`` / ``mistral`` for local models.
    host:
        Ollama API endpoint. Default is ``http://localhost:11434``.
    timeout:
        Maximum seconds to wait for a single LLM call.
    max_retries:
        Retries per LLM call before giving up.
    """

    def __init__(
        self,
        *,
        model: str = "minimax/m2:5",
        host: str = "http://localhost:11434",
        timeout: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._host = host
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = ollama.AsyncClient(host=host)

    # ─────────────────── public API ────────────────────────────────

    async def analyze(
        self,
        error_snippet: str,
        questions: list[ScrapedQuestion],
        *,
        iteration: int = 1,
        max_iterations: int = 5,
        previous_queries: list[str] | None = None,
    ) -> AIResolution:
        """Send error + scraped data to Ollama and return a structured resolution.

        If the model signals it needs more data (low confidence + refined queries),
        the caller should scrape more threads and call this method again.
        """
        user_prompt = _build_user_prompt(
            error_snippet,
            questions,
            iteration=iteration,
            max_iterations=max_iterations,
            previous_attempts=previous_queries,
        )

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._client.chat(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        options={
                            "temperature": 0.15,
                            "num_predict": 4096,
                        },
                    ),
                    timeout=self._timeout,
                )
                raw: str = response["message"]["content"]
                return self._parse_response(raw, questions)
            except asyncio.TimeoutError:
                if attempt == self._max_retries:
                    return AIResolution(
                        root_cause="Ollama request timed out.",
                        fix_recommendation=(
                            "The LLM did not respond in time. "
                            "If using a cloud model, check your internet connection. "
                            "For local models, ensure `ollama serve` is running and the "
                            f"model '{self._model}' is loaded."
                        ),
                        confidence="Low",
                        relevant_threads=[q.url for q in questions],
                        raw_response="(timeout)",
                        needs_more_data=True,
                        refined_queries=[],
                    )
                await asyncio.sleep(3 * attempt)
            except Exception as exc:
                if attempt == self._max_retries:
                    return AIResolution(
                        root_cause=f"Ollama connection error: {exc}",
                        fix_recommendation=(
                            f"Could not connect to Ollama at {self._host}.\n\n"
                            "Troubleshooting:\n"
                            "1. Ensure Ollama is running: `ollama serve`\n"
                            "2. Verify the model is available: `ollama list`\n"
                            f"3. Pull the model if needed: `ollama pull {self._model}`\n"
                            "4. For cloud models, ensure Ollama has internet access."
                        ),
                        confidence="Low",
                        relevant_threads=[q.url for q in questions],
                        raw_response=str(exc),
                        needs_more_data=True,
                    )
                await asyncio.sleep(2 * attempt)

        return AIResolution(
            root_cause="Unknown error during LLM analysis.",
            fix_recommendation="Retry later.",
            confidence="Low",
            relevant_threads=[],
            raw_response="",
            needs_more_data=True,
        )

    async def check_connection(self) -> tuple[bool, str]:
        """Verify Ollama is reachable. Returns (connected, message)."""
        try:
            await asyncio.wait_for(
                self._client.list(), timeout=10.0
            )
            return True, "Connected to Ollama."
        except asyncio.TimeoutError:
            return False, f"Timed out connecting to Ollama at {self._host}."
        except Exception as exc:
            return False, f"Cannot reach Ollama: {exc}"

    async def pull_model(self) -> str:
        """Pull/download the model if not already available."""
        try:
            await asyncio.wait_for(
                self._client.pull(model=self._model, stream=False),
                timeout=300.0,
            )
            return f"Model '{self._model}' is ready."
        except Exception as exc:
            return f"Failed to pull model: {exc}"

    # ─────────────────── response parser ───────────────────────────

    @staticmethod
    def _parse_response(raw: str, questions: list[ScrapedQuestion]) -> AIResolution:
        """Extract structured fields from the LLM's free-form response."""
        root_cause: str = ""
        fix: str = ""
        confidence: str = "Medium"
        threads: list[str] = []
        needs_more: bool = False
        refined: list[str] = []

        current_section: Optional[str] = None
        buffer: list[str] = []

        def _flush() -> None:
            nonlocal root_cause, fix, confidence, threads, needs_more, refined
            text = "\n".join(buffer).strip()
            if not text:
                return
            if current_section == "ROOT CAUSE":
                root_cause = text
            elif current_section == "FIX RECOMMENDATION":
                fix = text
            elif current_section == "CONFIDENCE":
                c = text.strip().lower()
                if "high" in c:
                    confidence = "High"
                elif "low" in c:
                    confidence = "Low"
                else:
                    confidence = "Medium"
            elif current_section == "RELEVANT THREADS":
                for line in text.splitlines():
                    stripped = line.strip().lstrip("•-*– ")
                    urls = re.findall(r"https?://[^\s)>]+", stripped)
                    threads.extend(urls)
                    if not urls and "stackoverflow.com" in stripped:
                        threads.append(stripped)
            elif current_section == "NEEDS_MORE_DATA":
                needs_more = text.strip().lower().startswith("y")
            elif current_section == "REFINED_QUERIES":
                for line in text.splitlines():
                    stripped = line.strip().lstrip("•-*–\"' ")
                    if stripped and len(stripped) > 5:
                        refined.append(stripped)

        # Map section headers
        SECTION_MAP = {
            "ROOT CAUSE": "ROOT CAUSE",
            "ROOT_CAUSE": "ROOT CAUSE",
            "FIX RECOMMENDATION": "FIX RECOMMENDATION",
            "FIX:": "FIX RECOMMENDATION",
            "SOLUTION": "FIX RECOMMENDATION",
            "CONFIDENCE": "CONFIDENCE",
            "RELEVANT THREAD": "RELEVANT THREADS",
            "RELEVANT_THREAD": "RELEVANT THREADS",
            "REFERENCES": "RELEVANT THREADS",
            "NEEDS_MORE_DATA": "NEEDS_MORE_DATA",
            "NEEDS MORE DATA": "NEEDS_MORE_DATA",
            "REFINED_QUERIES": "REFINED_QUERIES",
            "REFINED QUERIES": "REFINED_QUERIES",
            "ALTERNATIVE SEARCH": "REFINED_QUERIES",
        }

        for line in raw.splitlines():
            upper = line.strip().upper()
            matched_section: Optional[str] = None
            for key, section in SECTION_MAP.items():
                if upper.startswith(key):
                    matched_section = section
                    break

            if matched_section:
                _flush()
                current_section = matched_section
                buffer = []
                # Capture text after the colon on the same line
                after = line.split(":", 1)
                if len(after) > 1 and after[1].strip():
                    buffer.append(after[1].strip())
            else:
                buffer.append(line)

        _flush()

        # Fallbacks
        if not threads:
            threads = [q.url for q in questions[:3]]
        if not root_cause:
            root_cause = "Unable to parse root cause from LLM response."
        if not fix:
            fix = raw

        # Auto-set needs_more if confidence is Low and no refined queries yet
        if confidence == "Low" and not refined:
            needs_more = True

        return AIResolution(
            root_cause=root_cause,
            fix_recommendation=fix,
            confidence=confidence,
            relevant_threads=threads,
            raw_response=raw,
            needs_more_data=needs_more,
            refined_queries=refined,
        )
