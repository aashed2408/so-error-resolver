"""Stack Overflow scraper — search, list, and extract questions + answers.

Supports iterative scraping: the scraper tracks already-seen URLs and can
accept refined search queries from the AI engine to broaden the search until
a high-confidence fix is found.
"""

from __future__ import annotations

import asyncio
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

import httpx
from bs4 import BeautifulSoup, Tag

from proxy_manager import ProxyRotator

# ──────────────────────────── constants ──────────────────────────────────────

SO_BASE = "https://stackoverflow.com"
SEARCH_URL = SO_BASE + "/search"

# HTTP status codes we treat specially
FORBIDDEN = 403
RATE_LIMIT = 429
SERVICE_UNAVAILABLE = 503


@dataclass
class ScrapedAnswer:
    """A single Stack Overflow answer."""

    body_text: str
    code_blocks: list[str]
    vote_count: int = 0
    is_accepted: bool = False
    author: str = ""


@dataclass
class ScrapedQuestion:
    """A single Stack Overflow question with its top answers."""

    url: str
    title: str
    body_text: str
    code_blocks: list[str]
    votes: int = 0
    answers: list[ScrapedAnswer] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


# ──────────────────────────── helpers ────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", raw).strip()


def _extract_code_blocks(tag: Tag) -> list[str]:
    """Pull every <code> (or <pre><code>) block from *tag*."""
    blocks: list[str] = []
    for pre in tag.find_all("pre"):
        code = pre.find("code")
        blocks.append(code.get_text() if code else pre.get_text())
    # Also grab inline <code> with multi-line content
    for code_tag in tag.find_all("code"):
        text = code_tag.get_text()
        if text not in blocks and "\n" in text:
            blocks.append(text)
    return [b.strip() for b in blocks if b.strip()]


def _extract_post_text(tag: Tag) -> str:
    """Return the human-readable text of a post body, stripping code."""
    clone = BeautifulSoup(str(tag), "html.parser")
    for pre in clone.find_all("pre"):
        pre.decompose()
    return _clean_text(clone.get_text())


# ──────────────────────────── scraper class ──────────────────────────────────


class StackOverflowScraper:
    """Asynchronous Stack Overflow scraper with proxy rotation, retries,
    and support for iterative scraping with refined queries.

    The scraper maintains a set of already-scraped URLs to avoid duplicate
    work across multiple iterations of the search-analyze loop.

    Parameters
    ----------
    proxy_rotator:
        Manages proxy rotation and header spoofing.
    max_threads:
        Maximum concurrent requests when scraping question pages.
    request_timeout:
        HTTP request timeout in seconds.
    max_retries:
        Number of retry attempts per HTTP request before giving up.
    """

    def __init__(
        self,
        proxy_rotator: ProxyRotator,
        *,
        max_threads: int = 5,
        request_timeout: float = 20.0,
        max_retries: int = 3,
    ) -> None:
        self._rotator = proxy_rotator
        self._max_threads = max_threads
        self._timeout = request_timeout
        self._max_retries = max_retries
        self._seen_urls: set[str] = set()
        self._all_questions: list[ScrapedQuestion] = []

    # ─────────────────── public API ────────────────────────────────

    async def search_and_extract(
        self,
        error_message: str,
        *,
        max_results: int = 5,
        refined_queries: list[str] | None = None,
    ) -> list[ScrapedQuestion]:
        """Search Stack Overflow and extract question + answer content.

        On the first call, the primary query is derived from the error message.
        On subsequent calls, pass ``refined_queries`` from the AI engine to
        broaden the search. The scraper skips already-seen URLs automatically.

        Parameters
        ----------
        error_message:
            The raw error / traceback text.
        max_results:
            Maximum new questions to fetch per iteration.
        refined_queries:
            Optional list of alternative search queries suggested by the AI.

        Returns
        -------
        list[ScrapedQuestion]
            Newly scraped questions (not seen in previous iterations).
        """
        all_queries: list[str] = []

        # Primary query from error message
        primary = self._clean_error(error_message)
        all_queries.append(primary)

        # Add refined queries (avoid duplicates)
        if refined_queries:
            for rq in refined_queries:
                rq_clean = rq.strip().strip('"').strip("'")
                if rq_clean and rq_clean not in all_queries:
                    all_queries.append(rq_clean)

        new_urls: list[str] = []
        for query in all_queries:
            if len(new_urls) >= max_results:
                break
            urls = await self._search(query, max_results=max_results)
            for u in urls:
                if u not in self._seen_urls and u not in new_urls:
                    new_urls.append(u)
                if len(new_urls) >= max_results:
                    break

        if not new_urls:
            return []

        sem = asyncio.Semaphore(self._max_threads)

        async def _limited(url: str) -> Optional[ScrapedQuestion]:
            async with sem:
                return await self._scrape_question(url)

        tasks = [_limited(u) for u in new_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_questions: list[ScrapedQuestion] = []
        for r in results:
            if isinstance(r, ScrapedQuestion):
                new_questions.append(r)
                self._all_questions.append(r)

        return new_questions

    def get_all_questions(self) -> list[ScrapedQuestion]:
        """Return every question scraped across all iterations."""
        return list(self._all_questions)

    @property
    def total_scraped(self) -> int:
        """Number of unique questions scraped so far."""
        return len(self._all_questions)

    def reset(self) -> None:
        """Clear all cached data for a fresh session."""
        self._seen_urls.clear()
        self._all_questions.clear()

    # ─────────────────── error cleaning ────────────────────────────

    @staticmethod
    def _clean_error(raw: str) -> str:
        """Strip traceback noise and produce a concise search query."""
        lines = raw.strip().splitlines()
        meaningful: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("File ") and stripped.endswith('.py"'):
                continue
            if stripped.startswith("^") or stripped.startswith("~"):
                continue
            if stripped.startswith("During") or stripped.startswith("The above"):
                continue
            meaningful.append(stripped)
        # Prefer the last meaningful line (usually the exception type + message)
        if meaningful:
            query = meaningful[-1]
        else:
            query = raw.strip()
        # Remove common prefixes
        query = re.sub(r"^(raise\s+|Traceback.*?:\s*)", "", query)
        # Remove module paths like "module.submodule.ClassName:"
        query = re.sub(r"^[\w.]+\.", "", query)
        return query[:200]

    # ─────────────────── search ────────────────────────────────────

    async def _search(self, query: str, *, max_results: int) -> list[str]:
        """Scrape the Stack Overflow search results page and return question URLs."""
        params = {"q": query, "tab": "relevance"}
        url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"
        html = await self._fetch(url)
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        # SO search results — multiple selector strategies for resilience
        for selector in [
            "a.s-link",
            "a.question-hyperlink",
            "h3 a",
            "div.result-container a",
        ]:
            for a_tag in soup.select(selector):
                href = a_tag.get("href", "")
                if "/questions/" in href:
                    full = href if href.startswith("http") else SO_BASE + href
                    # Normalize: strip query params
                    full = full.split("?")[0]
                    if full not in links:
                        links.append(full)
                    if len(links) >= max_results:
                        break
            if links:
                break
        # Fallback: DuckDuckGo
        if not links:
            links = await self._duckduckgo_search(query, max_results=max_results)
        # Fallback: Google (HTML scraping)
        if not links:
            links = await self._google_search(query, max_results=max_results)
        return links

    async def _duckduckgo_search(
        self, query: str, *, max_results: int
    ) -> list[str]:
        """Fallback: scrape DuckDuckGo HTML results for Stack Overflow links."""
        encoded = urllib.parse.quote_plus(f"site:stackoverflow.com {query}")
        ddg_url = f"https://html.duckduckgo.com/html/?q={encoded}"
        html = await self._fetch(ddg_url)
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for a_tag in soup.select("a.result__a"):
            href = a_tag.get("href", "")
            if "stackoverflow.com/questions/" in href:
                href = href.split("?")[0]
                if href not in links:
                    links.append(href)
                if len(links) >= max_results:
                    break
        return links

    async def _google_search(
        self, query: str, *, max_results: int
    ) -> list[str]:
        """Last-resort fallback: scrape Google for Stack Overflow links."""
        encoded = urllib.parse.quote_plus(f"site:stackoverflow.com {query}")
        google_url = f"https://www.google.com/search?q={encoded}&num={max_results}"
        html = await self._fetch(google_url)
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for a_tag in soup.select("a"):
            href = a_tag.get("href", "")
            # Google wraps URLs in /url?q=...
            match = re.search(r"/url\?q=(https?://stackoverflow\.com/questions/[^\s&]+)", href)
            if match:
                link = match.group(1)
                link = link.split("?")[0]
                if link not in links:
                    links.append(link)
                if len(links) >= max_results:
                    break
        return links

    # ─────────────────── question page scraping ────────────────────

    async def _scrape_question(self, url: str) -> Optional[ScrapedQuestion]:
        """Fetch a single question page and extract content + answers."""
        self._seen_urls.add(url)
        html = await self._fetch(url)
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")

        # ── Title ──
        title_tag = soup.select_one("h1 a, h1")
        title = _clean_text(title_tag.get_text()) if title_tag else url

        # ── Question body ──
        q_body_tag = soup.select_one(
            "div.s-prose.js-post-body, div.post-text"
        )
        if q_body_tag is None:
            return None
        q_text = _extract_post_text(q_body_tag)
        q_code = _extract_code_blocks(q_body_tag)

        # ── Votes ──
        votes = 0
        vote_tag = soup.select_one(
            "div.js-vote-count, button.js-vote-count"
        )
        if vote_tag:
            try:
                votes = int(vote_tag.get_text(strip=True))
            except ValueError:
                pass

        # ── Tags ──
        tags = [
            a.get_text(strip=True)
            for a in soup.select("div.post-taglist a.post-tag, a.post-tag")
        ]

        # ── Answers ──
        answers: list[ScrapedAnswer] = []
        answer_divs = soup.select("div.answer, div.js-answer")
        for ad in answer_divs:
            a_body = ad.select_one(
                "div.s-prose.js-post-body, div.post-text"
            )
            if a_body is None:
                continue
            a_text = _extract_post_text(a_body)
            a_code = _extract_code_blocks(a_body)
            a_votes = 0
            a_vote_tag = ad.select_one(
                "div.js-vote-count, button.js-vote-count"
            )
            if a_vote_tag:
                try:
                    a_votes = int(a_vote_tag.get_text(strip=True))
                except ValueError:
                    pass
            is_accepted = "accepted-answer" in (ad.get("class") or [])
            author_tag = ad.select_one("div.user-details a")
            author = author_tag.get_text(strip=True) if author_tag else ""
            answers.append(
                ScrapedAnswer(
                    body_text=a_text,
                    code_blocks=a_code,
                    vote_count=a_votes,
                    is_accepted=is_accepted,
                    author=author,
                )
            )

        # Sort: accepted first, then by votes descending
        answers.sort(key=lambda a: (-int(a.is_accepted), -a.vote_count))

        return ScrapedQuestion(
            url=url,
            title=title,
            body_text=q_text,
            code_blocks=q_code,
            votes=votes,
            answers=answers,
            tags=tags,
        )

    # ─────────────────── HTTP layer ────────────────────────────────

    async def _fetch(self, url: str) -> Optional[str]:
        """GET *url* through a rotated proxy with retry + exponential back-off."""
        for attempt in range(1, self._max_retries + 1):
            proxy = await self._rotator.get_proxy()
            try:
                async with httpx.AsyncClient(
                    proxy=proxy,
                    timeout=self._timeout,
                    headers=self._rotator.get_headers(),
                    follow_redirects=True,
                ) as client:
                    resp = await client.get(url)
                    if resp.status_code == FORBIDDEN:
                        if proxy:
                            await self._rotator.mark_failure(proxy)
                        await asyncio.sleep(2 * attempt)
                        continue
                    if resp.status_code == RATE_LIMIT:
                        await asyncio.sleep(5 * attempt)
                        continue
                    if resp.status_code >= 400:
                        if proxy:
                            await self._rotator.mark_failure(proxy)
                        continue
                    if proxy:
                        await self._rotator.mark_success(proxy)
                    return resp.text
            except (httpx.TimeoutException, httpx.HTTPError, OSError):
                if proxy:
                    await self._rotator.mark_failure(proxy)
                await asyncio.sleep(1.5 * attempt)
        return None
