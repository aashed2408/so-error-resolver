"""Proxy Manager — rotation, health checks, and retry-aware proxy delivery."""

from __future__ import annotations

import asyncio
import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx


# ─────────────────────────── Placeholder proxy list ───────────────────────────
# Replace with real proxies or load from a file / env variable at runtime.
DEFAULT_PROXIES: list[str] = [
    "http://proxy1.example.com:8080",
    "http://proxy2.example.com:8080",
    "http://proxy3.example.com:8080",
    "http://proxy4.example.com:3128",
    "http://proxy5.example.com:9090",
]

# Fallback user-agents for header spoofing
USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) "
    "Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


@dataclass
class ProxyEntry:
    """Tracks a single proxy's health state."""

    url: str
    healthy: bool = True
    last_check: float = 0.0
    consecutive_failures: int = 0
    total_uses: int = 0

    # Mark unhealthy after this many consecutive failures
    FAIL_THRESHOLD: int = 3
    # Re-check unhealthy proxies after this many seconds (5 min)
    RECHECK_INTERVAL: float = 300.0


class ProxyRotator:
    """Round-robin proxy rotator with async health-checking and retry logic.

    Usage::

        rotator = ProxyRotator(proxies=["http://..."])
        proxy_url = await rotator.get_proxy()
        headers = rotator.get_headers()

    If no healthy proxies remain the rotator returns ``None`` (direct connection)
    so the caller can decide to proceed without a proxy.
    """

    def __init__(
        self,
        proxies: Optional[list[str]] = None,
        *,
        health_check_url: str = "https://httpbin.org/ip",
        health_check_timeout: float = 8.0,
    ) -> None:
        raw = proxies if proxies is not None else list(DEFAULT_PROXIES)
        self._entries: list[ProxyEntry] = [ProxyEntry(url=p) for p in raw]
        self._cycle = itertools.cycle(self._entries)
        self._health_check_url = health_check_url
        self._health_check_timeout = health_check_timeout
        self._lock = asyncio.Lock()

    # ─────────────────────── public helpers ────────────────────────

    async def get_proxy(self) -> Optional[str]:
        """Return the next healthy proxy URL, or ``None`` if all are down."""
        async with self._lock:
            attempts = len(self._entries)
            while attempts > 0:
                entry = next(self._cycle)
                attempts -= 1
                if entry.healthy:
                    entry.total_uses += 1
                    return entry.url
                # Maybe it's time to re-check a previously-unhealthy proxy
                if time.monotonic() - entry.last_check > ProxyEntry.RECHECK_INTERVAL:
                    is_ok = await self._check(entry)
                    if is_ok:
                        entry.total_uses += 1
                        return entry.url
            # All proxies exhausted — return None (caller can go direct)
            return None

    def get_headers(self) -> dict[str, str]:
        """Return a randomized set of HTTP headers to mimic a real browser."""
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "DNT": "1",
        }

    async def mark_failure(self, proxy_url: str) -> None:
        """Notify the rotator that a request through *proxy_url* failed."""
        async with self._lock:
            for entry in self._entries:
                if entry.url == proxy_url:
                    entry.consecutive_failures += 1
                    if entry.consecutive_failures >= ProxyEntry.FAIL_THRESHOLD:
                        entry.healthy = False
                    break

    async def mark_success(self, proxy_url: str) -> None:
        """Reset consecutive failure count after a successful request."""
        async with self._lock:
            for entry in self._entries:
                if entry.url == proxy_url:
                    entry.consecutive_failures = 0
                    entry.healthy = True
                    break

    async def health_check_all(self) -> dict[str, bool]:
        """Run a health check against every proxy and return a status map."""
        results: dict[str, bool] = {}
        tasks = [self._check(e) for e in self._entries]
        statuses = await asyncio.gather(*tasks, return_exceptions=True)
        for entry, status in zip(self._entries, statuses):
            results[entry.url] = bool(status) if not isinstance(status, BaseException) else False
        return results

    @property
    def healthy_count(self) -> int:
        return sum(1 for e in self._entries if e.healthy)

    @property
    def total_count(self) -> int:
        return len(self._entries)

    def stats(self) -> list[dict[str, object]]:
        """Return per-proxy statistics (useful for debugging)."""
        return [
            {
                "url": e.url,
                "healthy": e.healthy,
                "consecutive_failures": e.consecutive_failures,
                "total_uses": e.total_uses,
            }
            for e in self._entries
        ]

    # ─────────────────────── internal ──────────────────────────────

    async def _check(self, entry: ProxyEntry) -> bool:
        """Attempt to reach *health_check_url* through the proxy."""
        entry.last_check = time.monotonic()
        try:
            async with httpx.AsyncClient(
                proxy=entry.url,
                timeout=self._health_check_timeout,
                headers=self.get_headers(),
                follow_redirects=True,
            ) as client:
                resp = await client.get(self._health_check_url)
                if resp.status_code == 200:
                    entry.healthy = True
                    entry.consecutive_failures = 0
                    return True
        except (httpx.HTTPError, OSError):
            pass
        entry.consecutive_failures += 1
        if entry.consecutive_failures >= ProxyEntry.FAIL_THRESHOLD:
            entry.healthy = False
        return False
