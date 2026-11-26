"""Single HTTP client for every network call in the project.

- disk cache (requests-cache, SQLite in data/cache/); API keys are excluded from cache keys
- descriptive User-Agent with contact email
- per-domain rate limit, applied only to real (non-cached) requests
- optional robots.txt check (for web pages; not for APIs)
- retries with exponential backoff on 429 / 5xx / connection errors
- every call appended to data/processed/run_log.jsonl (secrets redacted)
- offline mode (VMAP_OFFLINE=1): cached responses only, anything else raises
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode, urlsplit
from urllib.robotparser import RobotFileParser

import requests
import requests_cache
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from vmap import config

SECRET_PARAMS = ("key", "api_key", "apikey", "access_token", "token")
RETRY_STATUS = {429, 500, 502, 503, 504}


class OfflineError(RuntimeError):
    """Raised in offline mode when a response is not in the cache."""


class RobotsDisallowed(RuntimeError):
    pass


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status


def _redact(params: dict | None) -> dict:
    return {k: ("***" if k.lower() in SECRET_PARAMS else v) for k, v in (params or {}).items()}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, HttpError):
        return exc.status in RETRY_STATUS
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))


class Client:
    def __init__(self, stage: str = "misc", cache_name: str | None = None, offline: bool | None = None):
        s = config.settings().get("http", {})
        self.stage = stage
        self.min_interval = float(s.get("min_interval_s", 1.0))
        self.timeout = float(s.get("timeout_s", 30))
        self.retries = int(s.get("retries", 4))
        self.offline = offline if offline is not None else os.environ.get("VMAP_OFFLINE") == "1"
        expire = {pat: timedelta(days=d) for pat, d in (s.get("expire_days") or {}).items()}
        config.CACHE.mkdir(parents=True, exist_ok=True)
        self.session = requests_cache.CachedSession(
            cache_name or str(config.CACHE / "http_cache"),
            backend="sqlite",
            expire_after=requests_cache.NEVER_EXPIRE,
            urls_expire_after=expire,
            allowable_codes=(200, 404),
            ignored_parameters=list(SECRET_PARAMS),
            match_headers=False,
            stale_if_error=False,
        )
        contact = config.secret("CONTACT_EMAIL", required=False) or "contact-not-set"
        self.session.headers["User-Agent"] = (
            f"vmap/0.1 (research map of music video filming locations in Spain; {contact})"
        )
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self._robots: dict[str, RobotFileParser] = {}

    # ---- public API -------------------------------------------------------
    def get(self, url: str, params: dict | None = None, *, check_robots: bool = False,
            headers: dict | None = None) -> requests.Response:
        if check_robots:
            self._check_robots(url)
        cached = self.session.get(url, params=params, headers=headers, only_if_cached=True,
                                  timeout=self.timeout)
        if cached.status_code != 504 and getattr(cached, "from_cache", False):
            self._log(url, params, cached.status_code, from_cache=True)
            return cached
        if self.offline:
            raise OfflineError(f"Not in cache (offline mode): {url} {_redact(params)}")
        return self._fetch(url, params, headers)

    def get_json(self, url: str, params: dict | None = None, **kw) -> dict:
        return self.get(url, params, **kw).json()

    # ---- internals --------------------------------------------------------
    def _fetch(self, url, params, headers) -> requests.Response:
        @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(self.retries),
               wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
        def _do():
            self._throttle(urlsplit(url).netloc)
            r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            self._log(url, params, r.status_code, from_cache=getattr(r, "from_cache", False))
            if r.status_code not in (200, 404):
                raise HttpError(r.status_code, url, r.text)
            return r
        return _do()

    def _throttle(self, domain: str) -> None:
        with self._lock:
            wait = self._last.get(domain, 0) + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[domain] = time.monotonic()

    def _check_robots(self, url: str) -> None:
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        rp = self._robots.get(base)
        if rp is None:
            rp = RobotFileParser()
            try:
                r = self.get(base + "/robots.txt")
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except HttpError:
                rp.parse([])  # robots.txt unavailable -> treat as allowed
            self._robots[base] = rp
        if not rp.can_fetch(self.session.headers["User-Agent"], url):
            raise RobotsDisallowed(url)

    def _log(self, url, params, status, from_cache: bool) -> None:
        config.RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        q = urlencode(_redact(params), doseq=True)
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stage": self.stage,
            "kind": "http",
            "domain": urlsplit(url).netloc,
            "url": url + (f"?{q}" if q else ""),
            "status": status,
            "from_cache": from_cache,
        }
        with open(config.RUN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
