"""Press search: "videoclip rodado en {localidad}" → result pages → passages about music videos.

1. Query plan: templates × municipalities (config/search.yaml `press`), ordered by priority
   (province capitals → towns ≥ min_population → everything else, larger first).
2. Search through a configurable `SearchProvider` (Serper, Tavily or Brave). Every query is cached; cached
   queries are free. Real queries spend a daily budget stored on disk; the run stops when it is used up
   and continues from the same place next time.        → data/interim/press_results.jsonl
3. Each result page (robots.txt respected, 1 req/s per domain, cached) → main text (trafilatura) →
   paragraphs that mention a music video, cut to `context_chars`.  → evidence/press.jsonl
Records carry no song/artist/location yet (extractor="none"); the extraction stage fills them.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import urlsplit

from vmap import config, places
from vmap.extract.schema import Evidence, evidence_id
from vmap.http import Client, HttpError, OfflineError, RobotsDisallowed, retrieved_at

RESULTS_JSONL = config.INTERIM / "press_results.jsonl"
QUOTA_FILE = config.PROCESSED / "press_quota.json"


class BudgetExceeded(RuntimeError):
    pass


# ---------------------------------------------------------------- providers
class SearchProvider(Protocol):
    name: str

    def is_cached(self, query: str) -> bool: ...
    def search(self, query: str) -> list[dict]: ...   # [{url, title, snippet}]


class TavilyProvider:
    name = "tavily"
    URL = "https://api.tavily.com/search"

    def __init__(self, client: Client, key: str, n: int):
        self.client, self.key, self.n = client, key, min(n, 20)

    def _payload(self, query: str) -> dict:
        return {"query": query, "max_results": self.n, "search_depth": "basic", "topic": "general",
                "country": "spain"}

    def is_cached(self, query: str) -> bool:
        return self.client.is_cached_post(self.URL, self._payload(query))

    def search(self, query: str) -> list[dict]:
        d = self.client.post_json(self.URL, self._payload(query),
                                  headers={"Authorization": f"Bearer {self.key}"})
        return [{"url": r["url"], "title": r.get("title"), "snippet": r.get("content")}
                for r in d.get("results", [])]


class BraveProvider:
    name = "brave"
    URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, client: Client, key: str, n: int):
        self.client, self.key, self.n = client, key, min(n, 20)

    def _params(self, query: str) -> dict:
        return {"q": query, "count": self.n, "country": "ES", "search_lang": "es"}

    def is_cached(self, query: str) -> bool:
        return self.client.cached(self.URL, self._params(query)) is not None

    def search(self, query: str) -> list[dict]:
        d = self.client.get_json(self.URL, self._params(query),
                                 headers={"X-Subscription-Token": self.key, "Accept": "application/json"})
        return [{"url": r["url"], "title": r.get("title"), "snippet": r.get("description")}
                for r in d.get("web", {}).get("results", [])]


class SerperProvider:
    """Google results via serper.dev. Endpoint and body fields as used by its public integrations;
    confirmed on the first real call (see docs/sources.md)."""
    name = "serper"
    URL = "https://google.serper.dev/search"

    def __init__(self, client: Client, key: str, n: int):
        self.client, self.key, self.n = client, key, min(n, 10)

    def _payload(self, query: str) -> dict:
        return {"q": query, "gl": "es", "hl": "es", "num": self.n}

    def is_cached(self, query: str) -> bool:
        return self.client.is_cached_post(self.URL, self._payload(query))

    def search(self, query: str) -> list[dict]:
        d = self.client.post_json(self.URL, self._payload(query),
                                  headers={"X-API-KEY": self.key, "Content-Type": "application/json"})
        return [{"url": r["link"], "title": r.get("title"), "snippet": r.get("snippet"), "date": r.get("date")}
                for r in d.get("organic", [])]


PROVIDERS = {"serper": SerperProvider, "tavily": TavilyProvider, "brave": BraveProvider}


def make_provider(client: Client, cfg: dict) -> SearchProvider:
    name = (os.environ.get("SEARCH_PROVIDER") or cfg.get("provider", "serper")).split("#")[0].strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown SEARCH_PROVIDER {name!r}; choose one of {sorted(PROVIDERS)}")
    key = config.secret("SEARCH_API_KEY", required=not client.offline) or ""
    return PROVIDERS[name](client, key, cfg.get("results_per_query", 20))


# ---------------------------------------------------------------- budget
class DailyBudget:
    """Real (non-cached) queries per UTC day, plus an optional lifetime cap (e.g. one-time free credit)."""

    def __init__(self, limit: int, total_limit: int | None = None, path=QUOTA_FILE):
        self.limit, self.total_limit, self.path = limit, total_limit, path

    def _state(self) -> dict:
        today = datetime.now(timezone.utc).date().isoformat()
        st = json.loads(self.path.read_text()) if self.path.exists() else {}
        total = st.get("total", 0)
        return st if st.get("date") == today else {"date": today, "queries": 0, "total": total}

    def remaining(self) -> int:
        st = self._state()
        left = self.limit - st["queries"]
        if self.total_limit is not None:
            left = min(left, self.total_limit - st.get("total", 0))
        return left

    def spend(self) -> None:
        st = self._state()
        if self.remaining() <= 0:
            raise BudgetExceeded(f"press search budget used up: {st['queries']}/{self.limit} today, "
                                 f"{st.get('total', 0)}/{self.total_limit} in total")
        st["queries"] += 1
        st["total"] = st.get("total", 0) + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(st))
        with open(config.RUN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "kind": "quota", "api": "press", "used_today": st["queries"],
                                "used_total": st["total"]}) + "\n")


# ---------------------------------------------------------------- query plan
def plan_queries(df, templates: list[dict], min_population: int) -> list[dict]:
    """Ordered list of {query, ine, place, template}; one query per name variant."""
    def tier(row) -> int:
        if row["is_capital"]:
            return 0
        if row["population"] == row["population"] and row["population"] >= min_population:  # not NaN
            return 1
        return 2

    allowed = {"capital": {0}, "town": {0, 1}, "all": {0, 1, 2}}
    rows = df.assign(_tier=df.apply(tier, axis=1), _pop=df["population"].fillna(0))
    rows = rows.sort_values(["_tier", "_pop"], ascending=[True, False])
    plan, seen = [], set()
    for t in (0, 1, 2):
        for ti, tpl in enumerate(templates):
            if t not in allowed[tpl.get("scope", "all")]:
                continue
            for _, r in rows[rows["_tier"] == t].iterrows():
                for name in str(r["variants"]).split("|"):
                    q = tpl["q"].format(lugar=name)
                    if q not in seen:
                        seen.add(q)
                        plan.append({"query": q, "ine": r["ine"], "place": name, "template": ti, "tier": t})
    return plan


# ---------------------------------------------------------------- page → passages
KEYWORD = re.compile(r"(?i)\b(video\s*clip|videoclip|v[íi]deo\s*(?:musical|clip)|music\s+video|clip\s+musical)")


def _window(par: str, n: int) -> str:
    if len(par) <= n:
        return par
    m = KEYWORD.search(par)
    start = max(0, (m.start() if m else 0) - n // 3)
    chunk = par[start:start + n]
    if start > 0 and " " in chunk:
        chunk = chunk[chunk.index(" ") + 1:]
    return chunk[: chunk.rfind(" ")] if " " in chunk else chunk


def html_passages(html: str | bytes, n: int) -> tuple[list[str], str | None]:
    """Main-text paragraphs mentioning a music video (≤ n chars each) and the site name."""
    import trafilatura

    text = trafilatura.extract(html, include_comments=False, include_tables=False, favor_precision=True)
    meta = trafilatura.extract_metadata(html)
    site = getattr(meta, "sitename", None) if meta else None
    if not text:
        return [], site
    pars = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n")]
    return [_window(p, n) for p in pars if len(p) >= 40 and KEYWORD.search(p)], site


def _domain(url: str) -> str:
    d = urlsplit(url).netloc.lower()
    return d[4:] if d.startswith("www.") else d


def skip_url(url: str, skip_domains: list[str]) -> bool:
    d = _domain(url)
    return any(d == s or d.endswith("." + s) for s in skip_domains)


# ---------------------------------------------------------------- pipeline
def run(max_queries: int | None = None, dry_run: bool = False, cached_only: bool = False, log=print) -> dict:
    cfg = config.search()["press"]
    plan = plan_queries(places.load(), cfg["templates"], cfg["min_population"])
    if dry_run:
        tiers = {t: sum(1 for p in plan if p["tier"] == t) for t in (0, 1, 2)}
        return {"planned": len(plan), "by_tier": tiers, "first": [p["query"] for p in plan[:10]]}

    client = Client(stage="press")
    provider = make_provider(client, cfg)
    budget = DailyBudget(min(max_queries or cfg["max_queries_per_day"], cfg["max_queries_per_day"]),
                         cfg.get("max_queries_total"))

    # 1-2. queries: cached ones are free; stop at the first query that would exceed the budget
    results, done, stopped = [], 0, None
    for p in plan:
        if not provider.is_cached(p["query"]):
            if cached_only or budget.remaining() <= 0:
                stopped = p["query"]
                break
            budget.spend()
        try:
            hits = provider.search(p["query"])
        except HttpError as e:
            log(f"  search failed for {p['query']!r}: {e}")
            stopped = p["query"]
            break
        done += 1
        for rank, h in enumerate(hits):
            results.append({**p, **h, "rank": rank, "provider": provider.name})
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"queries done: {done}/{len(plan)}; results: {len(results)}"
        + (f"; next run starts at: {stopped!r}" if stopped else ""))

    # 3. pages → passages
    by_url: dict[str, dict] = {}
    for r in results:
        if not skip_url(r["url"], cfg["skip_domains"]):
            by_url.setdefault(r["url"], r)
    records, failed = [], {"robots": 0, "http": 0, "not_html": 0, "no_passage": 0}
    for i, (url, r) in enumerate(by_url.items(), 1):
        try:
            resp = client.get(url, check_robots=True)
        except RobotsDisallowed:
            failed["robots"] += 1
            continue
        except (HttpError, OfflineError, Exception) as e:  # noqa: BLE001 - any fetch failure is logged and skipped
            failed["http"] += 1
            if not isinstance(e, (HttpError, OfflineError)):
                log(f"  fetch error {url}: {type(e).__name__}")
            continue
        if resp.status_code != 200 or "html" not in resp.headers.get("Content-Type", ""):
            failed["not_html"] += 1
            continue
        pars, site = html_passages(resp.content, cfg["context_chars"])   # bytes: trafilatura finds the encoding
        if not pars:
            failed["no_passage"] += 1
        for par in pars:
            records.append(Evidence(
                evidence_id=evidence_id(url, par), source_type="press", source_url=url,
                source_name=site or _domain(url), retrieved_at=retrieved_at(resp), lang=None,
                evidence_text=par, query=r["query"], page_title=r.get("title"),
            ))
        if i % 50 == 0:
            log(f"  pages {i}/{len(by_url)}")
    return {"queries_done": done, "planned": len(plan), "results": len(results), "pages": len(by_url),
            "passages": len(records), "skipped_pages": failed, "records": records,
            "budget_left_today": budget.remaining()}
