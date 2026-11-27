"""YouTube Data API v3 wrapper: cache first, then a daily quota budget stored on disk.

Only real (non-cached) calls spend budget. The budget day follows the API's quota reset
(midnight Pacific Time). search.list has its own bucket and its own budget.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from vmap import config
from vmap.http import Client, HttpError

BASE = "https://www.googleapis.com/youtube/v3/"
QUOTA_FILE = config.PROCESSED / "youtube_quota.json"


class QuotaExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, path=None, units: int | None = None, searches: int | None = None):
        s = config.settings()["youtube"]
        self.path = path or QUOTA_FILE
        self.limits = {"units": units if units is not None else s["daily_units_budget"],
                       "search": searches if searches is not None else s["daily_search_budget"]}
        self.tz = ZoneInfo(s.get("quota_timezone", "America/Los_Angeles"))

    def _today(self) -> str:
        return datetime.now(self.tz).date().isoformat()

    def _load(self) -> dict:
        state = {}
        if self.path.exists():
            state = json.loads(self.path.read_text())
        if state.get("date") != self._today():
            state = {"date": self._today(), "units": 0, "search": 0}
        return state

    def used(self) -> dict:
        return self._load()

    def remaining(self, bucket: str = "units") -> int:
        return self.limits[bucket] - self._load()[bucket]

    def spend(self, bucket: str, amount: int = 1) -> None:
        state = self._load()
        if state[bucket] + amount > self.limits[bucket]:
            raise QuotaExceeded(
                f"YouTube {bucket} budget for {state['date']} (Pacific) used up: "
                f"{state[bucket]}/{self.limits[bucket]}. Re-run tomorrow; cached calls are free.")
        state[bucket] += amount
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state))
        with open(config.RUN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "kind": "quota", "api": "youtube", "bucket": bucket,
                                "amount": amount, "used_today": state[bucket]}) + "\n")


class YouTube:
    def __init__(self, client: Client | None = None, budget: Budget | None = None):
        self.client = client or Client(stage="youtube")
        self.budget = budget or Budget()
        self.key = config.secret("YOUTUBE_KEY")

    def call(self, endpoint: str, params: dict, refresh: bool = False):
        """Returns (json, response). Raises QuotaExceeded before spending over budget.
        `refresh=True` ignores the cached copy (fresh statistics for one snapshot date)."""
        url = BASE + endpoint
        p = {**params, "key": self.key}
        r = None if refresh and not self.client.offline else self.client.cached(url, p)
        if r is None:
            if self.client.offline:
                r = self.client.get(url, p)        # raises OfflineError
            self.budget.spend("search" if endpoint == "search" else "units")
            try:
                r = self.client.get(url, p, refresh=refresh)
            except HttpError as e:
                if e.status == 403 and "quota" in str(e).lower():
                    raise QuotaExceeded(f"YouTube API reports quota exceeded: {e}") from e
                raise
        d = r.json()
        if "error" in d:
            raise HttpError(d["error"].get("code", 0), url, d["error"].get("message", ""))
        return d, r
