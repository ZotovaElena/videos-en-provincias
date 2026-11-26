"""Offline tests for vmap.http: a fake transport adapter replaces the network."""
import io
import json

import pytest
from requests.adapters import HTTPAdapter
from urllib3 import HTTPResponse

from vmap import config, http


class FakeAdapter(HTTPAdapter):
    """Serves canned responses by path; counts real sends."""

    def __init__(self, routes):
        super().__init__()
        self.routes = routes  # path -> (status, body)
        self.calls = []

    def send(self, request, **kwargs):
        self.calls.append(request.url)
        path = request.path_url.split("?")[0]
        status, body = self.routes.get(path, (404, "not found"))
        raw = HTTPResponse(body=io.BytesIO(body.encode()), status=status, preload_content=False,
                           headers={"Content-Type": "text/plain"}, reason="X")
        return self.build_response(request, raw)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_LOG", tmp_path / "run_log.jsonl")
    monkeypatch.delenv("VMAP_OFFLINE", raising=False)
    c = http.Client(stage="test", cache_name=str(tmp_path / "cache"), offline=False)
    c.min_interval = 0
    adapter = FakeAdapter({"/ok": (200, '{"a": 1}'), "/robots.txt": (200, "User-agent: *\nDisallow: /private")})
    c.session.mount("https://", adapter)
    c.adapter = adapter
    return c


def test_second_call_served_from_cache(client):
    assert client.get_json("https://ex.org/ok", {"q": "x"}) == {"a": 1}
    r = client.get("https://ex.org/ok", {"q": "x"})
    assert r.from_cache
    assert len(client.adapter.calls) == 1


def test_api_key_not_part_of_cache_key_nor_logged(client):
    client.get("https://ex.org/ok", {"q": "x", "key": "SECRET1"})
    r = client.get("https://ex.org/ok", {"q": "x", "key": "SECRET2"})
    assert r.from_cache and len(client.adapter.calls) == 1
    log = config.RUN_LOG.read_text()
    assert "SECRET" not in log
    recs = [json.loads(line) for line in log.splitlines()]
    assert recs[0]["from_cache"] is False and recs[1]["from_cache"] is True


def test_offline_mode_raises_for_uncached(client):
    client.get("https://ex.org/ok")
    client.offline = True
    assert client.get("https://ex.org/ok").status_code == 200  # cached: fine
    with pytest.raises(http.OfflineError):
        client.get("https://ex.org/other")
    assert len(client.adapter.calls) == 1


def test_robots_disallow(client):
    with pytest.raises(http.RobotsDisallowed):
        client.get("https://ex.org/private/page", check_robots=True)
    assert client.get("https://ex.org/ok", check_robots=True).status_code == 200


def test_http_error_not_cached(client):
    client.adapter.routes["/boom"] = (403, "forbidden")
    with pytest.raises(http.HttpError):
        client.get("https://ex.org/boom")
    with pytest.raises(http.HttpError):
        client.get("https://ex.org/boom")
    assert len(client.adapter.calls) == 2  # 403 is not retried and not cached
