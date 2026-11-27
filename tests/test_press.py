"""Offline tests for the press search stage and the places gazetteer."""
import pandas as pd
import pytest

from vmap import config, places
from vmap.sources import press

HTML = """<html><head><title>Noticia</title><meta property="og:site_name" content="El Diario Vasco"></head>
<body><nav>Menú portada deportes</nav><article><h1>La banda graba su videoclip en Getxo</h1>
<p>La banda madrileña ha elegido la playa de Ereaga, en Getxo, para rodar el videoclip de su nuevo
single, que se estrenará el próximo mes en todas las plataformas digitales.</p>
<p>El grupo actuará además en el festival de verano de la localidad con un concierto gratuito.</p>
<p>Según la productora, el vídeo musical se completó en dos jornadas de rodaje con vecinos como extras.</p>
</article><footer>Aviso legal</footer></body></html>"""


def test_html_passages_keeps_only_video_paragraphs():
    pars, site = press.html_passages(HTML, 500)
    assert site == "El Diario Vasco"
    assert any("playa de Ereaga, en Getxo" in p for p in pars)
    assert any("vídeo musical se completó" in p for p in pars)
    assert not any("festival de verano" in p for p in pars)


def test_html_passages_reads_bytes_with_declared_encoding():
    raw = HTML.replace("<head>", "<head><meta charset='iso-8859-1'>").encode("latin-1")
    pars, _ = press.html_passages(raw, 500)
    assert any("Getxo" in p and "vídeo" not in p or "Ereaga" in p for p in pars)
    assert not any("Ã" in p for p in pars)


def test_window_cuts_long_paragraph_around_keyword():
    par = "palabra " * 200 + "el videoclip se rodó en Teruel " + "otra " * 200
    out = press._window(par, 120)
    assert len(out) <= 120 and "videoclip" in out


def test_skip_url():
    skip = ["youtube.com", "instagram.com"]
    assert press.skip_url("https://www.youtube.com/watch?v=x", skip)
    assert press.skip_url("https://m.youtube.com/x", skip)
    assert not press.skip_url("https://www.diariovasco.com/a", skip)


def test_plan_order_and_scopes():
    df = pd.DataFrame([
        {"ine": "1", "name": "Pueblo", "variants": "Pueblo", "is_capital": False, "population": 500},
        {"ine": "2", "name": "Capital", "variants": "Capital|Capitala", "is_capital": True, "population": 90000},
        {"ine": "3", "name": "Ciudad", "variants": "Ciudad", "is_capital": False, "population": 50000},
    ])
    tpls = [{"q": "videoclip rodado en {lugar}", "scope": "all"},
            {"q": "music video filmed in {lugar} Spain", "scope": "capital"}]
    plan = press.plan_queries(df, tpls, min_population=20000)
    qs = [p["query"] for p in plan]
    assert qs[:4] == ["videoclip rodado en Capital", "videoclip rodado en Capitala",
                      "music video filmed in Capital Spain", "music video filmed in Capitala Spain"]
    assert qs[4:] == ["videoclip rodado en Ciudad", "videoclip rodado en Pueblo"]


def test_daily_budget_and_lifetime_cap(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(config, "RUN_LOG", tmp_path / "log.jsonl")
    b = press.DailyBudget(2, total_limit=3, path=tmp_path / "q.json")
    b.spend()
    b.spend()
    assert b.remaining() == 0
    with pytest.raises(press.BudgetExceeded):
        b.spend()
    st = json.loads((tmp_path / "q.json").read_text())
    (tmp_path / "q.json").write_text(json.dumps({**st, "date": "2000-01-01"}))   # next day
    assert b.remaining() == 1                     # daily reset, but only 1 left of the lifetime 3
    b.spend()
    assert b.remaining() == 0


class FakeClient:
    offline = False

    def __init__(self, response):
        self.response, self.calls = response, []

    def post_json(self, url, payload, headers=None):
        self.calls.append((url, payload, headers))
        return self.response

    def get_json(self, url, params, headers=None):
        self.calls.append((url, params, headers))
        return self.response


def test_tavily_and_brave_parsing():
    t = press.TavilyProvider(FakeClient({"results": [{"url": "u", "title": "t", "content": "c"}]}), "k", 50)
    assert t.search("q") == [{"url": "u", "title": "t", "snippet": "c"}]
    url, payload, headers = t.client.calls[0]
    assert payload["max_results"] == 20 and payload["country"] == "spain"
    assert headers == {"Authorization": "Bearer k"}
    b = press.BraveProvider(FakeClient({"web": {"results": [{"url": "u", "title": "t", "description": "d"}]}}),
                            "k", 20)
    assert b.search("q") == [{"url": "u", "title": "t", "snippet": "d"}]
    assert b.client.calls[0][1]["country"] == "ES"
    s = press.SerperProvider(FakeClient({"organic": [{"link": "u", "title": "t", "snippet": "s",
                                                      "position": 1}]}), "k", 20)
    assert s.search("q") == [{"url": "u", "title": "t", "snippet": "s", "date": None}]
    url, payload, headers = s.client.calls[0]
    assert payload == {"q": "q", "gl": "es", "hl": "es", "num": 10}
    assert headers["X-API-KEY"] == "k"


def test_name_variants():
    assert places.name_variants("Donostia/San Sebastián") == ["Donostia", "San Sebastián"]
    assert places.name_variants("el Pinós/Pinoso") == ["El Pinós", "Pinoso"]
