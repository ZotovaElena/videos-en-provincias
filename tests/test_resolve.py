"""Offline tests for Phase 5: clip linking, place → unit resolution, status and aggregation."""
import pandas as pd
import pytest

from vmap.extract.schema import Evidence
from vmap.resolve import geo as geomod
from vmap.resolve import link, pipeline

TOP = pd.DataFrame([
    {"youtube_id": "A1", "artist": "Shakira", "song_title_guess": "Chantaje", "view_count": 3_000_000_000,
     "title": "Shakira - Chantaje", "channel_title": "shakiraVEVO", "views_snapshot_date": "2026-11-26",
     "evidence_urls": "https://es.wikipedia.org/wiki/Chantaje"},
    {"youtube_id": "A2", "artist": "Shakira", "song_title_guess": "Chantaje (Versión Salsa)",
     "view_count": 37_000_000, "title": "x", "channel_title": "c", "views_snapshot_date": "2026-11-26",
     "evidence_urls": None},
    {"youtube_id": "B1", "artist": "Naiara", "song_title_guess": "Kenbo", "view_count": 2_000_000,
     "title": "x", "channel_title": "c", "views_snapshot_date": "2026-11-26", "evidence_urls": None},
    {"youtube_id": "B2", "artist": "Naiara", "song_title_guess": "Amor", "view_count": 3_000_000,
     "title": "x", "channel_title": "c", "views_snapshot_date": "2026-11-26", "evidence_urls": None},
])


def ev(text="El videoclip se rodó en Getxo.", **kw):
    base = {"evidence_id": "e", "source_type": "press", "source_url": "https://www.elcorreo.com/a",
            "source_name": "x", "retrieved_at": "2026-01-01", "evidence_text": text}
    return Evidence(**{**base, **kw})


@pytest.fixture(scope="module")
def linker():
    return link.Linker(TOP)


def test_link_methods(linker):
    assert linker.link(ev(youtube_ids=["A1", "ZZ"])) == (["A1"], "youtube_id")
    # a song page linking another song's video: no youtube_id link
    assert linker.link(ev(youtube_ids=["B2"], song_title="Kenbo"))[1] != "youtube_id"
    # several songs' videos on one page: only the record's song counts
    assert linker.link(ev(youtube_ids=["B1", "B2"], song_title="Kenbo")) == (["B1"], "youtube_id")
    assert linker.link(ev(source_url="https://es.wikipedia.org/wiki/Chantaje", page_kind="song"))[1] == "article_url"
    assert linker.link(ev(artist="Shakira", song_title="Chantaje")) == (["A1"], "artist_title")
    assert linker.link(ev(text="Naiara rueda en Getxo el videoclip de Kenbo.", artist="Naiara")) == \
        (["B1"], "title_in_text")
    assert linker.link(ev(artist="Desconocido", song_title="Luz")) == ([], "text_only")
    assert linker.link(ev(artist="Naiara")) == ([], "unlinked")


def test_versions_are_separate_clips(linker):
    a = linker.clip_for(ev(), ["A1"])
    b = linker.clip_for(ev(), ["A2"])
    assert a["clip_id"] != b["clip_id"] and a["song_title"] == "Chantaje"
    assert link.clip_id("Shakira", "Chantaje (Official Video)") == a["clip_id"]


def test_near_duplicates_go_to_review():
    clips = pd.DataFrame([{"clip_id": "1", "artist": "Estopa", "song_title": "Como Camarón"},
                          {"clip_id": "2", "artist": "Estopa", "song_title": "Como Camaron!"},
                          {"clip_id": "3", "artist": "Estopa", "song_title": "La raja de tu falda"}])
    d = link.near_duplicates(clips)
    assert len(d) == 0 or set(d[["clip_id_a", "clip_id_b"]].iloc[0]) == {"1", "2"}
    clips.loc[1, "song_title"] = "Como Camarones"
    assert len(link.near_duplicates(clips)) == 1


class FakeGeo(geomod.GeoResolver):
    """Resolver with a tiny municipality table and no network."""

    def __init__(self):
        self.munis = pd.DataFrame([
            {"ine": "48044", "unit_id": "48", "lat": 43.35, "lon": -3.0},
            {"ine": "35024", "unit_id": "35-LAN", "lat": 29.06, "lon": -13.55},
            {"ine": "30027", "unit_id": "30", "lat": 38.0, "lon": -1.5},
            {"ine": "16133", "unit_id": "16", "lat": 39.8, "lon": -1.4},
        ]).set_index("ine")
        self.province_name = {"48": "Bizkaia"}
        self.geocode_errors, self.offline_ok = 0, True

    def geocode(self, q):
        return []


def cand(kind, code, provinces):
    return {"kind": kind, "code": code, "name": code, "provinces": provinces}


def test_geo_resolution_by_kind():
    g = FakeGeo()
    r = g.resolve({"mention": "Getxo", "candidates": [cand("municipality", "48044", ["48"])]}, set())
    assert r["unit_ids"] == ["48"] and r["ine"] == "48044" and not r["reasons"]
    # municipality in an island province → its island, not the province
    r = g.resolve({"mention": "Teguise", "candidates": [cand("municipality", "35024", ["35"])]}, set())
    assert r["unit_ids"] == ["35-LAN"]
    r = g.resolve({"mention": "La Graciosa", "candidates": [cand("island", "35-GRA", ["35"])]}, set())
    assert r["unit_ids"] == ["35-GRA"]
    # "Madrid": municipality + province + region → the municipality wins
    r = g.resolve({"mention": "Bilbao", "candidates": [cand("region", "16", ["01", "20", "48"]),
                                                       cand("municipality", "48044", ["48"])]}, set())
    assert r["precision"] == "municipality"
    r = g.resolve({"mention": "Baleares", "candidates": [cand("province", "07", ["07"])]}, set())
    assert r["unit_ids"] == [] and r["reasons"] == ["island_province_level"]
    r = g.resolve({"mention": "Galicia", "candidates": [cand("region", "12", ["15", "27", "32", "36"])]}, set())
    assert r["reasons"] == ["region_level"]
    r = g.resolve({"mention": "La Rioja", "candidates": [cand("region", "17", ["26"])]}, set())
    assert r["unit_ids"] == ["26"]


def test_duplicate_town_needs_context():
    g = FakeGeo()
    p = {"mention": "Moya", "candidates": [cand("municipality", "30027", ["30"]), cand("municipality", "16133", ["16"])]}
    assert g.resolve(p, set())["reasons"] == ["several_municipalities"]
    assert g.resolve(p, {"16"})["unit_ids"] == ["16"]


def test_nominatim_filter():
    assert geomod._ok_result({"category": "natural", "type": "beach"})
    assert geomod._ok_result({"category": "place", "type": "village"})
    assert not geomod._ok_result({"category": "amenity", "type": "bar"})
    assert not geomod._ok_result({"category": "highway", "type": "residential"})


def test_status_and_aggregation(linker):
    g = FakeGeo()
    good = ev(artist="Shakira", song_title="Chantaje", partial=False,
              places=[{"mention": "Getxo", "candidates": [cand("municipality", "48044", ["48"])], "confidence": 0.9}])
    ids, m = linker.link(good)
    rows = pipeline.record_rows(good, ids, m, linker.clip_for(good, ids), g, set())
    assert rows[0]["status"] == "confirmed" and rows[0]["unit_id"] == "48"
    weak = good.model_copy(update={"evidence_id": "f", "source_url": "https://blog.example/x",
                                   "places": [{**good.places[0], "confidence": 0.6}]})
    rows2 = pipeline.record_rows(weak, ids, m, linker.clip_for(weak, ids), g, set())
    assert rows2[0]["status"] == "probable" and "low_confidence" in rows2[0]["reasons"]
    no = good.model_copy(update={"evidence_id": "h", "places": [{**good.places[0], "ambiguous": ["not_actual"]}]})
    assert pipeline.record_rows(no, ids, m, linker.clip_for(no, ids), g, set())[0]["status"] == "rejected"

    agg = pipeline.aggregate(pd.DataFrame(rows + rows2), ["elcorreo.com"]).iloc[0]
    assert agg["status"] == "confirmed" and agg["n_sources"] == 1      # only confirmed rows are counted
    assert agg["has_reference_press"] and agg["ine_municipality"] == "48044"


def test_reference_domains():
    assert pipeline.is_reference("m.elpais.com", ["elpais.com"])
    assert not pipeline.is_reference("noelpais.com", ["elpais.com"])
    assert pipeline.domain("https://www.heraldo.es/x") == "heraldo.es"
