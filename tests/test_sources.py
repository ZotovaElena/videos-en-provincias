"""Offline tests for the Wikidata and Wikipedia sources and the evidence schema."""
from vmap import config
from vmap.extract import schema
from vmap.sources import wikidata as wd
from vmap.sources import wikipedia as wp

CFG = config.search()["wikipedia"]

WIKITEXT = """{{Ficha de sencillo
| nombre = Canción X
| artista = [[Artista Y]]
}}
'''Canción X''' es un sencillo de [[Artista Y]].<ref>{{cita web|url=http://a.b}}</ref>

== Composición ==
La canción fue grabada en Londres en un estudio con varios músicos invitados.

== Videoclip ==
El [[videoclip]] fue dirigido por Z y se rodó en [[Bilbao|la ciudad de Bilbao]] y en {{lang|eu|Getxo}}.
[[Archivo:Foto.jpg|thumb|Rodaje en [[Bilbao]]]]
Enlace: [https://www.youtube.com/watch?v=abcdefghijk vídeo oficial]

=== Recepción ===
El vídeo alcanzó un millón de visitas en su primer día de publicación en internet.

== Lista de canciones ==
# Canción X — 3:30 minutos de duración total del sencillo

== Legado ==
Años después, otro videoclip de la banda volvió a usar escenas rodadas en Sevilla durante la gira.
"""


def test_wikitext_to_text():
    t = wp.wikitext_to_text("El [[videoclip]] se rodó en [[Bilbao|la ciudad de Bilbao]] y "
                            "{{lang|eu|Getxo}}.<ref>x</ref> [[Archivo:F.jpg|thumb|cap [[A]]]]'''ok'''")
    assert t == "El videoclip se rodó en la ciudad de Bilbao y Getxo. ok"


def test_passages_keep_video_sections_and_keyword_paragraphs():
    ps = wp.passages(WIKITEXT, CFG)
    by_heading = {h: (text, yt) for h, text, yt in ps}
    assert "Videoclip" in by_heading
    text, yt = by_heading["Videoclip"]
    assert "se rodó en la ciudad de Bilbao y en Getxo" in text
    assert "Foto.jpg" not in text
    assert yt == ["abcdefghijk"]
    assert "Recepción" in by_heading          # subsection of the video section
    assert "Legado" in by_heading             # keyword paragraph elsewhere
    assert "Composición" not in by_heading    # no video keyword, not a video section
    assert "Lista de canciones" not in by_heading


def test_clean_title():
    assert wp.clean_title("Despacito (canción)") == "Despacito"
    assert wp.clean_title("Hello (Adele song)") == "Hello (Adele song)"  # only generic suffixes
    assert wp.clean_title("Bad Romance (song)") == "Bad Romance"


def test_page_to_evidence_song_page():
    page = {"lang": "es", "title": "Canción X (canción)", "qid": "Q1", "wikitext": WIKITEXT,
            "retrieved_at": "2026-11-26T00:00:00+00:00"}
    info = {"Q1": {"classes": {"Q7366"}, "performer_labels": ["Artista Y"]}}
    recs = wp.page_to_evidence(page, info, CFG)
    assert recs and all(r.song_title == "Canción X" and r.artist == "Artista Y" for r in recs)
    assert all(r.page_kind == "song" and r.extractor == "none" for r in recs)
    assert recs[0].source_url == "https://es.wikipedia.org/wiki/Canción_X_(canción)"


def test_wikidata_rows_grouped_per_item_and_location():
    def row(item, loc, coord=None, perf=None, yt=None):
        b = {"item": {"value": f"http://www.wikidata.org/entity/{item}"}, "itemLabel": {"value": "Song"},
             "loc": {"value": f"http://www.wikidata.org/entity/{loc}"}, "locLabel": {"value": loc + "-name"}}
        if coord:
            b["coord"] = {"value": coord}
        if perf:
            b["performerLabel"] = {"value": perf}
        if yt:
            b["yt"] = {"value": yt}
        return b

    rows = [row("Q1", "Q10", "Point(-2.93 43.26)", "A", "y1"), row("Q1", "Q10", perf="B", yt="y2"),
            row("Q1", "Q11"), row("Q2", "Q12")]
    recs = wd.rows_to_evidence(rows, "2026-11-26T00:00:00+00:00", "music video")
    assert len(recs) == 3
    r1 = next(r for r in recs if r.location_qids == ["Q10"])
    assert r1.coords == [(43.26, -2.93)]            # (lat, lon)
    assert r1.artist == "A & B" and r1.youtube_ids == ["y1", "y2"]
    assert r1.partial is True                       # item has two locations
    assert r1.location_mentions[0] in r1.evidence_text
    assert next(r for r in recs if r.wikidata_qid == "Q2").partial is False


def test_schema_write_load_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(schema, "EVIDENCE_DIR", tmp_path)
    e = schema.Evidence(evidence_id="x", source_type="wikidata", source_url="u", source_name="n",
                        retrieved_at="t", evidence_text="text", coords=[(1.0, 2.0)])
    schema.write("wikidata", [e, e])
    loaded = schema.load("wikidata")
    assert len(loaded) == 1 and loaded[0].coords == [(1.0, 2.0)]
    assert len(schema.load_all()) == 1
