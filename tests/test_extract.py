"""Offline tests for the rule extractor and the gold-set evaluation (small hand-built gazetteer)."""
import pandas as pd
import pytest

from vmap.extract import evaluate, gazetteer, rules
from vmap.extract.schema import Evidence

PLACES = pd.DataFrame([
    {"ine": "48044", "province": "48", "population": 79000, "name": "Getxo", "variants": "Getxo"},
    {"ine": "28079", "province": "28", "population": 3_400_000, "name": "Madrid", "variants": "Madrid"},
    {"ine": "20069", "province": "20", "population": 188_000, "name": "Donostia/San Sebastián",
     "variants": "Donostia|San Sebastián"},
    {"ine": "15030", "province": "15", "population": 250_000, "name": "A Coruña", "variants": "A Coruña|La Coruña"},
    {"ine": "15078", "province": "15", "population": 98_000, "name": "Santiago de Compostela",
     "variants": "Santiago de Compostela"},
    {"ine": "46250", "province": "46", "population": 800_000, "name": "València", "variants": "València|Valencia"},
    {"ine": "06083", "province": "06", "population": 2300, "name": "Medellín", "variants": "Medellín"},
    {"ine": "23006", "province": "23", "population": 5300, "name": "Arjona", "variants": "Arjona"},
    {"ine": "03052", "province": "03", "population": 1300, "name": "Cañada", "variants": "Cañada"},
    {"ine": "30027", "province": "30", "population": 35000, "name": "Moya", "variants": "Moya"},
    {"ine": "16133", "province": "16", "population": 200, "name": "Moya", "variants": "Moya"},
    {"ine": "35024", "province": "35", "population": 24000, "name": "Teguise", "variants": "Teguise"},
    {"ine": "25061", "province": "25", "population": 85, "name": "Cabó", "variants": "Cabó"},
    {"ine": "15023", "province": "15", "population": 7000, "name": "Cee", "variants": "Cee"},
    {"ine": "24089", "province": "24", "population": 120_000, "name": "León", "variants": "León"},
    {"ine": "08101", "province": "08", "population": 280_000, "name": "L'Hospitalet de Llobregat",
     "variants": "L'Hospitalet de Llobregat"},
    {"ine": "05900", "province": "05", "population": 300, "name": "La Torre", "variants": "La Torre"},
    {"ine": "16900", "province": "16", "population": 100, "name": "Uña", "variants": "Uña"},
    {"ine": "12900", "province": "12", "population": 2300, "name": "Vilafranca", "variants": "Vilafranca"},
    {"ine": "11020", "province": "11", "population": 212_000, "name": "Jerez de la Frontera",
     "variants": "Jerez de la Frontera"},
])
ADMIN = pd.DataFrame([
    {"kind": "province", "code": "48", "region": "16", "name": "Bizkaia"},
    {"kind": "province", "code": "28", "region": "13", "name": "Madrid"},
    {"kind": "province", "code": "46", "region": "10", "name": "València/Valencia"},
    {"kind": "province", "code": "15", "region": "12", "name": "A Coruña"},
    {"kind": "region", "code": "12", "region": "12", "name": "Galicia"},
])
CFG = {
    "province_aliases": {"48": ["Vizcaya"]},
    "islands": {"35-GRA": ["La Graciosa"], "35-LAN": ["Lanzarote"], "38-LPA": ["La Palma"]},
    "municipality_aliases": {"15078": ["Santiago"], "11020": ["Jerez"]},
    "landmarks": ["Bardenas Reales"],
    "foreign_homonyms": ["Valencia", "Medellín", "Santiago", "La Palma"],
    "stop_names": ["Real"],
    "spain_cues": ["España", "Spain"],
    "foreign_cues": ["Colombia", "Venezuela", "Chile", "Mexico"],
}


@pytest.fixture(scope="module")
def g():
    return gazetteer.build(PLACES.assign(is_capital=False), ADMIN, CFG)


def ev(text, source_type="press", title=None, url="https://example.com/a", **kw):
    return Evidence(evidence_id="x", source_type=source_type, source_url=url, source_name="t",
                    retrieved_at="2026-01-01", evidence_text=text, page_title=title, **kw)


def places(g, text, **kw):
    r = rules.extract(ev(text, **kw), g, rules.ArtistIndex(["Ricardo Arjona", "Arjona"]))
    return {p["mention"]: p for p in r.places} if r else {}


def test_basic_spanish_sentence(g):
    p = places(g, "La banda ha rodado el videoclip de su nuevo single en la playa de Ereaga, en Getxo.")
    assert p["Getxo"]["candidates"][0]["code"] == "48044"
    assert p["Getxo"]["confidence"] >= 0.8


def test_place_before_trigger_and_english(g):
    assert "Madrid" in places(g, "En Madrid se rodó el videoclip de la canción.")
    assert "Madrid" in places(g, "The music video was shot in Madrid over two days.")


def test_bilingual_names(g):
    assert places(g, "El videoclip se grabó en Donostia.")["Donostia"]["candidates"][0]["code"] == "20069"
    assert places(g, "El videoclip se grabó en San Sebastián.")["San Sebastián"]["candidates"][0]["code"] == "20069"
    assert "La Coruña" in places(g, "El videoclip se grabó en La Coruña, España.")


def test_longest_match_and_santiago(g):
    p = places(g, "El videoclip fue rodado en Santiago de Compostela.")
    assert list(p) == ["Santiago de Compostela"] and not p["Santiago de Compostela"]["ambiguous"]
    p = places(g, "El videoclip fue rodado en Santiago.")
    assert p["Santiago"]["ambiguous"] == ["foreign_homonym"]
    assert "Santiago" not in places(g, "El videoclip fue rodado en Santiago, Chile.")


def test_islands(g):
    p = places(g, "El videoclip se rodó en La Graciosa, frente a Lanzarote.")
    assert p["La Graciosa"]["candidates"][0]["code"] == "35-GRA"
    assert "La Palma" not in places(g, "Grabó el videoclip con la palma de la mano.")


def test_homonym_needs_spanish_context(g):
    assert places(g, "The video was filmed in Valencia.")["Valencia"]["ambiguous"]
    assert not places(g, "The video was filmed in Valencia, Spain.")["Valencia"]["ambiguous"]
    assert not places(g, "The video was filmed in Valencia.", url="https://www.levante-emv.es/x")["Valencia"][
        "ambiguous"]
    assert "Medellín" not in places(g, "El video fue grabado en Medellín, Colombia, en el embalse.")
    p = places(g, "El video fue grabado en Medellín. Es su primer trabajo en Colombia.")
    assert p["Medellín"]["ambiguous"] == ["foreign_context"]


def test_people_and_artists_are_not_places(g):
    assert "Arjona" not in places(g, "El videoclip, que Ricardo Arjona rodó en Madrid, se estrenó ayer.")
    assert "Arjona" not in places(g, "Arjona filmed the video in Madrid.")


def test_enye_is_not_n(g):
    assert "Canadá" not in places(g, "El videoclip fue rodado en Canadá.")


def test_negative_cues_and_record_needs_video_word(g):
    assert not places(g, "La canción fue grabada en Madrid en los estudios Red Led.")
    assert "Madrid" not in places(g, "El videoclip, ambientado en Madrid, se estrenó en marzo.")
    assert "Madrid" not in places(g, "El videoclip se grabó tras su concierto en Madrid.")


def test_duplicate_municipality_name_is_ambiguous(g):
    assert places(g, "El videoclip se rodó en Moya.")["Moya"]["ambiguous"] == ["several_municipalities"]


def test_partial_and_counterfactual(g):
    r = rules.extract(ev("El videoclip se rodó en Getxo y en Madrid."), g)
    assert r.partial
    r = rules.extract(ev("Algunas escenas del videoclip se rodaron en Getxo."), g)
    assert r.partial
    r = rules.extract(ev("El videoclip se rodó en Getxo, Vizcaya."), g)
    assert not r.partial                       # province of the same town is not a second place
    p = places(g, "El videoclip iba a ser grabado en Madrid, pero se canceló.")
    assert "not_actual" in p["Madrid"]["ambiguous"]


def test_page_title_and_song(g):
    r = rules.extract(ev("Galería | La cantante rueda su próximo videoclip «Luz» en la capital.",
                         title="La cantante graba su nuevo videoclip en Getxo"), g)
    assert r.places[0]["field"] == "page_title" and r.song_title == "Luz"
    r = rules.extract(ev("The music video was filmed in Madrid.", source_type="wikipedia",
                         song_title="Up! (Shania Twain song)"), g)
    assert r.song_title == "Up!"


def test_unresolved_only_with_spanish_context(g):
    p = places(g, "El videoclip se rodó en las Bardenas Reales y en Getxo.")
    assert p["Bardenas Reales"]["candidates"][0]["kind"] == "landmark"
    p = places(g, "El videoclip se rodó en Riotinto, en España.")
    assert p["Riotinto"].get("unresolved")
    assert not places(g, "The video was filmed in Bucharest.")


def test_mentions_are_literal(g):
    text = "El videoclip se rodó en La Coruña y en Getxo."
    r = rules.extract(ev(text), g)
    assert all(m in text for m in r.location_mentions)


def test_evaluate_prf(g):
    gold = [{"evidence_id": "a", "gold_places": ["Getxo"], "gold_partial": False},
            {"evidence_id": "b", "gold_places": [], "gold_partial": None},
            {"evidence_id": "c", "gold_places": None}]
    texts = {"a": "El videoclip se rodó en Getxo.", "b": "El videoclip se rodó en Madrid."}
    res = evaluate.evaluate(gold, lambda i: rules.extract(ev(texts[i]), g), g)
    assert res["n"] == 2
    assert res["places"] == {"precision": 0.5, "recall": 1.0, "f1": 0.667, "tp": 1, "fp": 1, "fn": 0}
    assert res["provinces"]["tp"] == 1 and res["partial_accuracy"] == 1.0


def test_excel_round_trip(tmp_path):
    rows = [{"evidence_id": "a", "source_type": "press", "source_url": "u", "page_title": "t", "text": "x",
             "gold_places": ["Getxo", "Madrid"], "gold_partial": True, "gold_song": "Luz", "gold_artist": None},
            {"evidence_id": "b", "source_type": "press", "source_url": "u", "page_title": None, "text": "y",
             "gold_places": [], "gold_partial": None, "gold_song": None, "gold_artist": None},
            {"evidence_id": "c", "source_type": "press", "source_url": "u", "page_title": None, "text": "z",
             "gold_places": None, "gold_partial": None, "gold_song": None, "gold_artist": None}]
    path = evaluate.to_excel(rows, tmp_path / "g.xlsx")
    back = {r["evidence_id"]: r for r in evaluate.from_excel(path)}
    assert back["a"]["gold_places"] == ["Getxo", "Madrid"] and back["a"]["gold_partial"] is True
    assert back["a"]["gold_song"] == "Luz" and back["a"]["gold_artist"] is None
    assert back["b"]["gold_places"] == [] and back["c"]["gold_places"] is None


def test_gold_set_regressions(g):
    assert "Cee" not in places(g, "Fue grabado en el mismo restaurante donde se grabó Cee-Lo Green el video.")
    assert "León" not in places(g, "Dirigido por Juan Ponce de León, el video fue filmado en Argentina.")
    assert "León" in places(g, "El videoclip fue grabado en la Catedral de León, España.")
    assert not places(g, "El videoclip se rodó en las playas del Cabo.")
    p = places(g, "Lanzó «Lo noto», con un videoclip rodado en Hospitalet de Llobregat.")
    assert p["Hospitalet de Llobregat"]["candidates"][0]["code"] == "08101"
    r = rules.extract(ev("El nuevo videoclip, rodado en Jerez de la Frontera, ya está aquí.",
                         title="Soraya estrena su videoclip rodado en Jerez"), g)
    assert r.location_mentions == ["Jerez de la Frontera"] and not r.partial


def test_evaluate_gold_qualifiers(g):
    gold = [{"evidence_id": "a", "gold_places": ["Getxo, Bizkaia"], "gold_partial": None}]
    res = evaluate.evaluate(gold, lambda i: rules.extract(ev("El videoclip se rodó en Getxo, Bizkaia."), g), g)
    assert res["places"]["tp"] == 1 and res["places"]["fp"] == 0 and not res["errors"]


def test_place_noun_before_de_and_video_en(g):
    assert "Madrid" in places(g, "Desde el centro Lourdes de Madrid, principal escenario de la grabación del videoclip.")
    assert "Madrid" in places(g, "Lanzó «Lo noto», con un videoclip en Madrid, y superó el millón de visitas.")


def test_song_and_headline_artist(g):
    r = rules.extract(ev("Se han mostrado muy \"orgullosos\". El videoclip de «Lo que queda de mí» se rodó en Getxo."), g)
    assert r.song_title == "Lo que queda de mí"
    r = rules.extract(ev("El grupo graba entre el viernes y el sábado en Getxo el videoclip del tema ‘Nunca la primera’."), g)
    assert r.song_title == "Nunca la primera"
    assert rules.headline_artist("Naiara graba su nuevo videoclip en Zaragoza") == "Naiara"
    assert rules.headline_artist("La cantante Soraya estrena su nuevo videoclip") == "Soraya"
    assert rules.headline_artist("Galería | M Clan rueda su videoclip en Lleida") == "M Clan"
    assert rules.headline_artist("El videoclip se rodó en Getxo") is None


def test_phase5_false_positive_regressions(g):
    # a place inside a quoted song title or a hashtag is not a filming place
    assert "Madrid" not in places(g, 'The music video for "Adiós Madrid" was shot in Miami Beach.')
    # "Spanish director" is not Spanish context; small towns in Wikipedia need it
    p = places(g, "The video was filmed in Moya. It was directed by Spanish director X.", source_type="wikipedia")
    assert p["Moya"]["ambiguous"]
    r = rules.extract(ev("El vídeo se rodó en Teguise, España.", source_type="wikipedia"), g)
    assert not r.places[0].get("ambiguous")
    r = rules.extract(ev("El vídeo se rodó en Teguise.", source_type="wikipedia"), g)
    assert r.places[0]["ambiguous"] == ["small_town_unverified"]
    r = rules.extract(ev("El vídeo se rodó en Getxo.", source_type="wikipedia"), g)     # > 50k people
    assert not r.places[0].get("ambiguous")
    r = rules.extract(ev("El vídeo se rodó en Teguise."), g)                # press: no such requirement
    assert not r.places[0].get("ambiguous")


def test_phase5_second_round(g):
    assert "la Torre" not in places(g, "El videoclip fue grabado en Madrid, por la Torre del Agua de Besos.")
    assert "Uña" not in places(g, "Alicante, escenario de la grabación del primer videoclip de Uña y Carne.")
    assert "Uña" in places(g, "El videoclip se grabó en Uña, un pequeño pueblo.")
    assert "Vilafranca" not in places(
        g, "El videoclip se rodó en julio coincidiendo con el tradicional Concurs de Castells de Vilafranca.")
    assert "Teguise" not in places(g, "El videoclip de su canción 'Teguise de noche' se rodó en Getxo.")
    # "España" far away (chart positions) does not vouch for a name shared with a foreign city
    text = ("La canción llegó al top 15 en varios países como Argentina y España, y fue doble platino en "
            "México y en otros mercados de la región durante meses. Un video musical adjunto, lanzado con "
            "la canción, fue filmado en Medellín y dirigido por Eric.")
    p = places(g, text, source_type="wikipedia")
    assert p["Medellín"]["ambiguous"]


def test_group_named_after_town(g):
    assert "Madrid" not in places(g, "Grabó el videoclip en Getxo con los Castellers de Madrid.")
    assert "Madrid" in places(g, "Grabó el videoclip en el centro Lourdes de Madrid.")


def test_song_from_sentence_with_place(g):
    text = ("En febrero publicaron el sencillo «Fiesta en el desierto», que fue número uno. "
            "«Santo es el que vive (Acústica)», grabada en las calles de Getxo, fue su videoclip más visto.")
    r = rules.extract(ev(text, source_type="wikipedia", artist="Montesanto (banda)"), g)
    assert r.song_title == "Santo es el que vive (Acústica)" and r.artist == "Montesanto"
