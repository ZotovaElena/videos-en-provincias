"""Offline tests for the top-down seed stage, YouTube budget, clip normalisation and manual import."""
import json

import pytest

from vmap import config
from vmap.extract.schema import Evidence
from vmap.resolve import clips
from vmap.sources import manual, seed
from vmap.youtube import api

S = config.settings()["youtube"]


def video(**kw):
    base = {"youtube_id": "abcdefghijk", "title": "Artista - Canción (Official Video)",
            "channel_id": "UC1", "channel_title": "ArtistaVEVO", "category_id": "10",
            "duration_s": 210, "view_count": 100_000_000, "artist": "Artista",
            "song_title_guess": "Canción"}
    return {**base, **kw}


@pytest.mark.parametrize("title,reason", [
    ("Artista - Canción (Official Video)", ""),
    ("Artista - Canción (Lyric Video)", "title_not_music_video"),
    ("Artista - Canción (Letra)", "title_not_music_video"),
    ("Artista - Canción (Official Audio)", "title_not_music_video"),
    ("Artista - Canción (Live at Wembley)", "title_live"),
    ("Artista - Canción (En Vivo)", "title_live"),
    ("Artista - Live Your Life (Official Video)", ""),
    ("Artista - Canción | Slowed + Reverb", "title_not_music_video"),
    ("Artista - Canción (Karaoke Version)", "title_not_music_video"),
])
def test_youtube_title_filter(title, reason):
    assert seed.exclusion_reason(video(title=title), S) == reason


def test_other_filters():
    assert seed.exclusion_reason(video(channel_title="Artista - Topic"), S) == "topic_channel"
    assert seed.exclusion_reason(video(view_count=1000), S) == "below_min_views"
    assert seed.exclusion_reason(video(duration_s=30), S) == "duration"
    assert seed.exclusion_reason(video(category_id="22"), S) == "category_22"
    assert seed.exclusion_reason(video(made_for_kids=True), S) == "made_for_kids"


def test_iso_duration():
    assert seed.iso_duration_s("PT3M45S") == 225
    assert seed.iso_duration_s("PT1H2M") == 3720
    assert seed.iso_duration_s("P0D") == 0
    assert seed.iso_duration_s(None) is None


@pytest.mark.parametrize("title,artist,song", [
    ("Shakira - Chantaje (Official Video) ft. Maluma", "Shakira", "Chantaje"),
    ("Evanescence - My Immortal (Official HD Music Video)", "Evanescence", "My Immortal"),
    ("Despacito - Luis Fonsi ft. Daddy Yankee", "Luis Fonsi", "Despacito"),
    ("Bailando con lobos (Videoclip Oficial)", "Cabaret Pop", "Bailando con lobos"),
    ("QUEVEDO || BZRP Music Sessions #52", "Quevedo", "BZRP Music Sessions #52"),
])
def test_song_from_youtube_title(title, artist, song):
    assert clips.song_from_youtube_title(title, artist) == song


def test_normalisation():
    assert clips.norm_title("Canción (Videoclip Oficial)") == "cancion"
    assert clips.norm_title("Chantaje (feat. Maluma)") == "chantaje"
    assert clips.norm_artist("Shakira y Maluma") == clips.norm_artist("Maluma & Shakira")
    assert clips.artist_match("Shakira & Maluma", "Shakira")
    assert not clips.artist_match("Quevedo", "Shakira")
    assert clips.norm(float("nan")) == "" and clips.norm_title(None) == ""
    assert not clips.artist_match(float("nan"), "Shakira")


def test_evidence_index_by_title_artist_and_youtube_id():
    e1 = Evidence(evidence_id="1", source_type="wikipedia", source_url="u", source_name="w",
                  retrieved_at="t", evidence_text="El videoclip fue rodado en Barcelona.",
                  song_title="Canción", artist="Artista & Otro")
    e2 = Evidence(evidence_id="2", source_type="wikidata", source_url="u2", source_name="wd",
                  retrieved_at="t", evidence_text="x", youtube_ids=["abcdefghijk"])
    e3 = Evidence(evidence_id="3", source_type="wikipedia", source_url="u3", source_name="w",
                  retrieved_at="t", evidence_text="y", song_title="Canción", artist="Otra Persona")
    idx = seed.EvidenceIndex([e1, e2, e3])
    found = {e.evidence_id for e in idx.find(video())}
    assert found == {"1", "2"}                     # e3: same title, different artist
    assert seed._status(idx.find(video())) == "wikidata_location"
    assert seed._status([e1]) == "filming_info"
    assert seed._status([]) == "no_article"


def test_parse_most_viewed_list():
    wt = """{| class="wikitable"
|-
! Video name !! Uploader
|-
| [[Despacito]]<ref>{{cite AV media|url=https://www.youtube.com/watch?v=kJQP7kiw5Fk}}</ref> || [[Luis Fonsi|LuisFonsiVEVO]] || 8.9
|-
| [[Baby Shark|Baby Shark Dance]] || [[Pinkfong]] || 17
|}"""
    titles, ids = seed.parse_most_viewed_list(wt)
    assert titles == ["Despacito", "Luis Fonsi", "Baby Shark", "Pinkfong"]
    assert ids == ["kJQP7kiw5Fk"]


def test_budget_blocks_over_limit_and_resets_daily(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_LOG", tmp_path / "log.jsonl")
    b = api.Budget(path=tmp_path / "q.json", units=2, searches=1)
    b.spend("units")
    b.spend("units")
    with pytest.raises(api.QuotaExceeded):
        b.spend("units")
    b.spend("search")
    state = json.loads((tmp_path / "q.json").read_text())
    state["date"] = "2000-01-01"
    (tmp_path / "q.json").write_text(json.dumps(state))
    b.spend("units")                                   # new day → counter reset
    assert b.used()["units"] == 1


def test_manual_import_validates(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text(
        "song_title,artist,youtube_id,location,source_url,snippet,partial,retrieved_at,note\n"
        'Canción,Artista,,Bilbao,https://ex.org/a,"El videoclip se rodó en Bilbao.",,,\n'
        'Canción,Artista,,Getxo,https://ex.org/a,"El videoclip se rodó en Bilbao.",,,\n'
        'Canción,Artista,,Bilbao; Getxo,https://ex.org/b,"Rodado en Bilbao y   Getxo.",,,\n'
        "Canción,Artista,,Bilbao,,Rodado en Bilbao,,,\n", encoding="utf-8")
    recs, errors = manual.load_rows(p)
    assert len(recs) == 2 and len(errors) == 2
    assert recs[1].location_mentions == ["Bilbao", "Getxo"] and recs[1].partial is True
    assert recs[0].source_name == "ex.org"


def test_plan_channels_order_and_skips():
    A = seed.Artist
    ca = {"UCa": A("Hand", "yaml"), "UCb": A("Big", "wikidata_expansion"),
          "UCc": A("Small", "wikidata_expansion"), "UCd": A("Huge", "yaml"), "UCe": A("T", "yaml")}
    chans = {"UCb": {"title": "Big", "uploads": "UUb", "video_count": 10, "subscribers": 5_000_000},
             "UCc": {"title": "Small", "uploads": "UUc", "video_count": 10, "subscribers": 1000},
             "UCa": {"title": "Hand", "uploads": "UUa", "video_count": 10, "subscribers": 10},
             "UCd": {"title": "Huge", "uploads": "UUd", "video_count": 99999, "subscribers": 1},
             "UCe": {"title": "T - Topic", "uploads": "UUe", "video_count": 5, "subscribers": 1}}
    plan = seed.plan_channels(chans, ca, S, min_subs=100_000)
    order = list(plan)
    assert order.index("UCa") < order.index("UCb")          # hand-picked before expansion
    assert plan["UCb"]["status"] == "to_page"
    assert plan["UCc"]["status"] == "skipped_few_subscribers"
    assert plan["UCd"]["status"] == "skipped_too_many_uploads"
    assert plan["UCe"]["status"] == "skipped_topic_or_no_uploads"


class FakeYT:
    def __init__(self, items):
        self.items, self.calls = items, []

    def call(self, endpoint, params):
        self.calls.append((endpoint, params))
        return {"items": self.items}, None


def _item(ch, title):
    return {"snippet": {"channelId": ch, "channelTitle": title}}


def test_search_channel_prefers_matching_non_topic():
    yt = FakeYT([_item("UC1", "Vetusta Morla - Topic"), _item("UC2", "Fans de Vetusta"),
                 _item("UC3", "Vetusta Morla")])
    assert seed.search_channel(yt, "Vetusta Morla") == "UC3"
    assert yt.calls[0][0] == "search" and yt.calls[0][1]["type"] == "channel"
    assert seed.search_channel(FakeYT([_item("UC9", "Otro Canal")]), "Vetusta Morla") is None


def test_fill_missing_channels_only_for_hand_listed():
    A = seed.Artist
    arts = [A("Vega", "yaml"), A("Topic only", "yaml", channel_ids=["UCt"]),
            A("Fine", "yaml", channel_ids=["UCf"]), A("Auto", "wikidata_expansion")]
    chans = {"UCt": {"title": "Topic only - Topic"}, "UCf": {"title": "Fine"}}
    yt = FakeYT([_item("UCv", "Vega"), _item("UCx", "Topic only")])
    new = seed.fill_missing_channels(yt, arts, chans, log=lambda *_: None)
    assert len(yt.calls) == 2                      # Vega and the Topic-only artist; not Fine, not Auto
    assert arts[0].channel_ids == ["UCv"] and arts[0].status == "channel_from_youtube_search"
    assert "UCx" in new
