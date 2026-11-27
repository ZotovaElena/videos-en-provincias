"""Offline tests for Phase 6 candidate checks."""
from vmap import config
from vmap.youtube import ids

S = {**config.settings()["youtube"]}


def row(**kw):
    base = {"youtube_id": "abcdefghijk", "title": "Naiara - Kenbo (Videoclip Oficial)", "channel_id": "UC1",
            "channel_title": "Naiara", "published_at": "2024-01-01", "category_id": "10", "duration_s": 200,
            "view_count": 50_000, "made_for_kids": False, "views_snapshot_date": "2026-11-27"}
    return {**base, **kw}


def test_ids_from_urls():
    urls = ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?feature=x&v=AAAAAAAAAAA", "https://www.youtube.com/shorts/BBBBBBBBBBB",
            "https://www.youtube.com/@naiara", "https://m.youtube.com/watch?v=CCCCCCCCCCC&t=3"]
    assert ids.ids_from_urls(urls) == ["dQw4w9WgXcQ", "AAAAAAAAAAA", "CCCCCCCCCCC"]
    assert ids.watch_url("dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_check_accepts_small_official_video():
    ok, chan, _ = ids.check(row(), "Naiara", "Kenbo", S)
    assert ok and chan                                # no 1 M minimum here


def test_check_rejections():
    assert ids.check(row(title="Naiara - Kenbo (Lyric Video)"), "Naiara", "Kenbo", S)[2] == "title_not_music_video"
    assert ids.check(row(channel_title="Naiara - Topic"), "Naiara", "Kenbo", S)[2] == "topic_channel"
    assert ids.check(row(title="Naiara - Otra canción"), "Naiara", "Kenbo", S)[2] == "song_not_in_title"
    assert ids.check(row(title="Kenbo (cover)", channel_title="Fan123"), "Naiara", "Kenbo", S)[0] is False
    assert ids.check(row(title="Kenbo", channel_title="Fan123"), "Naiara", "Kenbo", S)[2] == "not_artist_channel"
    # fan or TV channels with the artist in the title are not official
    assert ids.check(row(title="THALIA - SUDOR", channel_title="THALIASCHILE"), "Thalía", "Sudor", S)[0] is False
    assert ids.check(row(title="Naiara - Kenbo (Eurovision)", channel_title="ESC Stuff"), "Naiara", "Kenbo", S)[0] is False
    # guest artist is not the song
    assert ids.check(row(title="Romeo Santos - La Diabla/Mi Santa (Official Video) ft. Tomatito",
                         channel_title="RomeoSantosVEVO"), "Romeo Santos", "Tomatito", S)[2] == "song_not_in_title"


def test_channel_rules():
    assert ids.channel_is_artist("Kelis", "KelisVEVO") and ids.channel_is_artist("Goldfrapp", "GoldfrappTV")
    assert ids.channel_is_artist("Romeo Santos", "RomeoSantosVEVO")
    assert not ids.channel_is_artist("Thalía", "THALIASCHILE")
    assert ids.check(row(title="Gisela - Kenbo"), "Gisela (cantante)", "Kenbo", S)[0] is False   # channel Naiara
    assert ids.check(row(title="Gisela - Kenbo", channel_title="Gisela"), "Gisela (cantante)", "Kenbo", S)[0]
    # collaborator's channel
    ok, chan, _ = ids.check(row(title="ROSALÍA & Travis Scott - TKN (Official Video)", channel_title="ROSALÍA"),
                            "Travis Scott", "TKN", S)
    assert ok and not chan


def test_choose_prefers_artist_channel_then_views():
    label = row(youtube_id="L" * 11, title="Naiara - Kenbo", channel_title="Sony Music", view_count=900_000)
    own = row(youtube_id="O" * 11, channel_title="NaiaraVEVO", view_count=100_000)
    best, checked = ids.choose([label, own], "Naiara", "Kenbo", S)
    assert best["youtube_id"] == "O" * 11 and len(checked) == 2


def test_slash_live_title_excluded():
    assert ids.check(row(title='SCHILLER: "Kenbo" // Live // with Unheilig', channel_title="Naiara"),
                     "Naiara", "Kenbo", S)[2] == "title_live"


def test_live_performance_excluded():
    assert ids.check(row(title="Naiara - Kenbo (Official Live Performance Video)"), "Naiara", "Kenbo", S)[2] == "title_live"
