"""Offline tests for hand-added videos and the local editor."""
import json
import threading
import urllib.request

import pandas as pd
import pytest

from vmap import editor, rank
from vmap.sources import manual


def test_youtube_id_parsing():
    assert manual.youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=3") == "dQw4w9WgXcQ"
    assert manual.youtube_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert manual.youtube_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert manual.youtube_id("not a link") is None


def test_manual_observation_and_source_rows():
    obs, err = manual.row_to_evidence({"artist": "Ultraligera", "song_title": "Europa", "location": "Madrid",
                                       "youtube_id": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                       "note": "Se ven las calles de Madrid."})
    assert err is None and obs.extractor == "manual:observation"
    assert obs.source_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ" and obs.youtube_ids == ["dQw4w9WgXcQ"]
    src, err = manual.row_to_evidence({"artist": "Huntza", "song_title": "Aldapan gora", "location": "Donostia",
                                       "source_url": "https://ex.org/a",
                                       "snippet": "El videoclip se rodó en Donostia."})
    assert err is None and src.extractor == "manual"
    _, err = manual.row_to_evidence({"artist": "X", "song_title": "Y", "location": "Bilbao",
                                     "source_url": "https://ex.org/a", "snippet": "Rodado en Getxo."})
    assert "literally" in err
    _, err = manual.row_to_evidence({"artist": "X", "song_title": "Y", "location": "Bilbao"})
    assert "observation" in err or "snippet" in err


@pytest.fixture
def tmp_manual(tmp_path, monkeypatch):
    monkeypatch.setattr(rank, "OVERRIDES", tmp_path / "overrides.csv")
    monkeypatch.setattr(editor, "VIDEO_LINKS", tmp_path / "video_links.csv")
    monkeypatch.setattr(editor, "MANUAL_CSV", tmp_path / "manual.csv")
    return tmp_path


def test_save_override_set_toggle_and_one_force(tmp_manual):
    editor.save_override("28", "a", "accept", "ok")
    editor.save_override("28", "a", "reject", "wrong")           # replaces the previous decision
    editor.save_override("28", "b", "force", "")
    editor.save_override("28", "c", "force", "")                 # only one forced winner per unit
    ov = pd.read_csv(rank.OVERRIDES, dtype=str).fillna("")
    assert ov.to_dict("records") == [{"unit_id": "28", "clip_id": "a", "action": "reject", "note": "wrong"},
                                     {"unit_id": "28", "clip_id": "c", "action": "force", "note": ""}]
    editor.save_override("28", "a", "", "")                      # undo
    assert len(pd.read_csv(rank.OVERRIDES)) == 1
    with pytest.raises(ValueError):
        editor.save_override("28", "a", "maybe", "")


def test_add_and_delete_manual(tmp_manual):
    assert editor.add_manual({"artist": "X", "song_title": "Y", "location": "Madrid"}) is not None
    assert editor.add_manual({"artist": "Ultraligera", "song_title": "Europa", "location": "Madrid",
                              "youtube_id": "dQw4w9WgXcQ", "note": "Se ven las calles de Madrid."}) is None
    assert len(pd.read_csv(editor.MANUAL_CSV)) == 1
    editor.delete_manual(0)
    assert len(pd.read_csv(editor.MANUAL_CSV)) == 0


def test_manual_rows_are_human_accepted():
    from vmap.extract import gazetteer, rules
    from vmap.resolve import link, pipeline
    from tests.test_resolve import FakeGeo, TOP

    places = pd.DataFrame([{"ine": "48044", "province": "48", "population": 79000, "name": "Getxo",
                            "variants": "Getxo", "is_capital": False}])
    admin = pd.DataFrame([{"kind": "province", "code": "48", "region": "16", "name": "Bizkaia"}])
    g = gazetteer.build(places, admin, {})
    ev, _ = manual.row_to_evidence({"artist": "Naiara", "song_title": "Kenbo", "location": "Getxo",
                                    "youtube_id": "https://youtu.be/AAAAAAAAAAA", "note": "Se ve la playa."})
    r = rules.extract(ev, g)
    assert r.places[0]["candidates"][0]["code"] == "48044" and r.places[0]["confidence"] == 1.0
    lk = link.Linker(TOP)
    ids, method = lk.link(r)
    assert method == "manual" and ids == ["AAAAAAAAAAA"]
    rows = pipeline.record_rows(r, ids, method, lk.clip_for(r, ids), FakeGeo(), set())
    assert rows[0]["status"] == "confirmed" and rows[0]["unit_id"] == "48"


def test_server_rejects_foreign_host_and_non_json(tmp_manual):
    httpd = editor.ThreadingHTTPServer(("127.0.0.1", 0), editor.Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        def post(body, ctype="application/json", host=None):
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/override", data=body, method="POST",
                                         headers={"Content-Type": ctype, **({"Host": host} if host else {})})
            try:
                return urllib.request.urlopen(req).status
            except urllib.error.HTTPError as e:
                return e.code
        ok = json.dumps({"unit_id": "28", "clip_id": "a", "action": "accept", "note": ""}).encode()
        assert post(ok, ctype="text/plain") == 403                # simple cross-site form posts are refused
        assert post(ok, host="evil.example") == 403               # DNS-rebinding style hosts are refused
        assert post(ok) == 200
        assert pd.read_csv(rank.OVERRIDES).iloc[0]["action"] == "accept"
    finally:
        httpd.shutdown()


def test_batch_save(tmp_manual):
    n = editor.save_overrides([{"unit_id": "28", "clip_id": "a", "action": "reject", "note": "x"},
                               {"unit_id": "45", "clip_id": "b", "action": "accept", "note": ""}])
    assert n == 2 and len(pd.read_csv(rank.OVERRIDES)) == 2


def test_video_links(tmp_manual, monkeypatch):
    from vmap.youtube import ids
    editor.save_overrides([{"kind": "video", "clip_id": "c1", "youtube_id": "https://youtu.be/dQw4w9WgXcQ"},
                           {"kind": "video", "clip_id": "c2", "youtube_id": "AAAAAAAAAAA"}])
    editor.save_video_link("c2", "")                              # removed again
    monkeypatch.setattr(ids, "VIDEO_LINKS", editor.VIDEO_LINKS)
    assert ids.video_links() == {"c1": "dQw4w9WgXcQ"}
    with pytest.raises(ValueError):
        editor.save_video_link("c3", "https://vimeo.com/123")
