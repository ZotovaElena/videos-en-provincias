"""Offline tests for Phase 7 ranking: marks, overrides, winner selection."""
import pandas as pd
import pytest

from vmap import rank


def loc(clip, unit, status="confirmed", partial=False, shared=False, domains=2, ref=True):
    return {"clip_id": clip, "unit_id": unit, "artist": f"A{clip}", "song_title": f"S{clip}", "status": status,
            "partial": partial, "shared_area": shared, "n_independent_domains": domains,
            "has_reference_press": ref, "n_sources": domains, "domains": "x.es", "location_mention": "X"}


def vid(clip, views):
    return {"clip_id": clip, "youtube_id": f"id{clip}", "youtube_url": f"https://www.youtube.com/watch?v=id{clip}",
            "title": "t", "channel_title": "c", "channel_url": "u", "view_count": views,
            "views_snapshot_date": "2026-11-27T10:00:00+00:00"}


def test_marks():
    assert rank.marks(loc("1", "28")) == ""
    assert rank.marks(loc("1", "28", partial=True)) == "†"
    assert rank.marks(loc("1", "28", shared=True)) == "†"
    assert rank.marks(loc("1", "28", domains=1)) == "‡"
    assert rank.marks(loc("1", "28", ref=False)) == "‡"
    assert rank.marks(loc("1", "28", partial=True, domains=1)) == "†‡"


def test_winner_is_max_views_among_confirmed_with_video():
    locs = pd.DataFrame([loc("a", "28"), loc("b", "28"), loc("c", "28", status="probable"), loc("d", "28")])
    videos = pd.DataFrame([vid("a", 100), vid("b", 900), vid("c", 10_000)])          # d: no video
    locs, _ = rank.apply_overrides(locs, pd.DataFrame(columns=rank.OVERRIDE_COLUMNS))
    w, cands, pairs = rank.rank(locs, videos, {"28": "Madrid", "05": "Ávila"})
    w = w.set_index("unit_id")
    assert w.at["28", "clip_id"] == "b" and w.at["28", "youtube_url"].endswith("idb")
    assert not w.at["05", "has_data"]                                     # Sin datos
    assert list(cands["clip_id"]) == ["b", "a"]
    assert set(pairs["status"]) == {"confirmed", "probable"}


def test_overrides():
    locs = pd.DataFrame([loc("a", "28"), loc("b", "28"), loc("c", "28", status="probable")])
    videos = pd.DataFrame([vid("a", 100), vid("b", 900), vid("c", 10_000)])
    ov = pd.DataFrame([{"unit_id": "28", "clip_id": "b", "action": "reject", "note": "wrong city"},
                       {"unit_id": "28", "clip_id": "c", "action": "accept", "note": "checked"},
                       {"unit_id": "99", "clip_id": "z", "action": "accept", "note": ""}])
    out, warnings = rank.apply_overrides(locs, ov)
    assert len(warnings) == 1
    w, _, pairs = rank.rank(out, videos, {"28": "Madrid"})
    assert w.iloc[0]["clip_id"] == "c" and "b" not in set(pairs["clip_id"])
    force = pd.DataFrame([{"unit_id": "28", "clip_id": "a", "action": "force", "note": "editor's choice"}])
    w, _, _ = rank.rank(rank.apply_overrides(locs, force)[0], videos, {"28": "Madrid"})
    assert w.iloc[0]["clip_id"] == "a" and w.iloc[0]["forced"]


def test_bad_override_action(tmp_path):
    p = tmp_path / "o.csv"
    p.write_text("unit_id,clip_id,action,note\n28,a,maybe,\n")
    with pytest.raises(ValueError):
        rank.load_overrides(p)


def test_display_names():
    assert rank.display_artist("AlvaroSoler", "Álvaro Soler - El Mismo Sol (Official Video)") == "Álvaro Soler"
    assert rank.display_artist("rozalen", "Rozalén - La Puerta Violeta") == "Rozalén"
    assert rank.display_artist("Montesanto (banda)", "Fiesta (Video Oficial)") == "Montesanto"
    assert rank.display_artist("Shakira", "Otra Persona - Canción") == "Shakira"
    assert rank.display_song("Dangerous  ft Sam Martin") == "Dangerous"
    assert rank.display_artist("Subtónica", '#subtonica #salvarnos SUBTÓNICA - "Es Hora" (video oficial)') == "SUBTÓNICA"
    assert rank.display_song("Es hora de cambiar #subtonica #eshoradecambiar") == "Es hora de cambiar"


def test_neighbours_never_share_a_colour():
    import geopandas as gpd
    from shapely.geometry import box

    from vmap.plot_map import unit_colors

    # 3x3 grid of touching squares (diagonals touch too) + one isolated island
    k = 100_000                                                         # 100 km cells (metric CRS)
    cells = [box(3e6 + x * k, 2e6 + y * k, 3e6 + (x + 1) * k, 2e6 + (y + 1) * k) for x in range(3) for y in range(3)]
    cells.append(box(4e6, 3e6, 4e6 + k, 3e6 + k))
    units = gpd.GeoDataFrame({"unit_id": [str(i) for i in range(10)], "geometry": cells}, crs=3035).to_crs(4326)
    col = unit_colors(units)
    geoms = dict(zip(units["unit_id"], cells))
    for a in geoms:
        for b in geoms:
            if a < b and geoms[a].intersects(geoms[b]):
                assert col[a] != col[b]


def test_label_text_and_point():
    from shapely.geometry import MultiPolygon, box

    from vmap.plot_map import label_point, label_text

    w = {"has_data": True, "artist_display": "Shakira", "song_display": "Chantaje", "mark": "‡"}
    assert label_text(w) == "Shakira – Chantaje ‡"
    assert label_text({"has_data": False}) == ""
    # unit with an islet: the label goes on the big part, not on the islet or between them
    geom = MultiPolygon([box(0, 0, 10, 10), box(20, 20, 20.5, 20.5)])
    assert box(0, 0, 10, 10).contains(label_point(geom))
