"""Phase 7: ranking. Overrides first, then per unit the confirmed clip with the most views.

Inputs:  data/interim/clip_locations.csv, location_evidence.csv, youtube_videos.csv, units.gpkg,
         data/manual/overrides.csv  (unit_id, clip_id, action = accept | reject | force, note)
Outputs (data/processed/):
  winners.csv             one row per unit (61); no winner → "Sin datos"
  candidates_by_unit.csv  top 5 confirmed clips with a video per unit
  clips_by_unit.csv       every (clip, unit) pair that is not rejected, confirmed and probable
  clips_by_place.csv      every (clip, place) pair that is not rejected, for town markers
Marks: † partial or shared area; ‡ fewer than 2 independent domains or no reference press.
Only raw viewCount values are used (selection of the maximum; no sums or averages).
"""
from __future__ import annotations

import pandas as pd

from vmap import config, units as un

PROCESSED = config.PROCESSED
OVERRIDES = config.MANUAL / "overrides.csv"
OVERRIDE_COLUMNS = ["unit_id", "clip_id", "action", "note"]
VIDEO_COLS = ["clip_id", "youtube_id", "youtube_url", "title", "channel_title", "channel_url", "view_count",
              "views_snapshot_date"]


def load_overrides(path=OVERRIDES) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=OVERRIDE_COLUMNS).to_csv(path, index=False)
    ov = pd.read_csv(path, dtype=str).fillna("")
    bad = ov[~ov["action"].isin(["accept", "reject", "force"])]
    if len(bad):
        raise ValueError(f"{path}: unknown action(s) {bad['action'].tolist()} (use accept | reject | force)")
    return ov


def apply_overrides(locs: pd.DataFrame, ov: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """accept → confirmed, reject → rejected, force → confirmed and wins its unit."""
    locs = locs.copy()
    locs["forced"] = False
    locs["override_note"] = ""
    warnings = []
    for o in ov.itertuples(index=False):
        m = (locs["unit_id"] == o.unit_id) & (locs["clip_id"] == o.clip_id)
        if not m.any():
            warnings.append(f"override {o.action} {o.unit_id}/{o.clip_id}: no such (unit, clip) pair")
            continue
        locs.loc[m, "status"] = "rejected" if o.action == "reject" else "confirmed"
        locs.loc[m, "forced"] = o.action == "force"
        locs.loc[m, "override_note"] = f"{o.action}: {o.note}".strip()
    return locs, warnings


def display_artist(artist, yt_title) -> str:
    """Artist as written in the official video title ('Álvaro Soler - El Mismo Sol' → 'Álvaro Soler') when it
    matches the clip artist; channel-derived names like 'AlvaroSoler' or 'rozalen' read badly on a map."""
    import re

    from vmap.resolve import clips as cl

    if isinstance(yt_title, str) and " - " in yt_title:
        left = re.sub(r"(?i)\s+(?:ft\.?|feat\.?|featuring)\s.*$", "", yt_title.split(" - ")[0])
        left = re.sub(r"\s+", " ", re.sub(r"#\w+", "", left)).strip()
        if left and (cl.artist_match(left, artist) or
                     cl.norm(left).replace(" ", "") == cl.norm(str(artist)).replace(" ", "")):
            return left
    return re.sub(r"\s*\([^)]*\)", "", str(artist)).strip()


def display_song(song) -> str:
    import re

    s = re.sub(r"#\w+", "", str(song))
    s = re.sub(r"(?i)\s+(?:ft\.?|feat\.?|featuring)\s.*$", "", s)
    return re.sub(r"\s+", " ", s).strip(" -")


def marks(row) -> str:
    dagger = bool(row["partial"]) or bool(row["shared_area"])
    ddagger = int(row["n_independent_domains"]) < 2 or not bool(row["has_reference_press"])
    return ("†" if dagger else "") + ("‡" if ddagger else "")


def rank(locs: pd.DataFrame, videos: pd.DataFrame, unit_names: dict[str, str], top_n: int = 5):
    c = locs.merge(videos[VIDEO_COLS], on="clip_id", how="left")
    c["mark"] = c.apply(marks, axis=1)
    c["artist_display"] = [display_artist(a, t) for a, t in zip(c["artist"], c["title"])]
    c["song_display"] = c["song_title"].map(display_song)
    eligible = c[(c["status"] == "confirmed") & c["view_count"].notna()]
    eligible = eligible.sort_values(["forced", "view_count"], ascending=[False, False])
    winners, cands = [], []
    for unit, name in sorted(unit_names.items()):
        g = eligible[eligible["unit_id"] == unit]
        if g.empty:
            winners.append({"unit_id": unit, "unit_name": name, "has_data": False})
            continue
        w = g.iloc[0]
        winners.append({"unit_id": unit, "unit_name": name, "has_data": True, "clip_id": w["clip_id"],
                        "song_title": w["song_title"], "artist": w["artist"], "label_note": "",
                        "song_display": w["song_display"], "artist_display": w["artist_display"],
                        "youtube_id": w["youtube_id"], "youtube_url": w["youtube_url"],
                        "view_count": int(w["view_count"]), "views_snapshot_date": w["views_snapshot_date"],
                        "mark": w["mark"], "location_mention": w["location_mention"],
                        "n_sources": int(w["n_sources"]), "sources": w["domains"], "forced": bool(w["forced"])})
        cands.append(g.head(top_n).assign(rank=range(1, min(top_n, len(g)) + 1)))
    winners = pd.DataFrame(winners)
    cands = pd.concat(cands, ignore_index=True) if cands else pd.DataFrame()
    all_pairs = c[c["status"] != "rejected"].sort_values(["unit_id", "view_count"], ascending=[True, False],
                                                         na_position="last")
    return winners, cands, all_pairs


def places_table(ev_rows: pd.DataFrame, pairs: pd.DataFrame, munis: pd.DataFrame) -> pd.DataFrame:
    """(clip, place) rows for town markers: municipalities and geocoded points; status from the
    (clip, unit) pair after overrides. Province/island-level mentions have no marker."""
    ok = pairs[["clip_id", "unit_id", "status", "mark", "youtube_url", "view_count", "views_snapshot_date",
                "title"]].rename(columns={"title": "youtube_title"})
    rows = ev_rows.dropna(subset=["clip_id", "unit_id", "lat"]).drop(columns=["status"])
    rows = rows.merge(ok, on=["clip_id", "unit_id"])
    names = munis.set_index("ine")["name"].to_dict()
    rows["place_key"] = rows.apply(lambda r: r["ine"] if isinstance(r["ine"], str)
                                   else f"pt:{r['lat']:.3f},{r['lon']:.3f}", axis=1)
    rows["place_name"] = rows.apply(lambda r: names.get(r["ine"], r["location_mention"]) if isinstance(r["ine"], str)
                                    else r["location_mention"], axis=1)
    rank_s = {"confirmed": 0, "probable": 1}
    rows = rows.assign(_s=rows["status"].map(rank_s)).sort_values("_s")
    out = rows.drop_duplicates(["place_key", "clip_id"])[
        ["place_key", "place_name", "unit_id", "lat", "lon", "precision", "clip_id", "artist", "song_title",
         "status", "mark", "youtube_url", "youtube_title", "view_count", "views_snapshot_date"]]
    return out.sort_values(["place_key", "view_count"], ascending=[True, False], na_position="last")


def run(log=print) -> dict:
    locs = pd.read_csv(config.INTERIM / "clip_locations.csv", dtype={"unit_id": str, "ine_municipality": str})
    videos = pd.read_csv(config.INTERIM / "youtube_videos.csv")
    ev_rows = pd.read_csv(config.INTERIM / "location_evidence.csv", dtype={"unit_id": str, "ine": str})
    units = un.load()
    locs, warnings = apply_overrides(locs, load_overrides())
    for w in warnings:
        log(f"  WARNING {w}")
    winners, cands, pairs = rank(locs, videos, dict(zip(units["unit_id"], units["name"])))
    places = places_table(ev_rows, pairs, un.municipalities())
    PROCESSED.mkdir(parents=True, exist_ok=True)
    winners.to_csv(PROCESSED / "winners.csv", index=False)
    cands.to_csv(PROCESSED / "candidates_by_unit.csv", index=False)
    pairs.to_csv(PROCESSED / "clips_by_unit.csv", index=False)
    places.to_csv(PROCESSED / "clips_by_place.csv", index=False)
    snaps = sorted({str(s)[:10] for s in winners["views_snapshot_date"].dropna()})
    return {"units_with_winner": int(winners["has_data"].sum()), "units_sin_datos": int((~winners["has_data"]).sum()),
            "clip_unit_pairs": len(pairs), "confirmed_pairs": int((pairs["status"] == "confirmed").sum()),
            "places": places["place_key"].nunique(), "overrides_applied": len(load_overrides()) - len(warnings),
            "snapshot_dates": snaps}
