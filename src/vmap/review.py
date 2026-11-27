"""Phase 7: human review page → data/processed/review.html (static) + review.csv.

Per unit: the winner, every confirmed and probable clip with its evidence (snippet + link), YouTube
video (link, channel, raw view count), an IMDb search link for the person to open (never fetched), and
the open doubts (reasons a place is only probable). Plus: places that could not be assigned to a unit,
and clips with no official YouTube video found (with the best unofficial candidate, if any).
Decisions go to data/manual/overrides.csv (unit_id, clip_id, action, note); then `vmap rank` again.
"""
from __future__ import annotations

import html
from urllib.parse import quote_plus

import pandas as pd

from vmap import config

REASON_TEXT = {
    "foreign_homonym": "name also used abroad, no Spanish context",
    "foreign_context": "name also used abroad, passage mentions a foreign country",
    "small_town_unverified": "small town named in Wikipedia without Spanish context",
    "several_municipalities": "several Spanish towns have this name",
    "low_confidence": "weak wording (no explicit 'rodado en X' near the place)",
    "no_clip": "song or artist not identified",
    "link_title_in_text": "video matched only by its title appearing in the text",
    "region_level": "only a region is named",
    "island_province_level": "only the island province is named, not the island",
    "near_boundary": "point within 100 m of a unit boundary",
    "geocode_unchecked": "geocoded without context",
    "geocode_low_importance": "geocoded to a minor feature",
    "geocode_outside_context": "geocoded outside the province the text points to",
    "not_geocoded": "place not found",
    "not_actual": "text says it was only planned",
}
CSS = """
body{font:14px/1.45 system-ui,sans-serif;margin:0;color:#222;background:#fafafa}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:22px} h2{font-size:18px;margin-top:32px;border-bottom:2px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;width:100%;background:#fff} td,th{border:1px solid #e3e3e3;padding:4px 6px;
vertical-align:top;text-align:left} th{background:#f0f0f0}
.conf{color:#1b6e2a;font-weight:600}.prob{color:#9a6a00}.nodata{color:#999;font-style:italic}
.win{background:#eef8ee} .ev{font-size:12px;color:#444;margin:2px 0} .ev a{color:#2457a6}
code{font-size:11px;background:#f3f3f3;padding:1px 3px;user-select:all} .reasons{font-size:12px;color:#9a6a00}
.small{font-size:12px;color:#666}
"""


def esc(x) -> str:
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else html.escape(str(x))


def views(x) -> str:
    return "" if x is None or pd.isna(x) else f"{int(x):,}".replace(",", " ")


def link(url, text) -> str:
    return f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(text)}</a>' if isinstance(url, str) else esc(text)


def _evidence_html(ids: str, ev_by_id: pd.DataFrame, unit: str, limit: int = 4) -> str:
    out = []
    for eid in str(ids).split()[:limit]:
        if eid not in ev_by_id.index:
            continue
        rows = ev_by_id.loc[[eid]]
        r = rows.iloc[0]
        out.append(f'<div class="ev">{link(r["source_url"], r["domain"])}: “{esc(r["evidence_text"][:300])}…”</div>')
    return "".join(out)


def build(winners, pairs, ev_rows, not_found) -> tuple[str, pd.DataFrame]:
    ev_by_id = ev_rows.drop_duplicates("evidence_id").set_index("evidence_id")
    parts = [f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Revisión videoclips</title>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{CSS}</style></head><body><main>",
             "<h1>Revisión: videoclip más visto rodado en cada provincia e isla</h1>",
             "<p class='small'>Decisions go to <code>data/manual/overrides.csv</code> "
             "(<code>unit_id,clip_id,action,note</code>; action = accept | reject | force), then run "
             "<code>vmap rank</code> and <code>vmap review</code>. IMDb links are searches for you to open; "
             "they are never fetched by the pipeline.</p>"]
    # summary
    parts.append("<table><tr><th>Unit</th><th>Winner</th><th>Views</th><th>Marks</th><th>Confirmed</th>"
                 "<th>Probable</th></tr>")
    flat = []
    for w in winners.itertuples(index=False):
        g = pairs[pairs["unit_id"] == w.unit_id]
        nc, np_ = int((g["status"] == "confirmed").sum()), int((g["status"] == "probable").sum())
        win = f"{esc(w.song_title)} — {esc(w.artist)}" if w.has_data else "<span class='nodata'>Sin datos</span>"
        parts.append(f"<tr><td><a href='#u{esc(w.unit_id)}'>{esc(w.unit_id)} {esc(w.unit_name)}</a></td><td>{win}</td>"
                     f"<td>{views(w.view_count) if w.has_data else ''}</td><td>{esc(w.mark) if w.has_data else ''}</td>"
                     f"<td>{nc}</td><td>{np_}</td></tr>")
    parts.append("</table>")
    # per unit
    for w in winners.itertuples(index=False):
        g = pairs[pairs["unit_id"] == w.unit_id]
        parts.append(f"<h2 id='u{esc(w.unit_id)}'>{esc(w.unit_id)} · {esc(w.unit_name)}</h2>")
        if g.empty:
            parts.append("<p class='nodata'>No clips with a place in this unit.</p>")
            continue
        parts.append("<table><tr><th>#</th><th>Clip</th><th>Video</th><th>Views</th><th>Status</th>"
                     "<th>Evidence</th></tr>")
        for i, r in enumerate(g.itertuples(index=False), 1):
            is_win = w.has_data and r.clip_id == w.clip_id
            cls = "conf" if r.status == "confirmed" else "prob"
            reasons = "".join(f"<div>· {esc(REASON_TEXT.get(x, x))}</div>" for x in str(r.reasons or "").split()
                              if r.status != "confirmed")
            imdb = f"https://www.imdb.com/find/?q={quote_plus(f'{r.artist} {r.song_title}')}"
            video = (f"{link(r.youtube_url, r.title)}<div class='small'>{link(r.channel_url, r.channel_title)}</div>"
                     if isinstance(r.youtube_url, str) else "<span class='nodata'>no official video found</span>")
            parts.append(
                f"<tr class='{'win' if is_win else ''}'><td>{i}{' 🏆' if is_win else ''}</td>"
                f"<td><b>{esc(r.song_title)}</b><br>{esc(r.artist)} {esc(r.mark)}<br>"
                f"<span class='small'>{esc(r.location_mention)}</span><br><code>{esc(r.clip_id)}</code><br>"
                f"<span class='small'>{link(imdb, 'IMDb search')}</span></td>"
                f"<td>{video}</td><td>{views(r.view_count)}</td>"
                f"<td class='{cls}'>{esc(r.status)}{esc(' · ' + r.override_note) if r.override_note else ''}"
                f"<div class='reasons'>{reasons}</div></td>"
                f"<td>{_evidence_html(r.evidence_ids, ev_by_id, w.unit_id)}</td></tr>")
            flat.append({"unit_id": w.unit_id, "unit_name": w.unit_name, "rank": i, "winner": is_win,
                         "status": r.status, "clip_id": r.clip_id, "artist": r.artist, "song_title": r.song_title,
                         "location_mention": r.location_mention, "youtube_url": r.youtube_url,
                         "view_count": r.view_count, "mark": r.mark, "reasons": r.reasons,
                         "evidence_ids": r.evidence_ids})
        parts.append("</table>")
    # places without unit
    loose = ev_rows[ev_rows["unit_id"].isna() & (ev_rows["status"] != "rejected") & ev_rows["clip_id"].notna()]
    parts.append(f"<h2>Places not assigned to a unit ({len(loose)})</h2><table><tr><th>Clip</th><th>Place</th>"
                 "<th>Why</th><th>Evidence</th></tr>")
    for r in loose.itertuples(index=False):
        why = ", ".join(REASON_TEXT.get(x, x) for x in str(r.reasons).split() if x in REASON_TEXT)
        parts.append(f"<tr><td><b>{esc(r.song_title)}</b><br>{esc(r.artist)}</td><td>{esc(r.location_mention)}</td>"
                     f"<td class='reasons'>{esc(why)}</td><td><div class='ev'>{link(r.source_url, r.domain)}: "
                     f"“{esc(str(r.evidence_text)[:250])}…”</div></td></tr>")
    parts.append("</table>")
    # no video
    parts.append(f"<h2>Clips without an official YouTube video ({len(not_found)})</h2>"
                 "<p class='small'>Not searched yet = the daily search budget ran out; run <code>vmap youtube</code> "
                 "again tomorrow. A candidate is an upload from a channel that is not the artist's.</p>"
                 "<table><tr><th>Clip</th><th>Searched</th><th>Unofficial candidate</th></tr>")
    for r in not_found.itertuples(index=False):
        cand = (f"{link(r.candidate_url, r.candidate_title)}<div class='small'>{esc(r.candidate_channel)}</div>"
                if isinstance(getattr(r, "candidate_url", None), str) else "")
        parts.append(f"<tr><td><b>{esc(r.song_title)}</b><br>{esc(r.artist)}<br><code>{esc(r.clip_id)}</code></td>"
                     f"<td>{'yes' if r.searched else 'not yet'}</td><td>{cand}</td></tr>")
    parts.append("</table></main></body></html>")
    return "".join(parts), pd.DataFrame(flat)


def run() -> dict:
    p = config.PROCESSED
    winners = pd.read_csv(p / "winners.csv", dtype={"unit_id": str})
    pairs = pd.read_csv(p / "clips_by_unit.csv", dtype={"unit_id": str}).fillna({"override_note": ""})
    ev_rows = pd.read_csv(config.INTERIM / "location_evidence.csv", dtype={"unit_id": str, "ine": str})
    nf_path = config.INTERIM / "youtube_not_found.csv"
    not_found = pd.read_csv(nf_path) if nf_path.exists() and nf_path.stat().st_size > 1 else pd.DataFrame(
        columns=["clip_id", "artist", "song_title", "searched", "candidate_url", "candidate_title", "candidate_channel"])
    for col in ("candidate_url", "candidate_title", "candidate_channel"):
        if col not in not_found:
            not_found[col] = None
    page, flat = build(winners, pairs, ev_rows, not_found)
    (p / "review.html").write_text(page, encoding="utf-8")
    flat.to_csv(p / "review.csv", index=False)
    return {"units": len(winners), "rows": len(flat), "page": str((p / "review.html").relative_to(config.ROOT))}
