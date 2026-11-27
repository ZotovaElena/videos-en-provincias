"""Wikidata: music items with a filming location (P915) in Spain → evidence records.

One SPARQL query per P31 class (keeps each query small, avoids timeouts). Rows are grouped per
(item, location). The location must be in Spain directly (P17) or through its admin chain (P131*/P17).
Units are assigned later from coordinates or the location's admin chain (geo resolution stage).
"""
from __future__ import annotations

import re
from collections import defaultdict

from vmap import config
from vmap.extract.schema import Evidence, evidence_id
from vmap.http import Client, retrieved_at

QUERY = """
SELECT ?item ?itemLabel ?loc ?locLabel ?coord ?performerLabel ?yt WHERE {
  ?item wdt:P31 wd:%(cls)s ;
        wdt:P915 ?loc .
  ?loc wdt:P131*/wdt:P17 wd:Q29 .
  OPTIONAL { ?loc wdt:P625 ?coord }
  OPTIONAL { ?item wdt:P175 ?performer }
  OPTIONAL { ?item wdt:P1651 ?yt }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "es,mul,en". }
}
"""

_POINT = re.compile(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)")


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _point(wkt: str | None) -> tuple[float, float] | None:
    m = _POINT.search(wkt or "")
    return (float(m.group(2)), float(m.group(1))) if m else None  # WKT is lon lat


def rows_to_evidence(rows: list[dict], fetched_at: str, class_label: str) -> list[Evidence]:
    """Group SPARQL bindings into one Evidence per (item, location)."""
    items: dict[str, dict] = {}
    for b in rows:
        item = _qid(b["item"]["value"])
        it = items.setdefault(item, {"label": b.get("itemLabel", {}).get("value"),
                                     "performers": set(), "yt": set(), "locs": defaultdict(dict)})
        if "performerLabel" in b:
            it["performers"].add(b["performerLabel"]["value"])
        if "yt" in b:
            it["yt"].add(b["yt"]["value"])
        loc = it["locs"][_qid(b["loc"]["value"])]
        loc["label"] = b.get("locLabel", {}).get("value")
        loc.setdefault("coords", set())
        if (pt := _point(b.get("coord", {}).get("value"))):
            loc["coords"].add(pt)

    out = []
    for item, it in items.items():
        n_locs = len(it["locs"])
        for loc_qid, loc in it["locs"].items():
            loc_label = loc["label"] or loc_qid
            text = (f"{it['label'] or item} ({item}) [{class_label}] — lugar de rodaje (P915): "
                    f"{loc_label} ({loc_qid})")
            url = f"https://www.wikidata.org/wiki/{item}"
            out.append(Evidence(
                evidence_id=evidence_id(url, text),
                source_type="wikidata",
                source_url=url,
                source_name="Wikidata",
                retrieved_at=fetched_at,
                song_title=it["label"] if it["label"] != item else None,
                artist=" & ".join(sorted(it["performers"])) or None,
                location_mentions=[loc_label],
                partial=n_locs > 1,
                evidence_text=text,
                extractor="wikidata:P915",
                confidence=1.0,
                wikidata_qid=item,
                location_qids=[loc_qid],
                coords=sorted(loc["coords"]),
                youtube_ids=sorted(it["yt"]),
            ))
    return out


def run(client: Client | None = None, refresh: bool = False) -> list[Evidence]:
    cfg = config.search()["wikidata"]
    client = client or Client(stage="wikidata")
    records: list[Evidence] = []
    for cls, label in cfg["classes"].items():
        r = client.get(cfg["endpoint"], {"query": QUERY % {"cls": cls}, "format": "json"},
                       headers={"Accept": "application/sparql-results+json"}, refresh=refresh)
        records += rows_to_evidence(r.json()["results"]["bindings"], retrieved_at(r), label)
    return records
