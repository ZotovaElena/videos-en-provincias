"""Small Wikidata API helpers (wbgetentities / wbsearchentities), batched and cached."""
from __future__ import annotations

from vmap.http import Client

API = "https://www.wikidata.org/w/api.php"

ARTIST_CLASSES = {"Q5", "Q215380", "Q5741069", "Q2088357", "Q9212979", "Q19849001", "Q281643",
                  "Q216337", "Q107357104"}


def entities(client: Client, ids, props: str = "claims|labels", langs: str = "es|mul|en") -> dict[str, dict]:
    ids = sorted({i for i in ids if i})
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 50):
        d = client.get_json(API, {"action": "wbgetentities", "ids": "|".join(ids[i:i + 50]),
                                  "props": props, "languages": langs, "format": "json"})
        out.update({k: v for k, v in d.get("entities", {}).items() if "missing" not in v})
    return out


def claim_values(ent: dict, prop: str) -> list:
    vals = []
    for c in ent.get("claims", {}).get(prop, []):
        dv = c.get("mainsnak", {}).get("datavalue")
        if not dv:
            continue
        v = dv["value"]
        vals.append(v["id"] if isinstance(v, dict) and "id" in v else v)
    return vals


def label(ent: dict, default: str = "") -> str:
    labs = ent.get("labels", {})
    return (labs.get("es") or labs.get("mul") or labs.get("en") or {}).get("value", default)


def search(client: Client, text: str, limit: int = 10, lang: str = "es") -> list[str]:
    d = client.get_json(API, {"action": "wbsearchentities", "search": text, "language": lang,
                              "uselang": lang, "type": "item", "limit": limit, "format": "json"})
    return [x["id"] for x in d.get("search", [])]
