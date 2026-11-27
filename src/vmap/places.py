"""Gazetteer of Spanish municipalities → data/interim/places.csv.

Names and INE codes come from the IGN municipal boundaries (NATCODE = 34 + CA(2) + province(2) + INE(5)).
Population and province capitals come from Wikidata (P772 INE code, P1082 population, current P36 capital
statements of each province). `is_capital` also includes the most populous municipality of each province,
so every province has at least one top-priority place. It is only used to order the press search.
"""
from __future__ import annotations

import pandas as pd
import yaml

from vmap import config, wd
from vmap.http import Client

IGN = config.RAW / "lineas_limite"
MUNI_SHP = [
    IGN / "SHP_ETRS89/recintos_municipales_inspire_peninbal_etrs89/recintos_municipales_inspire_peninbal_etrs89.shp",
    IGN / "SHP_WGS84/recintos_municipales_inspire_canarias_wgs84/recintos_municipales_inspire_canarias_wgs84.shp",
]
PLACES_CSV = config.INTERIM / "places.csv"

POP_QUERY = """
SELECT ?ine (MAX(?p) AS ?pop) WHERE { ?m wdt:P772 ?ine ; wdt:P1082 ?p . } GROUP BY ?ine
"""
PROVINCES_QUERY = """
SELECT DISTINCT ?prov WHERE { ?prov wdt:P31 wd:Q162620 . }
"""


def name_variants(nameunit: str) -> list[str]:
    """'Donostia/San Sebastián' → both; capitalise IGN's lowercase articles ('el Pinós' → 'El Pinós')."""
    out = []
    for part in nameunit.split("/"):
        p = part.strip()
        if p:
            out.append(p[0].upper() + p[1:])
    return out


def current_capitals(client: Client, endpoint: str) -> set[str]:
    """INE codes of the current capitals (P36 without end date P582) of the provinces of Spain."""
    rows = client.get_json(endpoint, {"query": PROVINCES_QUERY, "format": "json"},
                           headers={"Accept": "application/sparql-results+json"})["results"]["bindings"]
    provs = wd.entities(client, [b["prov"]["value"].rsplit("/", 1)[-1] for b in rows], props="claims")
    cap_qids = set()
    for e in provs.values():
        for c in e.get("claims", {}).get("P36", []):
            dv = c["mainsnak"].get("datavalue")
            if dv and "P582" not in c.get("qualifiers", {}) and c.get("rank") != "deprecated":
                cap_qids.add(dv["value"]["id"])
    caps = wd.entities(client, cap_qids, props="claims")
    return {v for e in caps.values() for v in wd.claim_values(e, "P772")}


def read_ign() -> pd.DataFrame:
    import geopandas as gpd

    missing = [p for p in MUNI_SHP if not p.exists()]
    if missing:
        raise FileNotFoundError(f"IGN municipal boundaries not found: {missing}. "
                                "Download 'Lineas limite municipales' from the IGN-CNIG Centro de Descargas "
                                f"into {IGN}.")
    df = pd.concat([gpd.read_file(p, ignore_geometry=True) for p in MUNI_SHP], ignore_index=True)
    df["ine"] = df["NATCODE"].str[-5:]
    df["province"] = df["NATCODE"].str[6:8]
    return df[~df["ine"].str.startswith("53")][["ine", "province", "NAMEUNIT"]]  # 53xxx: shared lands


def build(client: Client | None = None) -> pd.DataFrame:
    client = client or Client(stage="places")
    endpoint = config.search()["wikidata"]["endpoint"]
    df = read_ign()
    pop = client.get_json(endpoint, {"query": POP_QUERY, "format": "json"},
                          headers={"Accept": "application/sparql-results+json"})["results"]["bindings"]
    pops = {b["ine"]["value"]: int(float(b["pop"]["value"])) for b in pop}
    capitals = current_capitals(client, endpoint)
    df["population"] = df["ine"].map(pops).astype("Int64")
    df["is_capital"] = df["ine"].isin(capitals)
    # every province gets at least its most populous municipality in the top tier
    largest = df.sort_values("population", ascending=False).groupby("province")["ine"].first()
    df["is_capital"] |= df["ine"].isin(set(largest))
    df["name"] = df["NAMEUNIT"]
    aliases = yaml.safe_load((config.CONFIG / "place_aliases.yaml").read_text(encoding="utf-8")) or {}
    df["variants"] = [
        "|".join(dict.fromkeys(name_variants(n) + [str(a) for a in aliases.get(ine, [])]))
        for n, ine in zip(df["NAMEUNIT"], df["ine"])]
    df = df.drop(columns="NAMEUNIT").sort_values(["is_capital", "population"], ascending=[False, False])
    PLACES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PLACES_CSV, index=False)
    return df


def load() -> pd.DataFrame:
    if not PLACES_CSV.exists():
        raise FileNotFoundError(f"{PLACES_CSV} missing: run `vmap places` first.")
    return pd.read_csv(PLACES_CSV, dtype={"ine": str, "province": str})
