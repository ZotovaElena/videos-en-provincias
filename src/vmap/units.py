"""The 61 territorial units → data/interim/units.gpkg.

47 peninsular provinces + Ceuta + Melilla (IGN provincial polygons) and 12 islands cut out of the
provinces 07, 35 and 38 (config/islands.yaml). Islets join the nearest island of their province
(Cabrera → Mallorca); the small territories outside any province (Chafarinas, Alhucemas, Vélez de la
Gomera, Perejil, Isla de los Faisanes, ...) join the nearest unit.

Layers: `units` in EPSG:4326. Areas and distances are computed in EPSG:3035 (equal-area, covers the
Canaries too; EPSG:25830 is badly distorted there).
"""
from __future__ import annotations

from functools import lru_cache

import geopandas as gpd
import pandas as pd
import yaml
from shapely import make_valid
from shapely.ops import unary_union

from vmap import config

IGN = config.RAW / "lineas_limite"
PROV_SHP = [IGN / "SHP_ETRS89/recintos_provinciales_inspire_peninbal_etrs89/"
                  "recintos_provinciales_inspire_peninbal_etrs89.shp",
            IGN / "SHP_WGS84/recintos_provinciales_inspire_canarias_wgs84/"
                  "recintos_provinciales_inspire_canarias_wgs84.shp"]
MUNI_SHP = [IGN / "SHP_ETRS89/recintos_municipales_inspire_peninbal_etrs89/"
                  "recintos_municipales_inspire_peninbal_etrs89.shp",
            IGN / "SHP_WGS84/recintos_municipales_inspire_canarias_wgs84/"
                  "recintos_municipales_inspire_canarias_wgs84.shp"]
UNITS_GPKG = config.INTERIM / "units.gpkg"
METRIC = 3035
ISLAND_PROVINCES = {"07", "35", "38"}
NO_PROVINCE = "54"                       # IGN "Territorios no asociados a ninguna provincia"


def _read(paths) -> gpd.GeoDataFrame:
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"IGN boundaries not found: {missing}. Put the IGN 'Lineas limite' "
                                f"shapefiles in {IGN}.")
    return gpd.GeoDataFrame(pd.concat([gpd.read_file(p).to_crs(4326) for p in paths], ignore_index=True),
                            crs=4326)


def _parts(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.explode(index_parts=False).reset_index(drop=True).to_crs(METRIC)
    out["area_km2"] = out.area / 1e6
    return out


def build_islands(prov: gpd.GeoDataFrame, muni: gpd.GeoDataFrame, islands: dict) -> gpd.GeoDataFrame:
    """One row per island (metric CRS): main part found through its anchor municipality, plus islets."""
    rows = []
    for code in sorted(ISLAND_PROVINCES):
        parts = _parts(prov[prov["province"] == code])
        mparts = _parts(muni[muni["province"] == code])
        owner: dict[int, str] = {}
        for unit, spec in islands.items():
            if spec["province"] != code:
                continue
            pieces = mparts[mparts["NAMEUNIT"] == spec["anchor"]].sort_values("area_km2", ascending=False)
            rank = spec.get("part", 1)
            if len(pieces) < rank:
                raise ValueError(f"{unit}: anchor municipality {spec['anchor']!r} (province {code}) has "
                                 f"{len(pieces)} pieces, need piece #{rank}")
            pt = pieces.geometry.iloc[rank - 1].representative_point()
            hits = parts.index[parts.contains(pt)].tolist()
            if len(hits) != 1:
                raise ValueError(f"{unit}: anchor point lies in {len(hits)} parts of province {code}")
            if hits[0] in owner:
                raise ValueError(f"{unit} and {owner[hits[0]]} share the same main part")
            owner[hits[0]] = unit
        main = parts.loc[list(owner)].assign(unit_id=[owner[i] for i in owner])
        for i, part in parts.iterrows():                       # islets → nearest main island
            if i not in owner:
                owner[i] = main.loc[main.distance(part.geometry).idxmin(), "unit_id"]
        parts["unit_id"] = [owner[i] for i in parts.index]
        for unit, grp in parts.groupby("unit_id"):
            rows.append({"unit_id": unit, "name": islands[unit]["name"], "province": code, "kind": "island",
                         "geometry": unary_union(grp.geometry.tolist())})
    return gpd.GeoDataFrame(rows, crs=METRIC)


def build() -> gpd.GeoDataFrame:
    prov = _read(PROV_SHP)
    prov["province"] = prov["NATCODE"].str[4:6]
    muni = _read(MUNI_SHP)
    muni["province"] = muni["NATCODE"].str[6:8]
    with open(config.CONFIG / "islands.yaml", encoding="utf-8") as f:
        islands = yaml.safe_load(f)["islands"]

    mainland = prov[~prov["province"].isin(ISLAND_PROVINCES | {NO_PROVINCE})].to_crs(METRIC)
    units = gpd.GeoDataFrame({
        "unit_id": mainland["province"], "name": mainland["NAMEUNIT"], "province": mainland["province"],
        "kind": mainland["province"].map(lambda c: "city" if c in ("51", "52") else "province"),
        "geometry": mainland.geometry}, crs=METRIC)
    units = units.dissolve(by="unit_id", aggfunc="first").reset_index()
    units = pd.concat([units, build_islands(prov, muni, islands)], ignore_index=True)

    extra = _parts(prov[prov["province"] == NO_PROVINCE])           # loose territories → nearest unit
    units = units.set_index("unit_id")
    # nearest unit measured on the original units, so one merge does not pull the next islet along
    near = {i: units.distance(part.geometry).idxmin() for i, part in extra.iterrows()}
    for i, unit in near.items():
        units.loc[unit, "geometry"] = unary_union([units.loc[unit, "geometry"], extra.loc[i, "geometry"]])
    units = gpd.GeoDataFrame(units.reset_index(), crs=METRIC)
    units["geometry"] = units.geometry.map(make_valid)
    units["area_km2"] = (units.area / 1e6).round(1)
    return units.to_crs(4326).sort_values("unit_id").reset_index(drop=True)


def write(units: gpd.GeoDataFrame) -> None:
    UNITS_GPKG.parent.mkdir(parents=True, exist_ok=True)
    units.to_file(UNITS_GPKG, layer="units", driver="GPKG")


@lru_cache(maxsize=1)
def load() -> gpd.GeoDataFrame:
    if not UNITS_GPKG.exists():
        raise FileNotFoundError(f"{UNITS_GPKG} missing: run `vmap units` first.")
    return gpd.read_file(UNITS_GPKG, layer="units")


def locate(lon: float, lat: float, units: gpd.GeoDataFrame | None = None, max_km: float = 1.0) -> dict:
    """Unit of a point and its distance to the unit boundary (m). Points just off the coast (≤ max_km)
    go to the nearest unit; farther points → unit_id None."""
    from shapely.geometry import Point

    units = load() if units is None else units
    m = units.to_crs(METRIC)
    pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(METRIC).iloc[0]
    inside = m.index[m.contains(pt)].tolist()
    if inside:
        i = inside[0]
        return {"unit_id": units.loc[i, "unit_id"], "inside": True,
                "boundary_m": round(m.geometry.loc[i].boundary.distance(pt))}
    d = m.distance(pt)
    i = d.idxmin()
    if d[i] <= max_km * 1000:
        return {"unit_id": units.loc[i, "unit_id"], "inside": False, "boundary_m": round(d[i])}
    return {"unit_id": None, "inside": False, "boundary_m": round(d[i])}
