"""Place mention → unit(s).

- municipality → its unit (island provinces through the municipality's point: Teguise → Lanzarote);
  several municipalities with the same name → the one in a province named by the context, else review
- island → the island unit; province → the province (not for 07/35/38: an island is needed)
- region → its province when it has only one (La Rioja, Asturias...), else not assignable
- landmark / unresolved name / Wikidata item → coordinates (Wikidata P625 or Nominatim, cached,
  1 req/s, Spain only) → point in polygon. Nominatim areas that cover ≥ 2 units (Picos de Europa,
  Doñana...) are `shared_area` and go to every unit they cover.
Each result carries `reasons`: anything that should send the place to human review.
"""
from __future__ import annotations

from vmap import units as un, wd
from vmap.http import Client, HttpError, OfflineError

NOMINATIM = "https://nominatim.openstreetmap.org/search"
ISLAND_PROVINCES = {"07", "35", "38"}
SPECIFICITY = {"island": 0, "municipality": 1, "landmark": 2, "province": 3, "region": 4}
# Nominatim results accepted as filming places (OSM category → types; None = any type)
OK_CATEGORIES = {
    "place": {"city", "town", "village", "hamlet", "suburb", "neighbourhood", "quarter", "island", "islet",
              "locality", "isolated_dwelling", "square", "archipelago", "municipality"},
    "natural": None, "boundary": {"national_park", "protected_area", "administrative"},
    "tourism": {"attraction", "museum", "viewpoint", "theme_park"}, "historic": None,
    "leisure": {"park", "nature_reserve", "stadium", "beach_resort", "garden"},
    "building": {"cathedral", "church", "castle"}, "man_made": {"lighthouse", "bridge", "tower", "pier"},
    "amenity": {"place_of_worship", "theatre"}, "waterway": None, "landuse": {"military"},
}
NEAR_BOUNDARY_M = 100
SHARED_MIN_FRACTION = 0.05
MIN_IMPORTANCE_NO_CONTEXT = 0.25


def _ok_result(r: dict) -> bool:
    cat, typ = r.get("category") or r.get("class"), r.get("type")
    allowed = OK_CATEGORIES.get(cat, "missing")
    return allowed is None or (allowed != "missing" and typ in allowed)


class GeoResolver:
    def __init__(self, client: Client | None = None, offline_ok: bool = True):
        self.client = client or Client(stage="geo")
        self.units, self.units_m = un.load(), un.load_metric()
        self.munis = un.municipalities().set_index("ine")
        from vmap.extract import gazetteer as gz

        admin = gz.admin_names()
        prov = admin[admin["kind"] == "province"]
        self.province_name = {c: n.split("/")[0] for c, n in zip(prov["code"], prov["name"])}
        self.offline_ok = offline_ok
        self.geocode_errors = 0

    # ------------------------------------------------------------ helpers
    def _unit_of_muni(self, ine: str) -> dict:
        m = self.munis.loc[ine]
        return {"unit_ids": [m["unit_id"]], "ine": ine, "lat": float(m["lat"]), "lon": float(m["lon"]),
                "precision": "municipality", "shared_area": False, "reasons": []}

    def _point(self, lat: float, lon: float, precision: str, method: str) -> dict:
        loc = un.locate(lon, lat)
        reasons = []
        if loc["unit_id"] is None:
            reasons.append("outside_spain")
        elif loc["boundary_m"] < NEAR_BOUNDARY_M:
            reasons.append("near_boundary")
        return {"unit_ids": [loc["unit_id"]] if loc["unit_id"] else [], "ine": None, "lat": round(lat, 5),
                "lon": round(lon, 5), "precision": precision, "method": method, "shared_area": False,
                "reasons": reasons}

    def _area_units(self, geojson: dict) -> list[str]:
        import geopandas as gpd
        from shapely.geometry import shape

        geom = shape(geojson)
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            return []
        g = gpd.GeoSeries([geom], crs=4326).to_crs(un.METRIC).iloc[0]
        if g.area <= 0:
            return []
        frac = self.units_m.intersection(g).area / g.area
        return self.units.loc[frac[frac >= SHARED_MIN_FRACTION].index, "unit_id"].tolist()

    def geocode(self, q: str) -> list[dict]:
        params = {"q": q, "format": "jsonv2", "countrycodes": "es", "limit": 5, "polygon_geojson": 1,
                  "polygon_threshold": 0.005, "accept-language": "es"}
        try:
            return self.client.get_json(NOMINATIM, params)
        except (HttpError, OfflineError):
            self.geocode_errors += 1
            if not self.offline_ok:
                raise
            return []

    # ------------------------------------------------------------ main entry
    def resolve(self, place: dict, ctx_provinces: set[str]) -> dict:
        """One place mention (from rules.extract) → {unit_ids, ine, lat, lon, precision, shared_area, reasons}."""
        if place.get("field") == "wikidata":
            return self._wikidata(place)
        cands = place.get("candidates") or []
        if not cands:
            return self._geocoded(place["mention"], ctx_provinces, landmark=False)
        best = min(SPECIFICITY.get(c["kind"], 9) for c in cands)
        top = [c for c in cands if SPECIFICITY.get(c["kind"], 9) == best]
        kind = top[0]["kind"]
        if kind == "island":
            return {"unit_ids": [top[0]["code"]], "ine": None, "lat": None, "lon": None, "precision": "island",
                    "shared_area": False, "reasons": []}
        if kind == "municipality":
            if len(top) > 1:
                inside = [c for c in top if set(c["provinces"]) & ctx_provinces]
                if len(inside) != 1:
                    return self._none("several_municipalities")
                top = inside
            return self._unit_of_muni(top[0]["code"])
        if kind == "landmark":
            return self._geocoded(top[0]["name"], ctx_provinces, landmark=True)
        provs = sorted({p for c in top for p in c["provinces"]})
        if len(provs) != 1:
            return self._none(f"{kind}_level")
        if provs[0] in ISLAND_PROVINCES:
            return self._none("island_province_level")
        return {"unit_ids": provs, "ine": None, "lat": None, "lon": None, "precision": kind,
                "shared_area": False, "reasons": []}

    @staticmethod
    def _none(reason: str) -> dict:
        return {"unit_ids": [], "ine": None, "lat": None, "lon": None, "precision": None,
                "shared_area": False, "reasons": [reason]}

    def _wikidata(self, place: dict) -> dict:
        coords = place.get("coords") or []
        if not coords and place.get("qid"):
            ent = wd.entities(self.client, [place["qid"]], props="claims").get(place["qid"], {})
            coords = [(v["latitude"], v["longitude"]) for v in wd.claim_values(ent, "P625")]
        if not coords:
            return self._none("no_coordinates")
        lat, lon = coords[0]
        return self._point(lat, lon, "point", "wikidata:P625")

    def _geocoded(self, name: str, ctx_provinces: set[str], landmark: bool) -> dict:
        """Nominatim, Spain only. With a context province the result must lie in it."""
        ctx = sorted(ctx_provinces)
        queries = [f"{name}, {self.province_name[c]}" for c in ctx if c in self.province_name][:2] + [name]
        for q in queries:
            results = [r for r in self.geocode(q) if _ok_result(r)]
            if not results:
                continue
            r = max(results, key=lambda x: float(x.get("importance") or 0))
            res = self._point(float(r["lat"]), float(r["lon"]), "point", "nominatim")
            res["geocode_name"] = r.get("display_name", "")[:120]
            res["osm"] = f"{r.get('category') or r.get('class')}/{r.get('type')}"
            area = self._area_units(r["geojson"]) if r.get("geojson") else []
            if len(area) >= 2:
                res.update(unit_ids=area, shared_area=True, precision="area")
                res["reasons"] = [x for x in res["reasons"] if x != "near_boundary"]
            in_ctx = bool(ctx) and any(u[:2] in ctx_provinces for u in res["unit_ids"])
            if ctx and not in_ctx:
                res["reasons"].append("geocode_outside_context")
            elif not ctx and not landmark and float(r.get("importance") or 0) < MIN_IMPORTANCE_NO_CONTEXT:
                res["reasons"].append("geocode_low_importance")
            if not landmark and not ctx:
                res["reasons"].append("geocode_unchecked")
            return res
        return self._none("not_geocoded")
