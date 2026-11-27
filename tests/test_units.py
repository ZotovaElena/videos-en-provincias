"""Units: island splitting on synthetic shapes (always runs) and checks on the built units.gpkg
(skipped when data/interim/units.gpkg has not been built on this machine)."""
import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, box

from vmap import units as un

# --------------------------------------------------------------- synthetic (offline, no data files)


def test_build_islands_anchor_part_and_islets():
    # province "35": big island A, small island B, islet C close to B
    a, b, c = box(0, 0, 1, 1), box(2, 0, 2.2, 0.2), box(2.3, 0, 2.32, 0.02)
    prov = gpd.GeoDataFrame({"province": ["35"], "geometry": [MultiPolygon([a, b, c])]}, crs=4326)
    # municipality "T" covers half of A and all of B (like Teguise: Lanzarote + La Graciosa)
    muni = gpd.GeoDataFrame({"province": ["35", "35"], "NAMEUNIT": ["T", "U"],
                             "geometry": [MultiPolygon([box(0, 0, 0.5, 1), b]), box(0.5, 0, 1, 1)]}, crs=4326)
    islands = {"35-A": {"name": "A", "province": "35", "anchor": "U"},
               "35-B": {"name": "B", "province": "35", "anchor": "T", "part": 2}}
    out = un.build_islands(prov, muni, islands).to_crs(4326).set_index("unit_id")
    assert set(out.index) == {"35-A", "35-B"}
    assert out.loc["35-B", "geometry"].contains(c.centroid)       # islet joins nearest island
    assert out.loc["35-A", "geometry"].contains(a.centroid)


def test_build_islands_rejects_shared_main_part():
    prov = gpd.GeoDataFrame({"province": ["07"], "geometry": [box(0, 0, 1, 1)]}, crs=4326)
    muni = gpd.GeoDataFrame({"province": ["07", "07"], "NAMEUNIT": ["P", "Q"],
                             "geometry": [box(0, 0, 0.5, 1), box(0.5, 0, 1, 1)]}, crs=4326)
    islands = {"07-P": {"name": "P", "province": "07", "anchor": "P"},
               "07-Q": {"name": "Q", "province": "07", "anchor": "Q"}}
    with pytest.raises(ValueError, match="share the same main part"):
        un.build_islands(prov, muni, islands)


# --------------------------------------------------------------- built units (local data)
needs_data = pytest.mark.skipif(not un.UNITS_GPKG.exists(), reason="units.gpkg not built")


@pytest.fixture(scope="module")
def units():
    return un.load()


@needs_data
def test_61_valid_units(units):
    assert len(units) == 61 and units["unit_id"].is_unique
    assert units.is_valid.all()
    assert (units["kind"] == "island").sum() == 12
    assert {"51", "52", "07-MAL", "35-GRA", "38-HIE"} <= set(units["unit_id"])
    assert not {"07", "35", "38"} & set(units["unit_id"])


def _piece_point(name: str, rank: int):
    """A point on the rank-th largest piece of an IGN municipality (from the data, not typed by hand)."""
    muni = un._read(un.MUNI_SHP)
    parts = un._parts(muni[muni["NAMEUNIT"] == name]).sort_values("area_km2", ascending=False)
    pt = gpd.GeoSeries([parts.geometry.iloc[rank - 1].representative_point()], crs=un.METRIC).to_crs(4326)[0]
    return pt.x, pt.y, parts["area_km2"].iloc[rank - 1]


@needs_data
def test_la_graciosa_is_its_own_unit(units):
    lon, lat, area = _piece_point("Teguise", 2)          # La Graciosa: 2nd piece of Teguise (~27 km2)
    assert 20 < area < 35
    assert un.locate(lon, lat, units)["unit_id"] == "35-GRA"
    lon, lat, _ = _piece_point("Teguise", 1)             # the rest of Teguise is on Lanzarote
    assert un.locate(lon, lat, units)["unit_id"] == "35-LAN"


@needs_data
def test_cabrera_belongs_to_mallorca(units):
    lon, lat, area = _piece_point("Palma", 2)            # Cabrera: 2nd piece of Palma (~11 km2)
    assert 8 < area < 15 and lat < 39.3                  # south of Mallorca
    assert un.locate(lon, lat, units)["unit_id"] == "07-MAL"


@needs_data
def test_open_sea_has_no_unit(units):
    assert un.locate(-20.0, 40.0, units)["unit_id"] is None
