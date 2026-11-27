"""Phase 7: static map → data/processed/mapa.svg, mapa.png (2048 px wide), results.geojson.

Mainland + Balearics + Ceuta + Melilla in EPSG:25830; Canary Islands in a framed inset (EPSG:32628),
lower left. Sea light blue, neighbouring countries beige (Natural Earth), units in pastel colours from a
greedy graph colouring of the adjacency graph (neighbours never share a colour), "Sin datos" in grey.
Each label: song title (bold) + artist (smaller) + marks; small units and islands are labelled outside
with a leader line (adjustText). In the SVG each title links to its YouTube video.
"""
from __future__ import annotations

import textwrap

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402

from vmap import config, units as un  # noqa: E402

PALETTE = ["#f6d7a7", "#cfe6c1", "#c6dbf0", "#f2c5cf", "#e0d4ef", "#f9f1b3"]
NODATA, SEA, LAND, EDGE = "#d9d9d9", "#dcebf5", "#eee6d3", "#6f6f6f"
NEIGHBOURS = ["PRT", "FRA", "AND", "MAR", "DZA", "GIB", "SAH"]
CANARIAS = ("35", "38")
TITLE = "El videoclip más visto en YouTube rodado en cada provincia e isla"
LEGEND = ("† Rodado solo en parte allí o en una comarca compartida  ·  "
          "‡ Una sola fuente, o sin prensa de referencia que lo confirme")
# initial label offsets (km) for units whose label goes outside, before adjustText tidies them up
OFFSETS = {"07-MEN": (70, 70), "07-MAL": (120, -85), "07-IBI": (-20, 95), "07-FOR": (95, -60),
           "51": (70, -75), "52": (90, -45),
           "38-TFE": (0, 60), "38-LPA": (-10, 70), "38-GOM": (-30, -60), "38-HIE": (0, -55),
           "35-GCA": (0, -60), "35-LAN": (60, 0), "35-FUE": (65, -20), "35-GRA": (40, 45)}
OUTSIDE = set(OFFSETS)          # only islands, Ceuta and Melilla are labelled outside their unit


def unit_colors(units: gpd.GeoDataFrame) -> dict[str, str]:
    """Greedy colouring (DSatur) of the 'touches' graph; islands are isolated nodes."""
    m = gpd.GeoDataFrame(units[["unit_id"]], geometry=units.to_crs(un.METRIC).buffer(200), crs=un.METRIC)
    pairs = gpd.sjoin(m, m, predicate="intersects")
    g = nx.Graph()
    g.add_nodes_from(units["unit_id"])
    g.add_edges_from((a, b) for a, b in zip(pairs["unit_id_left"], pairs["unit_id_right"]) if a != b)
    col = nx.greedy_color(g, strategy="saturation_largest_first")
    if max(col.values()) >= len(PALETTE):
        raise ValueError(f"colouring needs {max(col.values()) + 1} colours, palette has {len(PALETTE)}")
    return {u: PALETTE[c] for u, c in col.items()}


def label_text(w) -> str:
    """'Artist – Song' plus marks, wrapped; '' for units without data."""
    if not w.get("has_data"):
        return ""
    mark = f" {w['mark']}" if isinstance(w.get("mark"), str) and w["mark"] else ""
    return textwrap.fill(f"{w['artist_display']} – {w['song_display']}{mark}", 20)


def label_point(geom):
    """A point inside the largest part of the unit (never on an islet or in the sea)."""
    parts = list(getattr(geom, "geoms", [geom]))
    return max(parts, key=lambda g: g.area).representative_point()


def _labels(ax, gdf: gpd.GeoDataFrame, fs: float):
    from adjustText import adjust_text

    texts, tx, ty, outside = [], [], [], []
    for _, w in gdf.iterrows():
        pt = label_point(w.geometry)
        dx, dy = OFFSETS.get(w["unit_id"], (0, 0))
        s = label_text(w)
        t = ax.text(pt.x + dx * 1000, pt.y + dy * 1000, s or "Sin datos", fontsize=fs if s else fs * 0.9,
                    ha="center", va="center", weight="bold" if s else "normal", color="#1d1d1d" if s else "#777",
                    style="normal" if s else "italic", zorder=6, linespacing=1.05)
        if s and isinstance(w.get("youtube_url"), str):
            t.set_url(w["youtube_url"])
        texts.append(t)
        tx.append(pt.x)
        ty.append(pt.y)
        outside.append(w["unit_id"] in OUTSIDE)
    # gentle forces: labels stay near their point (inside the unit); only crowded ones move a little
    adjust_text(texts, target_x=tx, target_y=ty, ax=ax, expand=(1.03, 1.1), min_arrow_len=12,
                force_text=(0.05, 0.1), force_pull=(0.05, 0.05), time_lim=20,
                arrowprops=dict(arrowstyle="-", color="#333", lw=0.7, shrinkA=2, shrinkB=0))
    for t, x, y, out in zip(texts, tx, ty, outside):
        if out:                                                  # label outside its unit: dot on the place
            ax.plot([x], [y], "o", ms=2.2, color="#333", zorder=7)


def plot(winners: pd.DataFrame, out_dir=config.PROCESSED) -> dict:
    units = un.load()
    colors = unit_colors(units)
    u = units.merge(winners.drop(columns=["unit_name"]), on="unit_id", how="left")
    u["fill"] = [colors[x] if d else NODATA for x, d in zip(u["unit_id"], u["has_data"].fillna(False))]
    ne = gpd.read_file(config.RAW / "naturalearth" / "ne_10m_admin_0_countries.shp")
    ne = ne[ne["ADM0_A3"].isin(NEIGHBOURS)]

    fig = plt.figure(figsize=(16, 12), dpi=128)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0, 0.07, 1, 0.85))
    ax.set_facecolor(SEA)
    main = u[~u["unit_id"].str.startswith(CANARIAS)].to_crs(25830)
    xmin, ymin, xmax, ymax = main.total_bounds
    ax.set_xlim(xmin - 470e3, xmax + 160e3)
    ax.set_ylim(ymin - 50e3, ymax + 70e3)
    ne.to_crs(25830).clip((xmin - 900e3, ymin - 400e3, xmax + 600e3, ymax + 400e3)).plot(
        ax=ax, color=LAND, edgecolor="#b8ab8f", lw=0.5)
    main.plot(ax=ax, color=main["fill"], edgecolor=EDGE, lw=0.45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)
    _labels(ax, main, fs=6.6)

    ins = fig.add_axes((0.012, 0.085, 0.27, 0.19))
    ins.set_facecolor(SEA)
    can = u[u["unit_id"].str.startswith(CANARIAS)].to_crs(32628)
    cx0, cy0, cx1, cy1 = can.total_bounds
    ins.set_xlim(cx0 - 60e3, cx1 + 90e3)
    ins.set_ylim(cy0 - 70e3, cy1 + 70e3)
    can.plot(ax=ins, color=can["fill"], edgecolor=EDGE, lw=0.45)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_xlabel("")
    ins.set_ylabel("")
    for s in ins.spines.values():
        s.set_edgecolor("#555")
        s.set_linewidth(0.8)
    ins.text(0.02, 0.97, "Canarias", transform=ins.transAxes, fontsize=8, va="top", style="italic", color="#444")
    _labels(ins, can, fs=6.2)

    snap = sorted({str(s)[:10] for s in winners["views_snapshot_date"].dropna()})
    date = snap[-1] if snap else "—"
    fig.text(0.5, 0.965, TITLE, ha="center", va="center", fontsize=19, weight="bold")
    fig.text(0.5, 0.935, "Entre los videoclips con lugar de rodaje documentado. Reproducciones a "
             f"{date}.", ha="center", va="center", fontsize=10, color="#444")
    fig.text(0.5, 0.05, LEGEND, ha="center", fontsize=9, color="#333")
    fig.text(0.5, 0.025, f"Reproducciones: YouTube (viewCount), {date} · Rodaje: prensa, Wikipedia, Wikidata · "
             "Límites: IGN-CNIG, Natural Earth · Condiciones de YouTube: https://www.youtube.com/t/terms",
             ha="center", fontsize=8, color="#555")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "mapa.svg")
    fig.savefig(out_dir / "mapa.png", dpi=128)
    plt.close(fig)

    gj = u.drop(columns=["fill"]).copy()
    gj["geometry"] = gj.geometry.simplify(0.001, preserve_topology=True)
    keep = ["unit_id", "name", "kind", "has_data", "song_display", "artist_display", "mark", "youtube_url",
            "view_count", "views_snapshot_date", "location_mention", "n_sources", "sources", "clip_id", "geometry"]
    gj = gj[[c for c in keep if c in gj.columns]]
    (out_dir / "results.geojson").write_text(gj.to_json(na="null", drop_id=True), encoding="utf-8")
    return {"svg": "mapa.svg", "png": "mapa.png", "geojson": "results.geojson", "snapshot": date,
            "colours_used": len(set(colors.values()))}


def run() -> dict:
    winners = pd.read_csv(config.PROCESSED / "winners.csv", dtype={"unit_id": str})
    return plot(winners)
