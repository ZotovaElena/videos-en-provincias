"""Phase 7b: interactive explorer → data/processed/explorer.html (one self-contained page, no server).

Leaflet (from unpkg) with the data embedded as JSON. No background tiles: sea colour, neighbouring
countries and the 61 units coloured like the static map.
- Top mode: each unit labelled with its winner (from `vmap rank`). All mode: number of videos per unit
  that pass the filters (a count of videos, not a YouTube metric).
- Click a unit: table of every clip with a place there, sorted by raw view count, with YouTube link,
  marks, status and evidence links (snippet on hover).
- Town markers: one per municipality / geocoded place, size by number of clips; popup with the list.
- Filters: minimum views, source type, show probable (styled as unconfirmed, never in the ranking).
"""
from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd

from vmap import config, units as un
from vmap.plot_map import NEIGHBOURS, unit_colors

LEAFLET = "https://unpkg.com/leaflet@1.9.4/dist/leaflet"


def _num(x):
    return None if x is None or pd.isna(x) else int(x)


def _str(x):
    return None if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)


def build_data() -> dict:
    p = config.PROCESSED
    winners = pd.read_csv(p / "winners.csv", dtype={"unit_id": str})
    pairs = pd.read_csv(p / "clips_by_unit.csv", dtype={"unit_id": str})
    places = pd.read_csv(p / "clips_by_place.csv", dtype={"unit_id": str})
    ev_rows = pd.read_csv(config.INTERIM / "location_evidence.csv", dtype={"unit_id": str})
    units = un.load()
    colors = unit_colors(units)

    ev = ev_rows.drop_duplicates("evidence_id").set_index("evidence_id")
    evidence = {eid: {"url": r["source_url"], "dom": r["domain"], "type": r["source_type"],
                      "txt": str(r["evidence_text"])[:300]} for eid, r in ev.iterrows()}
    clips = []
    for r in pairs.itertuples(index=False):
        ids = [e for e in str(r.evidence_ids).split() if e in evidence]
        clips.append({"u": r.unit_id, "c": r.clip_id, "s": _str(r.song_display) or _str(r.song_title),
                      "a": _str(r.artist_display) or _str(r.artist), "p": _str(r.location_mention),
                      "v": _num(r.view_count), "url": _str(r.youtube_url), "st": r.status, "m": _str(r.mark) or "",
                      "ev": ids, "src": sorted({evidence[e]["type"] for e in ids})})
    pl = []
    for key, g in places.groupby("place_key"):
        f = g.iloc[0]
        pl.append({"k": key, "n": _str(f["place_name"]), "u": f["unit_id"], "lat": float(f["lat"]),
                   "lon": float(f["lon"]), "clips": [{"c": c, "st": s} for c, s in zip(g["clip_id"], g["status"])]})

    w = winners.set_index("unit_id")
    simple = units.copy()
    simple["geometry"] = simple.geometry.simplify(0.002, preserve_topology=True)
    feats = []
    for _, r in simple.iterrows():
        wr = w.loc[r["unit_id"]]
        win = None
        if bool(wr["has_data"]):
            win = {"s": _str(wr["song_display"]), "a": _str(wr["artist_display"]), "m": _str(wr["mark"]) or "",
                   "url": _str(wr["youtube_url"]), "v": _num(wr["view_count"]), "c": _str(wr["clip_id"])}
        feats.append({"type": "Feature", "geometry": r.geometry.__geo_interface__,
                      "properties": {"id": r["unit_id"], "name": r["name"], "color": colors[r["unit_id"]],
                                     "win": win}})
    ne = gpd.read_file(config.RAW / "naturalearth" / "ne_10m_admin_0_countries.shp")
    ne = ne[ne["ADM0_A3"].isin(NEIGHBOURS)].clip((-20, 26, 6, 45))
    ne["geometry"] = ne.geometry.simplify(0.01, preserve_topology=True)
    snaps = sorted({str(s)[:10] for s in pairs["views_snapshot_date"].dropna()})
    return {"units": {"type": "FeatureCollection", "features": feats},
            "neighbours": json.loads(ne[["ADM0_A3", "geometry"]].to_json()),
            "clips": clips, "places": pl, "evidence": evidence, "snapshot": snaps[-1] if snaps else None}


PAGE = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Videoclips por provincia</title>
<link rel="stylesheet" href="__LEAFLET__.css">
<style>
:root{--sea:#dcebf5;--land:#eee6d3;--ink:#1f1f1f;--muted:#666;--panel:#fff;--line:#e2e2e2;--accent:#2457a6;--prob:#9a6a00}
*{box-sizing:border-box} html,body{margin:0;height:100%;font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink)}
#app{display:flex;height:100%} #map{flex:1;background:var(--sea)}
#side{width:440px;max-width:100%;overflow:auto;background:var(--panel);border-left:1px solid var(--line);padding:14px 16px}
@media (max-width:800px){#app{flex-direction:column}#map{height:55vh;flex:none}#side{width:100%;border-left:0;border-top:1px solid var(--line)}}
h1{font-size:17px;margin:0 0 6px} h2{font-size:16px;margin:14px 0 6px}
.ctl{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;font-size:13px;margin:8px 0;padding:8px;background:#f6f6f6;border-radius:6px}
.ctl label{white-space:nowrap} select{font:inherit}
table{border-collapse:collapse;width:100%;font-size:13px} td,th{padding:5px 4px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
th{font-size:12px;color:var(--muted);font-weight:600} td.v{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
a{color:var(--accent)} .song{font-weight:600} .muted{color:var(--muted);font-size:12px}
.prob td{color:var(--prob);font-style:italic} .badge{display:inline-block;font-size:11px;border:1px dashed var(--prob);color:var(--prob);border-radius:3px;padding:0 3px;font-style:normal}
.ev a{font-size:11px;margin-right:6px}
.lbl{background:transparent;border:0;box-shadow:none;font-size:11px;font-weight:600;color:#222;text-align:center;
text-shadow:0 0 3px #fff,0 0 3px #fff;white-space:normal;width:110px;line-height:1.15;padding:0}
.lbl::before{display:none} .lbl .n{font-size:13px}
footer{font-size:11px;color:var(--muted);margin-top:16px;border-top:1px solid var(--line);padding-top:8px}
button{font:inherit;font-size:12px}
</style></head><body><div id="app"><div id="map"></div><aside id="side">
<h1>El videoclip más visto rodado en cada provincia e isla</h1>
<div class="ctl">
 <label><input type="radio" name="mode" value="top" checked> Top</label>
 <label><input type="radio" name="mode" value="all"> Todos</label>
 <label>Mín. reproducciones <select id="minv">
  <option value="0">0</option><option value="1000000">1 M</option><option value="10000000">10 M</option>
  <option value="100000000">100 M</option><option value="1000000000">1000 M</option></select></label>
 <label><input type="checkbox" id="prob"> Mostrar no confirmados</label>
 <label><input type="checkbox" id="towns" checked> Localidades</label>
 <span>Fuentes:
  <label><input type="checkbox" class="src" value="press" checked> prensa</label>
  <label><input type="checkbox" class="src" value="wikipedia" checked> Wikipedia</label>
  <label><input type="checkbox" class="src" value="wikidata" checked> Wikidata</label>
  <label><input type="checkbox" class="src" value="manual" checked> manual</label></span>
 <button id="can">Canarias</button><button id="pen">Península</button>
</div>
<div id="detail"><p class="muted">Haz clic en una provincia o isla para ver todos sus videoclips, o en un
punto para ver los de esa localidad.</p></div>
<footer id="foot"></footer>
</aside></div>
<script src="__LEAFLET__.js"></script>
<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s);
const fmt = n => n == null ? "—" : n.toLocaleString("es-ES");
const safe = u => (typeof u === "string" && /^https?:\/\//.test(u)) ? u : null;
function el(tag, attrs, ...kids){const e=document.createElement(tag);for(const[k,v]of Object.entries(attrs||{})){if(v!=null)e.setAttribute(k,v)}
 for(const k of kids){if(k==null)continue;e.append(k instanceof Node?k:document.createTextNode(String(k)))}return e}
function a(href, text, title){const u=safe(href);return u?el("a",{href:u,target:"_blank",rel:"noopener",title:title||null},text):document.createTextNode(text)}

const byUnit = {}; for (const c of DATA.clips) (byUnit[c.u] ||= []).push(c);
const clipKey = c => c.u + "|" + c.c; const clipIndex = {}; for (const c of DATA.clips) clipIndex[clipKey(c)] = c;
const state = {mode:"top", minv:0, prob:false, towns:true, src:new Set(["press","wikipedia","wikidata","manual"]), unit:null};

function passes(c){
  if (c.st === "probable" && !state.prob) return false;
  if (c.st !== "confirmed" && c.st !== "probable") return false;
  if (state.minv > 0 && (c.v == null || c.v < state.minv)) return false;
  if (c.src.length && !c.src.some(s => state.src.has(s))) return false;
  return true;
}
const sortClips = l => l.slice().sort((x,y)=>(y.v??-1)-(x.v??-1));

const map = L.map("map", {zoomSnap:0.25, attributionControl:true});
map.attributionControl.setPrefix(false);
map.attributionControl.addAttribution("Límites: IGN-CNIG · Natural Earth · Lugares: © OpenStreetMap contributors");
L.geoJSON(DATA.neighbours, {style:{color:"#b8ab8f",weight:0.6,fillColor:"#eee6d3",fillOpacity:1}, interactive:false}).addTo(map);
const unitLayer = L.geoJSON(DATA.units, {style: styleUnit, onEachFeature: (f, layer) => {
  layer.on("click", () => showUnit(f.properties.id));
  layer.bindTooltip("", {permanent:true, direction:"center", className:"lbl"});
}}).addTo(map);
function styleUnit(f){
  const p = f.properties, sel = state.unit === p.id;
  const has = state.mode === "top" ? !!p.win : (byUnit[p.id]||[]).some(passes);
  return {color: sel ? "#222" : "#6f6f6f", weight: sel ? 2 : 0.6, fillColor: has ? p.color : "#d9d9d9", fillOpacity: 1};
}
function refreshUnits(){
  unitLayer.eachLayer(l => {
    const p = l.feature.properties; l.setStyle(styleUnit(l.feature));
    let node;
    if (state.mode === "top") {
      node = p.win ? el("div", {}, p.win.s || "")
                   : el("div", {style:"color:#888;font-weight:400;font-style:italic"}, "Sin datos");
    } else {
      const n = (byUnit[p.id]||[]).filter(passes).length;
      node = el("div", {}, n ? el("span", {class:"n"}, n) : "");
    }
    l.setTooltipContent(node);
  });
}

const placeLayer = L.layerGroup().addTo(map);
function refreshPlaces(){
  placeLayer.clearLayers(); if (!state.towns) return;
  for (const p of DATA.places){
    const cl = p.clips.map(x => clipIndex[p.u + "|" + x.c]).filter(c => c && passes(c));
    if (!cl.length) continue;
    const conf = cl.some(c => c.st === "confirmed");
    const m = L.circleMarker([p.lat, p.lon], {radius: 3 + 2.2*Math.sqrt(cl.length), color: conf ? "#7a2e2e" : "#9a6a00",
      weight: 1, dashArray: conf ? null : "2 2", fillColor: conf ? "#d9534f" : "#f0c36d", fillOpacity: 0.85});
    m.bindPopup(() => placePopup(p, cl)); m.bindTooltip(p.n || "", {direction:"top"}); m.addTo(placeLayer);
  }
}
function placePopup(p, cl){
  const box = el("div", {style:"max-width:280px"}, el("b", {}, p.n || ""), el("div", {class:"muted"}, cl.length + " videoclip(s)"));
  for (const c of sortClips(cl)) box.append(el("div", {style: c.st==="probable"?"color:#9a6a00;font-style:italic":""},
    a(c.url, c.s || "?"), " — ", c.a || "", " ", c.m, " · ", fmt(c.v), c.st==="probable" ? " (no confirmado)" : ""));
  return box;
}

function showUnit(id){
  state.unit = id; refreshUnits();
  const f = DATA.units.features.find(x => x.properties.id === id).properties;
  const list = sortClips((byUnit[id]||[]).filter(passes));
  const d = $("#detail"); d.replaceChildren();
  d.append(el("h2", {}, f.name));
  if (f.win) d.append(el("div", {}, "Ganador: ", el("span", {class:"song"}, a(f.win.url, f.win.s)), " — ", f.win.a, " ", f.win.m,
                         el("span", {class:"muted"}, " · " + fmt(f.win.v) + " reproducciones")));
  else d.append(el("div", {class:"muted"}, "Sin datos confirmados."));
  if (!list.length){ d.append(el("p", {class:"muted"}, "Ningún videoclip con estos filtros.")); return; }
  const t = el("table", {}, el("tr", {}, el("th", {}, "Videoclip"), el("th", {}, "Lugar"), el("th", {}, "Reproducciones"), el("th", {}, "Fuentes")));
  for (const c of list){
    const ev = el("div", {class:"ev"});
    for (const e of c.ev.slice(0, 4)) { const x = DATA.evidence[e]; if (x) ev.append(a(x.url, x.dom, x.txt)); }
    t.append(el("tr", {class: c.st === "probable" ? "prob" : ""},
      el("td", {}, el("div", {class:"song"}, a(c.url, c.s || "?")), el("div", {}, (c.a || "") + " " + c.m),
         c.st === "probable" ? el("span", {class:"badge"}, "no confirmado") : null,
         c.url ? null : el("div", {class:"muted"}, "sin vídeo oficial encontrado")),
      el("td", {}, c.p || ""), el("td", {class:"v"}, fmt(c.v)), el("td", {}, ev)));
  }
  d.append(t);
}

function refresh(){ refreshUnits(); refreshPlaces(); if (state.unit) showUnit(state.unit); }
document.querySelectorAll("input[name=mode]").forEach(r => r.addEventListener("change", e => { state.mode = e.target.value; refresh(); }));
$("#minv").addEventListener("change", e => { state.minv = +e.target.value; refresh(); });
$("#prob").addEventListener("change", e => { state.prob = e.target.checked; refresh(); });
$("#towns").addEventListener("change", e => { state.towns = e.target.checked; refresh(); });
document.querySelectorAll(".src").forEach(cb => cb.addEventListener("change", e => {
  e.target.checked ? state.src.add(e.target.value) : state.src.delete(e.target.value); refresh(); }));
$("#can").addEventListener("click", () => map.fitBounds([[27.5,-18.3],[29.5,-13.3]]));
$("#pen").addEventListener("click", () => map.fitBounds([[35.2,-9.5],[43.9,4.4]]));

const foot = $("#foot");
foot.append("Reproducciones: viewCount de YouTube tal como lo da la API, a " + (DATA.snapshot || "—") +
  " (sin sumas ni medias). Lugar de rodaje: prensa, Wikipedia y Wikidata; solo videoclips con rodaje documentado, ",
  "por lo que puede faltar alguno. † rodado solo en parte allí o en una zona compartida · ‡ una sola fuente o sin prensa de referencia. ",
  a("https://www.youtube.com/t/terms", "Condiciones de servicio de YouTube"), ".");
map.fitBounds([[35.2,-9.5],[43.9,4.4]]); refresh();
</script></body></html>
"""


def run() -> dict:
    data = build_data()
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = PAGE.replace("__LEAFLET__", LEAFLET).replace("__DATA__", blob)
    out = config.PROCESSED / "explorer.html"
    out.write_text(page, encoding="utf-8")
    return {"page": str(out.relative_to(config.ROOT)), "size_kb": round(out.stat().st_size / 1024),
            "clips": len(data["clips"]), "places": len(data["places"]), "snapshot": data["snapshot"]}
