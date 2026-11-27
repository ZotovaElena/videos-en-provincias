"""Local review editor: `vmap edit` → http://127.0.0.1:8765 (this computer only, Python stdlib server).

Tabs:
  Revisar         every (clip, unit) with its evidence; buttons accept / reject / force winner + note
                  → data/manual/overrides.csv (saved on every click)
  Añadir          form for a music video the person knows → data/manual/manual_evidence.csv
                  (quoted from a source, or the person's own observation of the video)
  Reconstruir     runs the pipeline again (manual → extract → resolve → youtube (no new searches)
                  → rank → review → explore [→ plot]) and shows the log
Only requests whose Host is this server and whose body is JSON are accepted (no cross-site writes).
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

from vmap import config, rank
from vmap.sources import manual
from vmap.sources.seed import MANUAL_COLUMNS, MANUAL_CSV

LOCK = threading.Lock()
VIDEO_LINKS = config.MANUAL / "video_links.csv"
VIDEO_COLUMNS = ["clip_id", "youtube_id", "note"]
REBUILD_STEPS = [["manual"], ["extract"], ["resolve"], ["youtube", "--max-queries", "0", "--no-search"],
                 ["rank"], ["review"], ["explore"]]


def _csv(path, columns) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in columns:
        if c not in df:
            df[c] = ""
    return df[columns]


def save_override(unit_id: str, clip_id: str, action: str, note: str) -> None:
    """Set (or with action '' remove) the decision for one (unit, clip) pair."""
    with LOCK:
        ov = _csv(rank.OVERRIDES, rank.OVERRIDE_COLUMNS)
        ov = ov[~((ov["unit_id"] == unit_id) & (ov["clip_id"] == clip_id))]
        if action == "force":                    # one forced winner per unit
            ov = ov[~((ov["unit_id"] == unit_id) & (ov["action"] == "force"))]
        if action:
            if action not in ("accept", "reject", "force"):
                raise ValueError(f"unknown action {action!r}")
            ov = pd.concat([ov, pd.DataFrame([{"unit_id": unit_id, "clip_id": clip_id, "action": action,
                                               "note": note}])], ignore_index=True)
        rank.OVERRIDES.parent.mkdir(parents=True, exist_ok=True)
        ov.to_csv(rank.OVERRIDES, index=False)


def save_video_link(clip_id: str, link: str, note: str = "") -> None:
    """Hand-given YouTube video for a clip ('' removes it). Wins over automatic matches in Phase 6."""
    yid = manual.youtube_id(link) if link else ""
    if link and not yid:
        raise ValueError(f"not a YouTube link or 11-character ID: {link!r}")
    with LOCK:
        df = _csv(VIDEO_LINKS, VIDEO_COLUMNS)
        df = df[df["clip_id"] != clip_id]
        if yid:
            df = pd.concat([df, pd.DataFrame([{"clip_id": clip_id, "youtube_id": yid, "note": note}])],
                           ignore_index=True)
        VIDEO_LINKS.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(VIDEO_LINKS, index=False)


def save_overrides(changes: list[dict]) -> int:
    """Apply several changes at once (the page's «Guardar cambios» button): decisions and video links."""
    for c in changes:
        if c.get("kind") == "video":
            save_video_link(str(c["clip_id"]), str(c.get("youtube_id", "")), str(c.get("note", "")))
        else:
            save_override(str(c["unit_id"]), str(c["clip_id"]), str(c.get("action", "")), str(c.get("note", "")))
    return len(changes)


def add_manual(row: dict) -> str | None:
    """Validate and append one hand-added video; returns an error message or None."""
    row = {c: str(row.get(c) or "").strip() for c in MANUAL_COLUMNS}
    _, err = manual.row_to_evidence(row)
    if err:
        return err
    with LOCK:
        df = _csv(MANUAL_CSV, MANUAL_COLUMNS)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        MANUAL_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(MANUAL_CSV, index=False)
    return None


def delete_manual(index: int) -> None:
    with LOCK:
        df = _csv(MANUAL_CSV, MANUAL_COLUMNS)
        if 0 <= index < len(df):
            df.drop(index=index).to_csv(MANUAL_CSV, index=False)


def rebuild(plot: bool) -> str:
    log = []
    steps = REBUILD_STEPS + ([["plot"]] if plot else [])
    for step in steps:
        cmd = [sys.executable, "-m", "vmap.cli", *step]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=config.ROOT)
        log.append(f"$ vmap {' '.join(step)}\n{r.stdout[-3000:]}{r.stderr[-2000:] if r.returncode else ''}")
        if r.returncode:
            log.append(f"STOPPED: step failed (exit {r.returncode})")
            break
    return "\n".join(log)


def snippet_around(text: str, mentions: list[str], width: int = 220) -> dict:
    """The part of the passage around the first place mention: {pre, hit, post} (hit = the mention)."""
    text = str(text or "")
    for m in mentions:
        i = text.find(m) if m else -1
        if i >= 0:
            a, b = max(0, i - width), min(len(text), i + len(m) + width)
            return {"pre": ("…" if a else "") + text[a:i], "hit": m,
                    "post": text[i + len(m):b] + ("…" if b < len(text) else "")}
    return {"pre": text[:2 * width] + ("…" if len(text) > 2 * width else ""), "hit": "", "post": ""}


def page_data() -> dict:
    p = config.PROCESSED
    winners = pd.read_csv(p / "winners.csv", dtype={"unit_id": str})
    pairs = pd.read_csv(p / "clips_by_unit.csv", dtype={"unit_id": str})
    ev = pd.read_csv(config.INTERIM / "location_evidence.csv", dtype={"unit_id": str}).drop_duplicates("evidence_id")
    ev = ev.set_index("evidence_id")
    # rejected pairs are not in clips_by_unit: add them back so a rejection can be undone
    locs = pd.read_csv(config.INTERIM / "clip_locations.csv", dtype={"unit_id": str})
    ov = _csv(rank.OVERRIDES, rank.OVERRIDE_COLUMNS)
    rejected = locs.merge(ov[ov["action"] == "reject"][["unit_id", "clip_id"]], on=["unit_id", "clip_id"])
    rows = []
    for r in pd.concat([pairs, rejected.assign(status="rejected")], ignore_index=True).itertuples(index=False):
        evs = []
        mentions = [m.strip() for m in str(getattr(r, "location_mention", "") or "").split(";") if m.strip()]
        for e in str(r.evidence_ids).split()[:4]:
            if e in ev.index:
                x = ev.loc[e]
                evs.append({"url": x["source_url"], "dom": x["domain"],
                            **snippet_around(x["evidence_text"], mentions)})
        g = lambda k: (None if pd.isna(getattr(r, k, None)) else getattr(r, k, None))  # noqa: E731
        rows.append({"u": r.unit_id, "c": r.clip_id, "a": g("artist_display") or g("artist"),
                     "s": g("song_display") or g("song_title"), "p": g("location_mention"),
                     "v": None if g("view_count") is None else int(g("view_count")), "url": g("youtube_url"),
                     "st": r.status, "m": g("mark") or "", "why": g("reasons") or "", "ev": evs})
    munis = pd.read_csv(config.INTERIM / "municipalities.csv", dtype=str)
    return {"units": [{"id": w.unit_id, "name": w.unit_name, "win": w.clip_id if w.has_data else None}
                      for w in winners.itertuples(index=False)],
            "rows": rows, "overrides": _csv(rank.OVERRIDES, rank.OVERRIDE_COLUMNS).to_dict("records"),
            "manual": _csv(MANUAL_CSV, MANUAL_COLUMNS).to_dict("records"),
            "videos": _csv(VIDEO_LINKS, VIDEO_COLUMNS).to_dict("records"),
            "places": sorted(set(munis["name"]))}


PAGE = r"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Editor videoclips</title>
<style>
:root{--line:#e3e3e3;--muted:#666;--ok:#1b6e2a;--bad:#a12a2a;--force:#7a4cc2;--prob:#9a6a00}
*{box-sizing:border-box} body{margin:0;font:14px/1.45 system-ui,sans-serif;color:#222;background:#fafafa}
header{display:flex;gap:4px;padding:8px 12px;background:#fff;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:2}
header button{border:1px solid var(--line);background:#f4f4f4;padding:6px 12px;border-radius:6px;cursor:pointer;font:inherit}
header button.on{background:#222;color:#fff}
header button.save:not(:disabled){background:var(--ok);color:#fff;border-color:var(--ok)} header button:disabled{opacity:.5;cursor:default}
.card.dirty{border-color:#e0a800;box-shadow:0 0 0 2px #fff3c4} .tab{display:none;height:calc(100vh - 50px)} .tab.on{display:flex}
#units{width:250px;overflow:auto;border-right:1px solid var(--line);background:#fff}
#units div{padding:6px 10px;cursor:pointer;border-bottom:1px solid #f0f0f0;font-size:13px}
#units div.sel{background:#eef3fb} #units .n{float:right;color:var(--muted);font-size:12px}
#list{flex:1;overflow:auto;padding:12px 16px} .card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:10px}
.card.win{border-color:#8bc48f} .st{font-size:12px;font-weight:600} .st.confirmed{color:var(--ok)} .st.probable{color:var(--prob)} .st.rejected{color:var(--bad)}
.why{font-size:12px;color:var(--prob)} .ev{font-size:12px;color:#444;margin-top:4px} .ev a{color:#2457a6}
.acts{display:flex;gap:6px;align-items:center;margin-top:8px;flex-wrap:wrap}
.acts button{border:1px solid var(--line);background:#fff;padding:3px 10px;border-radius:5px;cursor:pointer;font:inherit;font-size:13px}
.acts button.accept.on{background:var(--ok);color:#fff}.acts button.reject.on{background:var(--bad);color:#fff}.acts button.force.on{background:var(--force);color:#fff}
.acts input{flex:1;min-width:160px;padding:3px 6px;border:1px solid var(--line);border-radius:5px;font:inherit;font-size:13px}
.muted{color:var(--muted);font-size:12px} .song{font-weight:600}
form{max-width:640px;background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:16px}
form label{display:block;margin:8px 0 2px;font-weight:600;font-size:13px} form input,form textarea{width:100%;padding:6px;border:1px solid #ccc;border-radius:5px;font:inherit}
form .hint{font-weight:400;color:var(--muted);font-size:12px} .msg{margin:8px 0;font-weight:600} .msg.err{color:var(--bad)} .msg.ok{color:var(--ok)}
#manlist{flex:1;overflow:auto;padding:16px} table{border-collapse:collapse;width:100%;font-size:13px;background:#fff} td,th{border:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
pre{white-space:pre-wrap;background:#111;color:#ddd;padding:12px;border-radius:6px;font-size:12px;max-height:70vh;overflow:auto}
#tab-build{flex-direction:column;padding:16px;overflow:auto}
.filters{display:flex;gap:12px;font-size:13px;margin-bottom:10px}
</style></head><body>
<header><button data-tab="review" class="on">Revisar</button><button data-tab="add">Añadir videoclip</button>
<button data-tab="build">Reconstruir</button>
<span id="pend" class="muted" style="margin-left:auto;align-self:center"></span>
<button id="discard" disabled>Descartar</button><button id="save" class="save" disabled>Guardar cambios</button></header>

<section class="tab on" id="tab-review"><div id="units"></div><div id="list"></div></section>

<section class="tab" id="tab-add"><form id="addf">
<h3 style="margin-top:0">Añadir un videoclip que conoces</h3>
<label>Artista</label><input name="artist" required>
<label>Canción</label><input name="song_title" required>
<label>Lugar(es) de rodaje <span class="hint">— municipio, isla o lugar; varios separados por «;»</span></label>
<input name="location" list="places" required>
<label>Vídeo de YouTube <span class="hint">— enlace o ID del videoclip oficial</span></label><input name="youtube_id" placeholder="https://www.youtube.com/watch?v=...">
<label>Fuente <span class="hint">— artículo u otra página (opcional si es tu propia observación)</span></label><input name="source_url" placeholder="https://...">
<label>Frase de la fuente <span class="hint">— copia literal donde aparece el lugar (si hay fuente)</span></label><textarea name="snippet" rows="2"></textarea>
<label>Nota / observación propia <span class="hint">— p. ej. «se ven las calles de Madrid» (si no hay frase de fuente)</span></label><textarea name="note" rows="2"></textarea>
<label style="font-weight:400"><input type="checkbox" name="partial" style="width:auto"> Rodado solo en parte allí</label>
<div id="addmsg" class="msg"></div><button type="submit">Guardar</button>
<datalist id="places"></datalist></form>
<div id="manlist"></div></section>

<section class="tab" id="tab-build"><p>Vuelve a calcular todo con tus decisiones y los vídeos añadidos
(sin búsquedas nuevas): manual → extract → resolve → youtube → rank → review → explore.</p>
<label><input type="checkbox" id="withplot"> También el mapa estático (≈1 min más)</label>
<p><button id="go">Reconstruir</button> <span id="buildst" class="muted"></span></p><pre id="log"></pre></section>

<script>
let D = null, unit = null, onlyPending = false;
const pending = new Map();                 // "unit|clip" → {unit_id, clip_id, action, note}, not saved yet
const $ = s => document.querySelector(s);
function el(t, a, ...k){const e=document.createElement(t);for(const[x,v]of Object.entries(a||{}))if(v!=null)e.setAttribute(x,v);
 for(const c of k){if(c==null)continue;e.append(c instanceof Node?c:document.createTextNode(String(c)))}return e}
const safe = u => (typeof u==="string" && /^https?:\/\//.test(u)) ? u : null;
const link = (u,t,title) => safe(u) ? el("a",{href:safe(u),target:"_blank",rel:"noopener",title:title||null},t) : document.createTextNode(t);
const fmt = n => n==null ? "—" : n.toLocaleString("es-ES");
async function api(path, body){const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})});
 if(!r.ok) throw new Error(await r.text()); return r.json()}
async function load(){D = await (await fetch("/api/data")).json(); renderUnits(); renderList(); renderManual();
 const dl=$("#places"); dl.replaceChildren(...D.places.map(p=>el("option",{value:p})));}
const saved = (u,c) => D.overrides.find(o => o.unit_id===u && o.clip_id===c);
const savedVideo = c => (D.videos.find(v => v.clip_id === c) || {}).youtube_id || "";
const videoOf = c => pending.has("video|" + c) ? pending.get("video|" + c).youtube_id : savedVideo(c);
const ytId = s => { const m = (s||"").trim().match(/(?:v=|youtu\.be\/|embed\/|shorts\/|live\/)([A-Za-z0-9_-]{11})|^([A-Za-z0-9_-]{11})$/); return m ? (m[1]||m[2]) : null; };
function stageVideo(c, link){
  const id = link.trim() ? ytId(link) : "";
  if (id === null) { alert("No parece un enlace de YouTube: " + link); renderList(); return; }
  if (id === savedVideo(c)) pending.delete("video|" + c); else pending.set("video|" + c, {kind:"video", clip_id:c, youtube_id:id});
  updateBar(); renderList();
}
const decision = (u,c) => pending.get(u+"|"+c) || saved(u,c);
function stage(r, action, note){
  const s = saved(r.u, r.c), key = r.u + "|" + r.c;
  if (action === "force")                  // one forced winner per unit: un-force the others (pending)
    for (const o of D.rows.filter(x => x.u === r.u && x.c !== r.c)) { const d = decision(o.u, o.c);
      if (d && d.action === "force") pending.set(o.u+"|"+o.c, {unit_id:o.u, clip_id:o.c, action:"", note:""}); }
  if (s ? (s.action === action && (s.note||"") === note) : !action) pending.delete(key);   // back to saved state
  else pending.set(key, {unit_id:r.u, clip_id:r.c, action, note});
  updateBar(); renderList();
}
function updateBar(){ const n = pending.size; $("#pend").textContent = n ? n + " cambio(s) sin guardar" : "";
  $("#save").disabled = $("#discard").disabled = !n; }
$("#save").onclick = async () => { await api("/api/overrides", {changes:[...pending.values()]}); pending.clear(); updateBar(); await load(); };
$("#discard").onclick = () => { pending.clear(); updateBar(); renderList(); };
window.addEventListener("beforeunload", e => { if (pending.size) { e.preventDefault(); e.returnValue = ""; } });

function renderUnits(){
  const box = $("#units"); box.replaceChildren();
  box.append(el("div", {class:"muted", style:"cursor:default"}, el("label", {}, (() => {const cb=el("input",{type:"checkbox"});cb.checked=onlyPending;
    cb.onchange=()=>{onlyPending=cb.checked;renderList()};return cb})(), " solo no confirmados")));
  for (const u of D.units){
    const rows = D.rows.filter(r => r.u === u.id), pend = rows.filter(r => r.st === "probable").length;
    const d = el("div", {class: u.id===unit ? "sel" : ""}, u.id + " · " + u.name, el("span",{class:"n"}, rows.length ? `${rows.length} (${pend}?)` : ""));
    d.onclick = () => { unit = u.id; renderUnits(); renderList(); }; box.append(d);
  }
}
function renderList(){
  const box = $("#list"); box.replaceChildren();
  if (!unit){ box.append(el("p",{class:"muted"},"Elige una provincia o isla a la izquierda.")); return; }
  const u = D.units.find(x => x.id === unit);
  let rows = D.rows.filter(r => r.u === unit).sort((a,b)=>(b.v??-1)-(a.v??-1));
  if (onlyPending) rows = rows.filter(r => r.st === "probable");
  box.append(el("h2",{}, u.name));
  if (!rows.length) box.append(el("p",{class:"muted"},"No hay videoclips. Puedes añadir uno en «Añadir videoclip»."));
  for (const r of rows){
    const dec = decision(r.u, r.c), dirty = pending.has(r.u + "|" + r.c) || pending.has("video|" + r.c);
    const myVid = videoOf(r.c);
    const vin = el("input",{placeholder: r.url ? "otro vídeo (enlace de YouTube)" : "pegar enlace de YouTube del videoclip",
                            value: myVid ? "https://www.youtube.com/watch?v=" + myVid : ""});
    vin.onchange = () => stageVideo(r.c, vin.value);
    const vidRow = el("div",{class:"acts"}, el("span",{class:"muted"}, "🎬 Vídeo:"), vin,
      myVid ? el("span",{class:"muted"}, link("https://www.youtube.com/watch?v=" + myVid, "abrir")) : null);
    const note = el("input",{placeholder:"nota (opcional)", value: dec ? (dec.note || "") : ""});
    note.onchange = () => { const d = decision(r.u, r.c); if (d && d.action) stage(r, d.action, note.value); };
    const btn = (act, label) => { const b = el("button",{class: act + (dec && dec.action===act ? " on" : "")}, label);
      b.onclick = () => stage(r, (dec && dec.action===act) ? "" : act, note.value); return b; };
    const why = r.st !== "confirmed" && r.why ? el("div",{class:"why"}, "Dudas: " + r.why.split(" ").join(", ")) : null;
    const ev = el("div",{class:"ev"});
    for (const e of r.ev) ev.append(el("div",{}, link(e.url, e.dom), ": “" + e.pre, e.hit ? el("mark",{}, e.hit) : null, e.post + "”"));
    box.append(el("div",{class:"card" + (u.win===r.c ? " win" : "") + (dirty ? " dirty" : "")},
      el("div",{}, el("span",{class:"song"}, link(r.url, (r.a ? r.a + " – " : "") + (r.s || "?"))), " ", r.m,
         u.win===r.c ? " 🏆" : "", el("span",{class:"muted"}, " · " + fmt(r.v) + " · " + (r.p || ""))),
      el("div",{}, el("span",{class:"st " + r.st}, r.st), r.url ? null : el("span",{class:"muted"}," · sin vídeo oficial")),
      why, ev, vidRow,
      el("div",{class:"acts"}, btn("accept","✓ Aceptar"), btn("reject","✗ Rechazar"), btn("force","★ Forzar ganador"), note,
         dirty ? el("span",{class:"muted", style:"color:#b07d00"}, "sin guardar") : el("span",{class:"muted"}, r.c.slice(0,10)))));
  }
}
function renderManual(){
  const box = $("#manlist"); box.replaceChildren(el("h3",{},"Videoclips añadidos a mano (" + D.manual.length + ")"));
  if (!D.manual.length) return;
  const t = el("table",{}, el("tr",{}, ...["Artista","Canción","Lugar","Vídeo","Fuente / nota",""].map(h=>el("th",{},h))));
  D.manual.forEach((m,i) => { const del = el("button",{},"Borrar"); del.onclick = async () => { if (confirm("¿Borrar esta fila?")) { await api("/api/manual/delete",{index:i}); await load(); } };
    t.append(el("tr",{}, el("td",{},m.artist), el("td",{},m.song_title), el("td",{},m.location), el("td",{}, m.youtube_id ? link(m.youtube_id.startsWith("http")?m.youtube_id:"https://www.youtube.com/watch?v="+m.youtube_id, "vídeo") : ""),
      el("td",{}, m.source_url ? link(m.source_url, "fuente") : "", " ", m.snippet || m.note), el("td",{}, del))); });
  box.append(t);
}
$("#addf").onsubmit = async e => { e.preventDefault(); const f = new FormData(e.target), row = Object.fromEntries(f.entries());
  row.partial = f.get("partial") ? "yes" : ""; const msg = $("#addmsg");
  try { const r = await api("/api/manual", row);
    if (r.error){ msg.className="msg err"; msg.textContent = r.error; } else { msg.className="msg ok"; msg.textContent="Guardado. Pulsa «Reconstruir» para incluirlo."; e.target.reset(); await load(); }
  } catch(err){ msg.className="msg err"; msg.textContent = String(err); } };
$("#go").onclick = async () => { if (pending.size && !confirm("Hay cambios sin guardar; no se incluirán. ¿Reconstruir igualmente?")) return; const b=$("#go"); b.disabled=true; $("#buildst").textContent="trabajando… (1–3 min)"; $("#log").textContent="";
  try { const r = await api("/api/rebuild",{plot:$("#withplot").checked}); $("#log").textContent = r.log; $("#buildst").textContent="hecho"; await load(); }
  catch(err){ $("#log").textContent = String(err); $("#buildst").textContent="error"; } b.disabled=false; };
document.querySelectorAll("header button[data-tab]").forEach(b => b.onclick = () => {
  document.querySelectorAll("header button[data-tab]").forEach(x=>x.classList.toggle("on", x===b));
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("on", t.id==="tab-"+b.dataset.tab)); });
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "vmap-editor"

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode(), "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        if not self._host_ok():
            return self._send(403, b"forbidden", "text/plain")
        if self.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/api/data":
            return self._json(page_data())
        self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if not self._host_ok() or not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self._send(403, b"forbidden", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        try:
            if self.path == "/api/override":
                save_override(body["unit_id"], body["clip_id"], body.get("action", ""), body.get("note", ""))
                return self._json({"ok": True})
            if self.path == "/api/overrides":
                return self._json({"ok": True, "saved": save_overrides(body.get("changes", []))})
            if self.path == "/api/manual":
                err = add_manual(body)
                return self._json({"ok": err is None, "error": err})
            if self.path == "/api/manual/delete":
                delete_manual(int(body["index"]))
                return self._json({"ok": True})
            if self.path == "/api/rebuild":
                return self._json({"log": rebuild(bool(body.get("plot")))})
        except (KeyError, ValueError) as e:
            return self._json({"error": str(e)}, 400)
        self._send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args):
        pass


def serve(port: int = 8765) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Editor: http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
