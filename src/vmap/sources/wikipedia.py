"""Wikipedia (es, en): passages about music videos → raw evidence records.

Candidate pages:
  1. es/en articles of the Wikidata items found by the wikidata stage (sitelinks);
  2. CirrusSearch phrase queries from config/search.yaml.
Pages are fetched as wikitext in batches of 50, converted to plain text, and split into sections.
Kept passages:
  - every paragraph of a section whose heading looks like "Videoclip / Vídeo musical / Music video";
  - elsewhere, paragraphs that mention a music video keyword.
Records have no location_mentions yet (extractor="none"); the extraction stage fills them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from vmap import config
from vmap.extract import schema
from vmap.extract.schema import Evidence, evidence_id
from vmap.http import Client, retrieved_at

WD_API = "https://www.wikidata.org/w/api.php"
BATCH = 50

SONG_CLASSES = {"Q7366", "Q134556", "Q105543609", "Q2188189", "Q193977"}
ARTIST_CLASSES = {"Q5", "Q215380", "Q5741069", "Q2088357", "Q9212979", "Q19849001"}
ALBUM_CLASSES = {"Q482994", "Q208569", "Q169930", "Q209939"}

YOUTUBE_RE = re.compile(r"(?:youtube\.com/(?:watch\?(?:[^\s|\]}]*&)?v=|embed/|shorts/)|youtu\.be/)"
                        r"([A-Za-z0-9_-]{11})")
YOUTUBE_TPL_RE = re.compile(r"\{\{\s*youtube\s*\|(?:\s*(?:id|v)\s*=)?\s*([A-Za-z0-9_-]{11})", re.I)
HEADING_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.M)
TITLE_SUFFIX_RE = re.compile(r"\s*\((?:canción|song|sencillo|single|tema)\b[^)]*\)\s*$", re.I)
FILE_PREFIXES = ("file:", "archivo:", "imagen:", "image:", "category:", "categoría:", "media:")


# ---------------------------------------------------------------- wikitext → text
def _innermost(pattern: str, repl, text: str) -> str:
    rx = re.compile(pattern, re.S)
    while True:
        new = rx.sub(repl, text)
        if new == text:
            return new
        text = new


def _link(m: re.Match) -> str:
    inner = m.group(1)
    if inner.lower().lstrip(": ").startswith(FILE_PREFIXES):
        return ""
    return inner.split("|")[-1]


def _template(m: re.Match) -> str:
    parts = m.group(1).split("|")
    name = parts[0].strip().lower()
    if name in ("lang", "lang-en", "nowrap", "nobr", "small") and len(parts) > 1:
        return parts[-1]
    return ""


def wikitext_to_text(wt: str) -> str:
    t = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<(gallery|timeline|math|score|syntaxhighlight)[^>]*>.*?</\1>", "", t, flags=re.S | re.I)
    t = _innermost(r"\{\{([^{}]*)\}\}", _template, t)
    t = _innermost(r"\{\|(?:(?!\{\|).)*?\|\}", "", t)          # tables
    t = _innermost(r"\[\[([^\[\]]*)\]\]", _link, t)
    t = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\]", "", t)
    t = re.sub(r"'{2,5}", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("&nbsp;", " ")
    lines = []
    for line in t.split("\n"):
        s = line.strip()
        if s.startswith(("|", "!", "{|", "|}", "__")):
            continue
        lines.append(re.sub(r"^[*#:;]+\s*", "", s))
    t = "\n".join(lines)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" ([,.;:])", r"\1", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


@dataclass
class Section:
    heading: str
    level: int
    raw: str
    text: str
    in_video_section: bool


def split_sections(wt: str, heading_re: re.Pattern) -> list[Section]:
    marks = [(m.start(), m.end(), len(m.group(1)), wikitext_to_text(m.group(2)))
             for m in HEADING_RE.finditer(wt)]
    bounds = [(0, 0, 1, "")] + marks
    out: list[Section] = []
    video_level = None   # level of the enclosing video heading, if any
    for i, (start, end, level, heading) in enumerate(bounds):
        body_end = bounds[i + 1][0] if i + 1 < len(bounds) else len(wt)
        if video_level is not None and level <= video_level:
            video_level = None
        if heading and heading_re.search(heading):
            video_level = level
        raw = wt[end:body_end]
        out.append(Section(heading, level, raw, wikitext_to_text(raw), video_level is not None))
    return out


def _window(par: str, kw_re: re.Pattern, max_chars: int) -> str:
    if len(par) <= max_chars:
        return par
    m = kw_re.search(par)
    start = max(0, (m.start() if m else 0) - max_chars // 3)
    start = par.rfind(". ", 0, start) + 2 if par.rfind(". ", 0, start) >= 0 else start
    chunk = par[start:start + max_chars]
    return chunk[: chunk.rfind(" ")] if len(chunk) == max_chars and " " in chunk else chunk


def passages(wt: str, cfg: dict) -> list[tuple[str, str, list[str]]]:
    """(section heading, passage text, youtube ids in that section)."""
    heading_re = re.compile(cfg["section_heading_regex"])
    kw_re = re.compile(cfg["video_keyword_regex"])
    out = []
    for sec in split_sections(wt, heading_re):
        yt = sorted(set(YOUTUBE_RE.findall(sec.raw)) | set(YOUTUBE_TPL_RE.findall(sec.raw)))
        for par in (p.strip() for p in sec.text.split("\n\n")):
            if len(par) < 30:
                continue
            if sec.in_video_section or kw_re.search(par):
                out.append((sec.heading or "(intro)", _window(par, kw_re, cfg["max_paragraph_chars"]), yt))
    return out


def clean_title(title: str) -> str:
    return TITLE_SUFFIX_RE.sub("", title).strip()


# ---------------------------------------------------------------- API calls
def _chunks(xs, n=BATCH):
    xs = list(xs)
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def search_titles(client: Client, lang: str, query: str, limit: int, refresh=False) -> list[int]:
    ids, offset = [], 0
    while offset < min(limit, 10000):
        d = client.get_json(f"https://{lang}.wikipedia.org/w/api.php", {
            "action": "query", "list": "search", "srsearch": query, "srnamespace": 0,
            "srlimit": 500, "sroffset": offset, "srprop": "", "format": "json", "formatversion": 2,
        }, refresh=refresh)
        hits = d["query"]["search"]
        ids += [h["pageid"] for h in hits]
        if "continue" not in d or not hits:
            break
        offset = d["continue"]["sroffset"]
    return ids[:limit]


def sitelink_titles(client: Client, qids: list[str], langs: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {lang: [] for lang in langs}
    for chunk in _chunks(sorted(set(qids))):
        d = client.get_json(WD_API, {"action": "wbgetentities", "ids": "|".join(chunk),
                                     "props": "sitelinks", "format": "json"})
        for ent in d.get("entities", {}).values():
            for lang in langs:
                if (sl := ent.get("sitelinks", {}).get(f"{lang}wiki")):
                    out[lang].append(sl["title"])
    return out


def fetch_pages(client: Client, lang: str, *, pageids=(), titles=()) -> list[dict]:
    """Latest wikitext + Wikidata item for pages, 50 per request, following API continuation."""
    pages = []
    for key, values in (("pageids", pageids), ("titles", titles)):
        for chunk in _chunks(values):
            params = {"action": "query", "prop": "revisions|pageprops", "rvprop": "content",
                      "rvslots": "main", "ppprop": "wikibase_item", "redirects": 1,
                      "format": "json", "formatversion": 2, key: "|".join(map(str, chunk))}
            cont: dict = {}
            while True:
                r = client.get(f"https://{lang}.wikipedia.org/w/api.php", {**params, **cont})
                d = r.json()
                for p in d.get("query", {}).get("pages", []):
                    if p.get("missing") or not p.get("revisions"):
                        continue
                    pages.append({"lang": lang, "title": p["title"],
                                  "qid": p.get("pageprops", {}).get("wikibase_item"),
                                  "wikitext": p["revisions"][0]["slots"]["main"]["content"],
                                  "retrieved_at": retrieved_at(r)})
                if "continue" not in d:
                    break
                cont = d["continue"]
    return pages


def item_info(client: Client, qids: list[str]) -> dict[str, dict]:
    """P31 classes and P175 performer labels for each item."""
    info, performer_qids = {}, set()
    for chunk in _chunks(sorted(set(qids))):
        d = client.get_json(WD_API, {"action": "wbgetentities", "ids": "|".join(chunk),
                                     "props": "claims", "format": "json"})
        for qid, ent in d.get("entities", {}).items():
            claims = ent.get("claims", {})
            def vals(p):
                return [c["mainsnak"]["datavalue"]["value"]["id"] for c in claims.get(p, [])
                        if c["mainsnak"].get("datavalue")]
            info[qid] = {"classes": set(vals("P31")), "performers": vals("P175")}
            performer_qids.update(info[qid]["performers"])
    labels = {}
    for chunk in _chunks(sorted(performer_qids)):
        d = client.get_json(WD_API, {"action": "wbgetentities", "ids": "|".join(chunk),
                                     "props": "labels", "languages": "es|mul|en", "format": "json"})
        for qid, ent in d.get("entities", {}).items():
            lab = ent.get("labels", {})
            labels[qid] = (lab.get("es") or lab.get("mul") or lab.get("en") or {}).get("value", qid)
    for it in info.values():
        it["performer_labels"] = [labels.get(q, q) for q in it["performers"]]
    return info


def page_kind(classes: set[str]) -> str:
    if classes & SONG_CLASSES:
        return "song"
    if classes & ARTIST_CLASSES:
        return "artist"
    if classes & ALBUM_CLASSES:
        return "album"
    return "other"


def page_to_evidence(page: dict, info: dict, cfg: dict) -> list[Evidence]:
    it = info.get(page["qid"] or "", {"classes": set(), "performer_labels": []})
    kind = page_kind(it["classes"])
    song = clean_title(page["title"]) if kind == "song" else None
    artist = (" & ".join(it["performer_labels"]) or None) if kind in ("song", "album") else (
        clean_title(page["title"]) if kind == "artist" else None)
    url = f"https://{page['lang']}.wikipedia.org/wiki/{page['title'].replace(' ', '_')}"
    out = []
    for heading, text, yt in passages(page["wikitext"], cfg):
        out.append(Evidence(
            evidence_id=evidence_id(url, text), source_type="wikipedia", source_url=url,
            source_name=f"Wikipedia ({page['lang']})", retrieved_at=page["retrieved_at"],
            lang=page["lang"], song_title=song, artist=artist, evidence_text=text,
            page_title=page["title"], page_kind=kind, section=heading,
            wikidata_qid=page["qid"], youtube_ids=yt,
        ))
    return out


def run(client: Client | None = None, refresh: bool = False, log=print) -> list[Evidence]:
    cfg = config.search()["wikipedia"]
    client = client or Client(stage="wikipedia")
    wd_qids = [e.wikidata_qid for e in schema.load("wikidata") if e.wikidata_qid]
    seeds = sitelink_titles(client, wd_qids, cfg["langs"]) if wd_qids else {}
    pages = []
    for lang in cfg["langs"]:
        ids: list[int] = []
        for q in cfg["queries"].get(lang, []):
            found = search_titles(client, lang, q, cfg["max_results_per_query"], refresh=refresh)
            log(f"  [{lang}] {len(found):5d} hits  {q}")
            ids += found
        ids = list(dict.fromkeys(ids))
        lang_pages = fetch_pages(client, lang, pageids=ids, titles=seeds.get(lang, []))
        log(f"  [{lang}] {len(lang_pages)} pages fetched ({len(ids)} from search, "
            f"{len(seeds.get(lang, []))} from Wikidata)")
        pages += lang_pages
    # same page can arrive from search and from sitelinks
    pages = list({(p["lang"], p["title"]): p for p in pages}.values())
    info = item_info(client, [p["qid"] for p in pages if p["qid"]])
    records = [e for p in pages for e in page_to_evidence(p, info, cfg)]
    return records
