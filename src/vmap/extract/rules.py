"""Rule-based extraction: filming verb + place name in the same sentence → place mentions.

For each evidence passage (and, for press, the page title):
1. Split into sentences; keep sentences with a filming trigger (rodado/grabado/filmed/shot/...).
   "grabar/record" triggers need a music-video word in the same sentence (otherwise it is usually the
   song being recorded in a studio). Other triggers accept a music-video word in the sentence, the
   passage, or a music-video section heading.
2. Find place names with the gazetteer, near a trigger (up to 150 chars after it, or 60 chars before it),
   with no "set in / born in / concert / tour ..." cue in between.
3. Drop single-word names that look like part of a person's name ("Pablo Alborán", "Andrés Iniesta").
4. Flag ambiguity: names shared by several Spanish municipalities, and names that are also well-known
   places abroad (Valencia, Córdoba, Santiago...) when nothing in the passage points to Spain.
5. Capitalised phrases after "en/in/at" that the gazetteer does not know are kept as `unresolved`
   (resolved in Phase 5) when the passage has a Spanish context.
Each mention is a literal substring of the field it came from (evidence_text or page_title).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from vmap import config
from vmap.extract import gazetteer as gz
from vmap.extract.schema import Evidence

VERSION = "rules:v1"
EXTRACTED_JSONL = config.INTERIM / "extracted.jsonl"

VIDEO = re.compile(r"(?i)\b(video\s*clip|videoclip|v[íi]deo|clip|music\s+video|vid[eé]o\s*musical)")
VIDEO_HEADING = re.compile(r"(?i)video|v[íi]deo|clip")
FILM = re.compile(
    r"(?i)\b(rodad[oa]s?|rod[óo]|rodaron|rodaje|rodajes|rodar|rodar[áa]n?|rueda|ruedan|rodando|"
    r"filmad[oa]s?|film[óo]|filmaron|filmaci[óo]n|filmar|filmando|localizaci[óo]n(?:es)?|locaciones|"
    r"filmed|filming|shooting|location shoot|on location|"
    r"(?:was|were|been|being|is|are)\s+(?:\w+\s+)?shot|shot\s+(?:in|at|on))\b")
# "con un videoclip en Hospitalet de Llobregat": the video noun itself followed by en/in
VIDEO_IN = re.compile(r"(?i)\b(video\s*clip|videoclip|v[íi]deo\s*musical|music\s+video)\s+(?:en|in)\b")
RECORD = re.compile(
    r"(?i)\b(grabad[oa]s?|grab[óo]|grabaron|grabaci[óo]n|grabaciones|grabar|grabar[áa]n?|graba|graban|"
    r"grabando|recorded|recording)\b")
# cues that the place is not where the video was filmed
NEGATIVE = re.compile(
    r"(?i)\b(ambientad[oa]|set in|takes? place|inspirad[oa]|homenaje|tribute|dedicad[oa]|nacid[oa]|born|"
    r"natal|concierto|concert|gira|tour|festival|estudio|studio|premiere|estren[oóa]|presentad[oa]|"
    r"álbum|album|disco|actuaci[óo]n|performed|cantante \w+ de|coincidiendo|concurso|concurs|competici[óo]n)\b")
PARTIAL = re.compile(
    r"(?i)\b(parte|partes|algunas escenas|algunos planos|parcialmente|tambi[ée]n|adem[áa]s|entre otros|"
    r"partly|partially|some scenes|parts? of|also|as well as|among others|additional scenes)\b")
PREP = re.compile(r"(?i)\b(?:en|in|at|desde|por|entre|near|around|a|de|del|of|from)\s+"
                  r"(?:(?:el|la|los|las|the|l['’])\s*)?$")
FUNCTION_WORDS = {"en", "in", "at", "de", "del", "desde", "por", "entre", "and", "y", "e", "i", "a", "the",
                  "of", "from", "near", "cerca", "junto", "around", "to", "hacia", "para", "como", "with",
                  "con", "on", "el", "la", "los", "las", "un", "una", "su", "sus", "its", "their", "his",
                  "her", "this", "este", "esta", "that", "while", "mientras", "when", "cuando", "also"}
SENTENCE = re.compile(r"(?<=[.!?…])\s+(?=[\"“«¿¡(]?[A-ZÁÉÍÓÚÑ0-9])")
UNRESOLVED = re.compile(
    r"\b(?:en|in|at)\s+((?:(?:el|la|los|las|the)\s+)?[A-ZÁÉÍÓÚÑÜÇ][\w'’\-]+"
    r"(?:\s+(?:(?:de|del|de la|de los|de las|of|of the|y|i|d['’])\s+)?[A-ZÁÉÍÓÚÑÜÇ][\w'’\-]+){0,4})")
NOT_PLACES = re.compile(
    r"(?i)^(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"youtube|vevo|instagram|tiktok|mtv|netflix|spotify|twitter|facebook|internet|the|los|las|el|la)$")
DISAMBIG = re.compile(r"\s*\((?:[^()]*\b(?:song|canci[óo]n|single|sencillo)\b[^()]*)\)\s*$", re.I)
QUOTE = re.compile(r"[“«\"‘']([^”»\"’'\n]{2,60})[”»\"’']")
# a quote right after the word for the song or the video: videoclip de «X», el tema 'X', the song "X"
SONG_QUOTE = re.compile(
    r"(?i)\b(?:videoclip|v[íi]deo(?:\s*clip)?|canci[óo]n|tema|sencillo|single|song|track)\s+(?:de\s+|del\s+tema\s+|for\s+)?"
    r"(?:su\s+(?:nuev[oa]\s+)?(?:canci[óo]n|tema|sencillo|single)\s+)?[“«\"‘']([^”»\"’'\n]{2,60})[”»\"’']")
# press headlines: "Naiara graba su nuevo videoclip...", "La cantante Soraya estrena..."
HEADLINE_ARTIST = re.compile(
    r"^(?:[^|:]*[|:]\s*)?(?i:(?:el|la|los|las)\s+(?:cantante|grupo|banda|rapero|rapera|artista|cantautora?|"
    r"dúo|duo|músico|música|dj)\s+(?:[a-záéíóúñ]+\s+)?)?"
    r"([A-ZÁÉÍÓÚÑ0-9][\w'’.\-&]*(?:\s+(?:[A-ZÁÉÍÓÚÑ0-9][\w'’.\-&]*|de|del|y|&|x))*)\s+"
    r"(?:graba|grabará|grabó|rueda|rodará|rodó|estrena|lanza|presenta|elige|escoge|filma|publica)\b")
COUNTERFACTUAL = re.compile(
    r"(?i)\b(inicialmente|en un principio|iba a ser|iban a|pensad[oa] para|estaba previsto|se cancel|"
    r"originally|initially|was (?:supposed|planned|scheduled)|were (?:supposed|planned|scheduled)|cancel+ed)\b")
AFTER_MAX, BEFORE_MAX = 150, 60
SMALL_TOWN = 50_000     # Wikipedia mentions of smaller towns need Spanish context (many share names abroad)


def sentences(text: str) -> list[tuple[str, int]]:
    out, pos = [], 0
    for part in SENTENCE.split(text):
        i = text.find(part, pos)
        out.append((part, i))
        pos = i + len(part)
    return out


def _triggers(sent: str, sent_video: bool, passage_video: bool) -> list[tuple[int, int, str]]:
    out = []
    if sent_video or passage_video:
        out += [(m.start(), m.end(), m.group()) for m in FILM.finditer(sent)]
    if sent_video:
        out += [(m.start(), m.end(), m.group()) for m in RECORD.finditer(sent)]
        out += [(m.start(), m.end(), m.group()) for m in VIDEO_IN.finditer(sent)]
    return out


def _nearest(start: int, end: int, trig: list[tuple[int, int, str]], sent: str):
    """Closest trigger within the window with no negative cue in between, else None."""
    best = None
    for ts, te, word in trig:
        if start >= te and start - te <= AFTER_MAX:
            gap, between = start - te, sent[te:start]
        elif ts >= end and ts - end <= BEFORE_MAX:
            gap, between = ts - end, sent[end:ts]
        else:
            continue
        if NEGATIVE.search(between):
            continue
        if best is None or gap < best[0]:
            best = (gap, word)
    return best


PLACE_NOUNS = {"catedral", "ciudad", "provincia", "plaza", "calle", "puerto", "playa", "castillo", "universidad",
               "ayuntamiento", "museo", "palacio", "teatro", "estadio", "parque", "reino", "montes", "sierra",
               "valle", "isla", "islas", "costa", "bahía", "río", "barrio", "basílica", "iglesia", "monasterio",
               "alcázar", "comarca", "campo", "vega", "tierra", "tierras", "montaña", "montañas", "puente",
               "estación", "aeropuerto", "centro", "casco", "paseo", "avenida", "torre", "muralla", "real", "mayor",
               "toros", "san", "santa", "santo"}


ORG_NOUNS = {"castellers", "colla", "banda", "coro", "orquesta", "club", "peña", "grupo", "asociación",
             "fundación", "festival", "fiestas", "feria", "carnaval", "hermandad", "cofradía", "escuela",
             "conservatorio", "orfeón", "agrupación", "unión", "deportivo", "atlético", "real", "amigos"}


def _person_like(sent: str, start: int, end: int, entries) -> bool:
    """Single capitalised word glued to another capitalised word: 'Pablo Alborán', 'Real Madrid',
    or a surname with 'de': 'Juan Ponce de León' (but not 'Catedral de León')."""
    # group or event named after the town: "los Castellers de Vilafranca", "Fiestas de Pamplona"
    if (m := re.search(r"([^\W\d_][\w'’]*)\s+(?:de|del|d['’])\s*$", sent[:start])) and \
            m.group(1).lower() in ORG_NOUNS:
        return True
    # first name + surname + de: "Juan Ponce de León" (not "centro Lourdes de Valladolid")
    if (m := re.search(r"([^\W\d_][\w'’]*)\s+([^\W\d_][\w'’]*)\s+(?:de|del|de la)\s$", sent[:start])):
        ws = [m.group(1), m.group(2)]
        if all(w[0].isupper() and w.lower() not in FUNCTION_WORDS and w.lower() not in PLACE_NOUNS for w in ws):
            return True
    prev = re.search(r"([^\W\d_][\w'’]*)\s$", sent[:start])
    nxt = re.match(r"\s([^\W\d_][\w'’]*)", sent[end:])
    big = any((e.population or 0) >= 100_000 for e in entries)
    if prev and prev.group(1)[0].isupper() and prev.group(1).lower() not in FUNCTION_WORDS:
        return True
    return bool(nxt and nxt.group(1)[0].isupper() and nxt.group(1).lower() not in FUNCTION_WORDS and not big)


QUOTED = re.compile(r"[“«\"]([^”»\"\n]{1,80})[”»\"]|(?<!\w)['‘]([^'’\n]{1,60})['’](?!\w)|#\w+")


def _quoted_spans(sent: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in QUOTED.finditer(sent)]


def _spanish(entries) -> bool:
    return any(e.kind in ("municipality", "province", "region", "island") and not e.homonym for e in entries)


def _candidates(entries) -> list[dict]:
    seen, out = set(), []
    for e in entries:
        if (e.kind, e.code) not in seen:
            seen.add((e.kind, e.code))
            out.append(e.as_dict())
    return out


def extract_field(text: str, field: str, g: gz.Gazetteer, passage_video: bool,
                  artist_names: frozenset[str] = frozenset()) -> tuple[list[dict], bool]:
    """Place mentions near filming triggers in one text field; second value = partial cue found."""
    found, partial = [], False
    for sent, off in sentences(text):
        sent_video = bool(VIDEO.search(sent))
        trig = _triggers(sent, sent_video, passage_video)
        if not trig:
            continue
        hits = []
        quoted = _quoted_spans(sent)
        for m in g.find(sent):
            entries = m["entries"]
            if any(qs <= m["start"] and m["end"] <= qe for qs, qe in quoted):
                continue            # inside a quoted title: «Cambia el Paso» is a song, not El Paso
            if m["n_tokens"] == 1 and all(e.kind in ("municipality", "province") for e in entries):
                if _person_like(sent, m["start"], m["end"], entries):
                    continue
            prep = bool(PREP.search(sent[max(0, m["start"] - 25):m["start"]]))
            if m["n_tokens"] == 1 and gz.ntok(m["mention"]) in artist_names and not prep:
                continue            # 'Arjona filmed the video...' is the singer, not the town
            near = _nearest(m["start"], m["end"], trig, sent)
            if not near:
                continue
            small = all(e.kind == "municipality" and (e.population or 0) < 2000 for e in entries)
            if m["n_tokens"] == 1 and small and not prep:
                continue
            if small and not re.search(r"(?i)\b(?:en|in|at)\s+(?:(?:el|la|los|las|the)\s+)?$",
                                       sent[max(0, m["start"] - 12):m["start"]]):
                continue            # tiny town needs "en X": "videoclip de Uña y Carne" is a band
            if re.match(r"\s+(?:de|del|de la|de los|de las|d['’])\s*[A-ZÁÉÍÓÚÑ]", sent[m["end"]:]) and \
                    all((e.population or 0) < 100_000 for e in entries if e.kind == "municipality") and \
                    any(e.kind == "municipality" for e in entries):
                continue            # start of a longer name: "Torre del Agua", "Concurs de Castells"
            if small and not any(m["mention"].casefold() == e.name.casefold() for e in entries):
                continue            # tiny town needs its exact spelling: "del Cabo" is not Cabó (Lleida)
            tail = sent[m["end"]:m["end"] + 40]
            hits.append({"mention": m["mention"], "field": field, "start": off + m["start"], "tail": tail,
                         "candidates": _candidates(entries), "_entries": entries,
                         "trigger": near[1], "trigger_gap": near[0], "prep": prep, "sent_video": sent_video})
        taken = [(h["start"] - off, h["start"] - off + len(h["mention"])) for h in hits]
        for m in UNRESOLVED.finditer(sent):
            s, e = m.start(1), m.end(1)
            phrase = m.group(1)
            if any(s < te and ts < e for ts, te in taken) or NOT_PLACES.match(phrase.split()[0]):
                continue
            if (g.spain_cues and g.spain_cues.fullmatch(phrase)) or \
                    (g.foreign_cues and g.foreign_cues.search(phrase)):
                continue            # a country or a known foreign place, not a Spanish location
            near = _nearest(s, e, trig, sent)
            if near:
                hits.append({"mention": phrase, "field": field, "start": off + s, "candidates": [],
                             "_entries": [], "trigger": near[1], "trigger_gap": near[0], "prep": True,
                             "sent_video": sent_video, "unresolved": True})
        if hits:
            if COUNTERFACTUAL.search(sent):
                for h in hits:
                    h["counterfactual"] = True
            found += hits
            partial |= bool(PARTIAL.search(sent))
    return found, partial


def _distinct_places(places: list[dict]) -> int:
    """Count places, not counting a province/community that contains another mentioned place."""
    keys, areas = set(), []
    for p in places:
        ents = p["_entries"]
        if not ents:
            keys.add(("u", p["mention"].casefold()))
        elif all(e.kind in ("province", "region") for e in ents):
            areas.append({c for e in ents for c in e.provinces})
        else:
            keys.add(tuple(sorted((e.kind, e.code) for e in ents)))
    covered = {c for p in places for e in p["_entries"] if e.kind not in ("province", "region")
               for c in e.provinces}
    return len(keys) + sum(1 for a in areas if not (a & covered))


def _confidence(p: dict) -> float:
    c = 0.5 + (0.2 if p["sent_video"] else 0) + (0.2 if p["prep"] else 0)
    if p.get("unresolved"):
        c = min(c, 0.4)
    if p.get("ambiguous"):
        c -= 0.3
    return round(max(0.05, min(0.95, c)), 2)


class ArtistIndex:
    """Known artist names (from the top-down candidates), matched case-sensitively as whole words."""

    def __init__(self, names):
        names = sorted({n.strip() for n in names if isinstance(n, str) and len(n.strip()) >= 4}, key=len,
                       reverse=True)
        self.names = frozenset(gz.ntok(n) for n in names)
        self.rx = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, names)) + r")(?!\w)") if names else None

    def find(self, *texts: str | None) -> str | None:
        if not self.rx:
            return None
        for t in texts:
            if t and (m := self.rx.search(t)):
                return m.group(1)
        return None


def find_song(text: str, title: str | None, places: list[dict]) -> str | None:
    """Quoted title next to 'videoclip/canción/tema/song', else a short capitalised quote in a video sentence."""
    names = {p["mention"] for p in places}
    for t in (text, title or ""):
        for q in SONG_QUOTE.findall(t):
            if q.strip() not in names:
                return q.strip()
    for t in [s for s, _ in sentences(text) if VIDEO.search(s)] + [title or ""]:
        for q in QUOTE.findall(t):
            q = q.strip()
            if q not in names and q[:1].isupper() | q[:1].isdigit() and len(q.split()) <= 6 \
                    and not q.endswith((".", ",")):
                return q
    return None


def headline_artist(text: str | None) -> str | None:
    if not text or not (m := HEADLINE_ARTIST.search(text.strip())):
        return None
    name = m.group(1).strip()
    first = name.split()[0].lower()
    return None if first in FUNCTION_WORDS or first in {"vídeo", "video", "galería", "fotos"} else name


def _domain(url: str) -> str:
    from urllib.parse import urlsplit

    return urlsplit(url).netloc.lower()


def extract(ev: Evidence, g: gz.Gazetteer, artists: ArtistIndex | None = None) -> Evidence | None:
    """Evidence copy with places, partial, song and artist filled; None when no filming place found."""
    if ev.source_type == "wikidata":
        places = [{"mention": m, "field": "wikidata", "candidates": [], "qid": q, "coords": ev.coords,
                   "confidence": 1.0} for m, q in zip(ev.location_mentions, ev.location_qids)]
        return ev.model_copy(update={"places": places}) if places else None

    title = ev.page_title if ev.source_type == "press" else None
    passage_video = bool(VIDEO.search(ev.evidence_text) or (ev.section and VIDEO_HEADING.search(ev.section))
                         or (title and VIDEO.search(title)))
    names = artists.names if artists else frozenset()
    places, partial = extract_field(ev.evidence_text, "evidence_text", g, passage_video, names)
    if title:
        tp, tpart = extract_field(title, "page_title", g, passage_video, names)
        places += [p for p in tp if p["mention"] not in {q["mention"] for q in places}]
        partial |= tpart
    if not places:
        return None

    full = ev.evidence_text + "\n" + (title or "")
    es_domain = _domain(ev.source_url).endswith(".es")
    spain = bool(g.spain_cues and g.spain_cues.search(full)) or es_domain

    def spain_near(p: dict) -> bool:
        """'España' within ~150 chars of the mention (not 'top 15 en ... España' far away)."""
        text = title if p.get("field") == "page_title" else ev.evidence_text
        s = p.get("start", 0)
        return es_domain or bool(g.spain_cues and g.spain_cues.search(text or "", max(0, s - 150), s + 150))
    foreign = bool(g.foreign_cues and g.foreign_cues.search(full))
    areas = [h for h in g.find(full) if any(e.kind in ("province", "region", "island") for e in h["entries"])]
    for p in places:
        ents = p["_entries"]
        provs = {c for e in ents for c in e.provinces}
        tail = p.pop("tail", "")
        # "Medellín, Colombia" / "Valencia, Spain": the words right after the name decide
        next_foreign = bool(g.foreign_cues and re.match(r"\s*[,(]\s*", tail)
                            and g.foreign_cues.match(re.sub(r"^\s*[,(]\s*", "", tail)))
        next_spain = bool(g.spain_cues and re.match(r"\s*[,(]\s*", tail)
                          and g.spain_cues.match(re.sub(r"^\s*[,(]\s*", "", tail)))
        support = spain_near(p) or next_spain \
            or any(q is not p and q["_entries"] and not any(e.homonym for e in q["_entries"])
                   and _spanish(q["_entries"]) for q in places) \
            or any(h["mention"] != p["mention"] and provs & {c for e in h["entries"] for c in e.provinces}
                   for h in areas)
        reasons = []
        munis = {e.code for e in ents if e.kind == "municipality"}
        if len(munis) > 1 and not any(e.kind in ("province", "region", "island") for e in ents):
            reasons.append("several_municipalities")
        if next_foreign:
            reasons.append("foreign_place")
        elif ents and any(e.homonym for e in ents) and not support:
            reasons.append("foreign_context" if foreign else "foreign_homonym")
        small = ents and all(e.kind == "municipality" and (e.population or 0) < SMALL_TOWN for e in ents)
        if ev.source_type == "wikipedia" and small and not support and not reasons:
            reasons.append("small_town_unverified")   # Bello (Colombia), Santa Bárbara (California)...
        if p.pop("counterfactual", False):
            reasons.append("not_actual")
        p["ambiguous"] = reasons or None
        p["confidence"] = _confidence(p)
    # unresolved names only count next to a clear Spanish place or an explicit Spain mention
    clear = spain or any(p["_entries"] and not p["ambiguous"] for p in places)
    for p in places:
        if p.get("unresolved") and not clear:
            p["ambiguous"] = ["no_spanish_context"]
    places = [p for p in places if not (p.get("unresolved") and p["ambiguous"])
              and "foreign_place" not in (p["ambiguous"] or [])]
    # same place named twice ("Jerez de la Frontera" ... "Jerez"): keep the longest mention
    merged: dict = {}
    for p in places:
        key = tuple(sorted((c["kind"], c["code"]) for c in p["candidates"])) or ("u", p["mention"].casefold())
        if key not in merged or len(p["mention"]) > len(merged[key]["mention"]):
            merged[key] = p
    places = list(merged.values())
    if not places:
        return None

    partial = partial or _distinct_places(places) > 1
    song = DISAMBIG.sub("", ev.song_title) if ev.song_title else find_song(ev.evidence_text, title, places)
    artist = ev.artist or (artists.find(title, ev.evidence_text) if artists else None) \
        or headline_artist(title) or headline_artist(ev.evidence_text)
    for p in places:
        p.pop("_entries", None)
    return ev.model_copy(update={
        "places": places, "location_mentions": list(dict.fromkeys(p["mention"] for p in places)),
        "partial": partial, "song_title": song, "artist": artist, "extractor": VERSION,
        "confidence": max(p["confidence"] for p in places)})


def load_artists() -> ArtistIndex:
    import pandas as pd

    from vmap.sources import seed

    names = []
    if seed.TOP_CSV.exists():
        names += pd.read_csv(seed.TOP_CSV, usecols=["artist"])["artist"].dropna().tolist()
    return ArtistIndex(names)


def write(records: list[Evidence], path: Path = EXTRACTED_JSONL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json(exclude_none=True) + "\n")
    tmp.replace(path)
    return path


def load(path: Path = EXTRACTED_JSONL) -> list[Evidence]:
    with open(path, encoding="utf-8") as f:
        return [Evidence.model_validate(json.loads(line)) for line in f if line.strip()]
