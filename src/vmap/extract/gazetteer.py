"""Place-name gazetteer for the rule extractor: municipalities, provinces, communities, islands, landmarks.

Names are matched token by token on accent-free lowercase text, longest match first, and the
capitalised words of a name must be capitalised in the text ("la palma de la mano" is not La Palma).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd
import yaml
from unidecode import unidecode

from vmap import config

ADMIN_CSV = config.INTERIM / "admin_names.csv"
GAZ_YAML = config.CONFIG / "gazetteer.yaml"
IGN = config.RAW / "lineas_limite"
ADMIN_SHP = {
    "province": [IGN / "SHP_ETRS89/recintos_provinciales_inspire_peninbal_etrs89/"
                       "recintos_provinciales_inspire_peninbal_etrs89.shp",
                 IGN / "SHP_WGS84/recintos_provinciales_inspire_canarias_wgs84/"
                       "recintos_provinciales_inspire_canarias_wgs84.shp"],
    "region": [IGN / "SHP_ETRS89/recintos_autonomicas_inspire_peninbal_etrs89/"
                     "recintos_autonomicas_inspire_peninbal_etrs89.shp"],
}
ARTICLES = {"el", "la", "los", "las", "l", "a", "o", "es", "sa", "ses", "s", "the"}
TOKEN = re.compile(r"[^\W_]+")


def tokens(text: str) -> list[tuple[str, int, int]]:
    """(original token, start, end) for each word."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(text)]


def ntok(tok: str) -> str:
    """Accent-free lowercase, but ñ stays distinct from n (Cañada is not Canadá)."""
    return unidecode(tok.replace("ñ", "nn").replace("Ñ", "NN")).lower()


@dataclass(frozen=True)
class Entry:
    name: str                          # canonical form as written in the source list
    kind: str                          # municipality | province | region | island | landmark
    code: str                          # INE 5 / province 2 / community 2 / unit id / landmark name
    provinces: tuple[str, ...] = ()    # INE province codes it lies in (empty if unknown)
    population: int | None = None
    homonym: bool = False              # also a well-known place outside Spain

    def as_dict(self) -> dict:
        return {"kind": self.kind, "code": self.code, "name": self.name, "provinces": list(self.provinces)}


@dataclass
class Gazetteer:
    index: dict[tuple[str, ...], list[Entry]] = field(default_factory=dict)
    stop: set[tuple[str, ...]] = field(default_factory=set)
    caps: dict[tuple[str, ...], tuple[bool, ...]] = field(default_factory=dict)
    spain_cues: re.Pattern | None = None
    foreign_cues: re.Pattern | None = None
    maxlen: int = 1

    def add(self, name: str, entry: Entry) -> None:
        toks = tokens(name)
        if not toks:
            return
        key = tuple(ntok(t) for t, _, _ in toks)
        bucket = self.index.setdefault(key, [])
        if entry not in bucket:
            bucket.append(entry)
        # capitalised words of the name (articles excluded) must be capitalised in the text
        self.caps.setdefault(key, tuple(t[0].isupper() and ntok(t) not in ARTICLES for t, _, _ in toks))
        self.maxlen = max(self.maxlen, len(key))
        # "L'Hospitalet de Llobregat" is often written "Hospitalet de Llobregat": add the form without the
        # leading article when at least two words remain (not "Santos" for "Los Santos")
        if len(key) >= 3 and key[0] in ARTICLES and entry.kind == "municipality":
            rest = name[toks[1][1]:]
            if rest != name:
                self.add(rest, entry)

    def find(self, text: str) -> list[dict]:
        """Non-overlapping place mentions, longest first: [{mention, start, end, entries, n_tokens}]."""
        toks = tokens(text)
        norm = [ntok(t) for t, _, _ in toks]
        out, i = [], 0
        while i < len(toks):
            hit = None
            for n in range(min(self.maxlen, len(toks) - i), 0, -1):
                key = tuple(norm[i:i + n])
                if key in self.index and self._caps_ok(key, toks[i:i + n]):
                    hit = (n, key)
                    break
            if not hit:
                i += 1
                continue
            n, key = hit
            start, end = toks[i][1], toks[i + n - 1][2]
            glued = (end < len(text) - 1 and text[end] in "-‐" and text[end + 1].isalpha()) or \
                    (start > 1 and text[start - 1] in "-‐" and text[start - 2].isalpha())
            if key not in self.stop and not glued:      # "Cee-Lo" is not Cee
                out.append({"mention": text[start:end], "start": start, "end": end,
                            "entries": self.index[key], "n_tokens": n})
            i += n
        return out

    def _caps_ok(self, key, toks) -> bool:
        need = self.caps[key]
        # a lowercase article is only accepted in long names ("el Puerto de Santa María"), not "la Torre"
        if not toks[0][0][0].isupper() and (ntok(toks[0][0]) not in ARTICLES or len(key) < 3):
            return False
        return all(t[0][0].isupper() for t, c in zip(toks, need) if c)


def _cue_regex(words: list[str]) -> re.Pattern:
    alts = sorted({re.escape(w) for w in words}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)")


def admin_names() -> pd.DataFrame:
    """Province and community names from the IGN files (cached as a small CSV)."""
    if ADMIN_CSV.exists():
        return pd.read_csv(ADMIN_CSV, dtype=str)
    import geopandas as gpd

    rows = []
    for kind, paths in ADMIN_SHP.items():
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"IGN file not found: {p}")
            df = gpd.read_file(p, ignore_geometry=True)
            for natcode, name in zip(df["NATCODE"], df["NAMEUNIT"]):
                code = natcode[4:6] if kind == "province" else natcode[2:4]
                if code != "00" and not name.startswith("Territorios"):
                    rows.append({"kind": kind, "code": code, "region": natcode[2:4], "name": name})
    out = pd.DataFrame(rows).drop_duplicates()
    ADMIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ADMIN_CSV, index=False)
    return out


def build(places: pd.DataFrame, admin: pd.DataFrame, cfg: dict) -> Gazetteer:
    g = Gazetteer()
    homonyms = {tuple(ntok(t) for t, _, _ in tokens(n)) for n in cfg.get("foreign_homonyms", [])}

    def hom(name: str) -> bool:
        return tuple(ntok(t) for t, _, _ in tokens(name)) in homonyms

    muni_alias = {str(k): v for k, v in (cfg.get("municipality_aliases") or {}).items()}
    for r in places.itertuples(index=False):
        names = str(r.variants).split("|") + muni_alias.get(str(r.ine), [])
        pop = None if pd.isna(r.population) else int(r.population)
        for n in names:
            g.add(n, Entry(n, "municipality", str(r.ine), (str(r.province),), pop, hom(n)))

    prov_alias = {str(k): v for k, v in (cfg.get("province_aliases") or {}).items()}
    reg_alias = {str(k): v for k, v in (cfg.get("region_aliases") or {}).items()}
    provs = admin[admin["kind"] == "province"]
    reg_provs = provs.groupby("region")["code"].apply(lambda s: tuple(sorted(s))).to_dict()
    for r in provs.itertuples(index=False):
        for n in [p.strip() for p in r.name.split("/")] + prov_alias.get(r.code, []):
            g.add(n, Entry(n, "province", r.code, (r.code,), None, hom(n)))
    for r in admin[admin["kind"] == "region"].itertuples(index=False):
        for n in [p.strip() for p in r.name.split("/")] + reg_alias.get(r.code, []):
            g.add(n, Entry(n, "region", r.code, reg_provs.get(r.code, ()), None, hom(n)))
    for unit, names in (cfg.get("islands") or {}).items():
        for n in names:
            g.add(n, Entry(n, "island", unit, (unit[:2],), None, hom(n)))
    for n in cfg.get("landmarks", []):
        g.add(n, Entry(n, "landmark", n, (), None, hom(n)))

    g.stop = {tuple(ntok(t) for t, _, _ in tokens(n)) for n in cfg.get("stop_names", [])}
    g.spain_cues = _cue_regex(cfg.get("spain_cues", []))
    g.foreign_cues = _cue_regex(cfg.get("foreign_cues", []))
    return g


@lru_cache(maxsize=1)
def load() -> Gazetteer:
    from vmap import places as pl

    with open(GAZ_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return build(pl.load(), admin_names(), cfg)
