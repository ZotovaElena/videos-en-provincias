"""Normalisation of song titles and artist names, and song-title parsing from YouTube titles."""
from __future__ import annotations

import re

from rapidfuzz import fuzz
from unidecode import unidecode

_NOISE_BRACKET = re.compile(
    r"[\(\[](?:[^\)\]]*\b(?:official|oficial|video|v[ií]deo|videoclip|clip|hd|4k|remaster(?:ed)?|"
    r"music|lyric|audio|visuali[sz]er|explicit)\b[^\)\]]*)[\)\]]", re.I)
# "(feat. X)", "[con X]", "(with X)" anywhere; bare "ft./feat./featuring X" up to the next separator
_FEAT = re.compile(r"\s*[\(\[]\s*(?:ft\.?|feat\.?|featuring|con|with)\s+[^\)\]]*[\)\]]"
                   r"|\s+(?:ft\.|feat\.|featuring)\s+[^\-–—|\(\[]*", re.I)
_SEP = re.compile(r"\s+[-–—|]\s+|\s*\|\s*")


def norm(text: str | None) -> str:
    """casefold, strip accents and punctuation, collapse spaces."""
    if not isinstance(text, str) or not text:      # None / NaN from pandas
        return ""
    t = unidecode(text).casefold()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def norm_title(title: str | None) -> str:
    if not isinstance(title, str) or not title:
        return ""
    t = _NOISE_BRACKET.sub("", title)
    t = _FEAT.sub("", t)
    return norm(t)


def split_artists(artist: str | None) -> list[str]:
    if not isinstance(artist, str) or not artist:
        return []
    parts = re.split(r"\s*(?:,|&|\by\b|\band\b|\bx\b|\bft\.?|\bfeat\.?|\bcon\b)\s*", artist, flags=re.I)
    return sorted({norm(p) for p in parts if norm(p)})


def norm_artist(artist: str | None) -> str:
    """Canonical artist key: individual names normalised and sorted."""
    return " & ".join(split_artists(artist))


def artist_match(a: str | None, b: str | None, cutoff: int = 85) -> bool:
    """True if any individual artist of `a` matches one of `b` (fuzzy)."""
    xs, ys = split_artists(a), split_artists(b)
    return any(fuzz.ratio(x, y) >= cutoff or (len(x) > 3 and x in y) or (len(y) > 3 and y in x)
               for x in xs for y in ys)


def song_from_youtube_title(title: str, artist: str | None) -> str:
    """Best guess of the song name from 'Artist - Song (Official Video)' style titles."""
    t = _NOISE_BRACKET.sub("", title).strip()
    parts = [p.strip(" \"'“”«»") for p in _SEP.split(t) if p.strip()]
    if len(parts) > 1 and artist:
        non_artist = [p for p in parts if not artist_match(p, artist)
                      and fuzz.partial_ratio(norm(artist), norm(p)) < 90]
        if non_artist:
            return _FEAT.sub("", non_artist[0]).strip()
    return _FEAT.sub("", parts[0] if parts else t).strip()
