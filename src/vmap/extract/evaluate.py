"""Gold set for the extractor: sampling passages to annotate, and P/R/F1 against the annotations.

Gold file: tests/gold/extraction_gold.jsonl, one passage per line. The annotator fills:
  gold_places   list of Spanish places where the music video was filmed, written as in the text
                (evidence text or page title); [] if the passage names none
  gold_partial  true if only part of the video was filmed there or several places are named
  gold_song, gold_artist   as in the text, or null if not stated
Lines whose gold_places is still null are skipped by the evaluation.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from vmap import config
from vmap.extract import gazetteer as gz
from vmap.extract import rules
from vmap.extract.schema import Evidence
from vmap.resolve import clips

GOLD = config.ROOT / "tests" / "gold" / "extraction_gold.jsonl"
TODO = config.ROOT / "tests" / "gold" / "extraction_gold_todo.jsonl"


def _has_trigger(ev: Evidence) -> bool:
    t = ev.evidence_text + " " + (ev.page_title or "")
    return bool(rules.VIDEO.search(t) and (rules.FILM.search(t) or rules.RECORD.search(t)))


def sample(evidence: list[Evidence], extracted: list[Evidence], n: int = 60, seed: int = 7) -> list[dict]:
    """Half passages where the rules found a place, half filming-like passages where they found none."""
    rng = random.Random(seed)
    found = {e.evidence_id for e in extracted}
    pool = [e for e in evidence if e.source_type in ("wikipedia", "press")]
    pos = [e for e in pool if e.evidence_id in found]
    neg = [e for e in pool if e.evidence_id not in found and _has_trigger(e)]
    picked = rng.sample(pos, min(n // 2, len(pos))) + rng.sample(neg, min(n - n // 2, len(neg)))
    rng.shuffle(picked)
    return [{"evidence_id": e.evidence_id, "source_type": e.source_type, "source_url": e.source_url,
             "page_title": e.page_title, "text": e.evidence_text,
             "gold_places": None, "gold_partial": None, "gold_song": None, "gold_artist": None, "notes": ""}
            for e in picked]


def write_sample(rows: list[dict], path: Path = TODO) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _key(s: str) -> str:
    return clips.norm(s)


def _same(a: str, b: str) -> bool:
    a, b = _key(a), _key(b)
    return a == b or (len(a) > 3 and a in b) or (len(b) > 3 and b in a)


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn}


def _provinces(names: list[str], g: gz.Gazetteer) -> set[str]:
    out = set()
    for n in names:
        for h in g.find(n):
            ents = h["entries"]
            if len({c for e in ents for c in e.provinces}) == 1:
                out |= {c for e in ents for c in e.provinces}
    return out


def evaluate(gold: list[dict], predict, g: gz.Gazetteer, min_conf: float = 0.5) -> dict:
    """`predict(evidence_id) -> Evidence | None`. Places count when confidence >= min_conf."""
    pl = {"tp": 0, "fp": 0, "fn": 0}
    pv = {"tp": 0, "fp": 0, "fn": 0}
    part = song = artist = n_part = n_song = n_artist = 0
    errors = []
    rows = [r for r in gold if r.get("gold_places") is not None]
    for r in rows:
        pred = predict(r["evidence_id"])
        pp = [p for p in (pred.places if pred else []) if (p.get("confidence") or 0) >= min_conf]
        pred_names = [p["mention"] for p in pp]
        gold_names = r["gold_places"]
        matched = set()
        for gname in gold_names:
            hit = next((i for i, pn in enumerate(pred_names) if i not in matched and _same(gname, pn)), None)
            if hit is None:
                pl["fn"] += 1
            else:
                matched.add(hit)
                pl["tp"] += 1
        pl["fp"] += len(pred_names) - len(matched)
        gp = _provinces(gold_names, g)
        prp = set()
        for p in pp:          # province level = what could reach the map: one province, not ambiguous
            provs = {c for x in p["candidates"] for c in x["provinces"]}
            if len(provs) == 1 and not p.get("ambiguous"):
                prp |= provs
        pv["tp"] += len(gp & prp)
        pv["fp"] += len(prp - gp)
        pv["fn"] += len(gp - prp)
        if set(map(_key, gold_names)) != set(map(_key, pred_names)):
            errors.append({"evidence_id": r["evidence_id"], "gold": gold_names, "pred": pred_names})
        if r.get("gold_partial") is not None and gold_names:
            n_part += 1
            part += bool(pred and pred.partial) == bool(r["gold_partial"])
        if r.get("gold_song"):
            n_song += 1
            song += bool(pred and pred.song_title and clips.norm_title(pred.song_title)
                         == clips.norm_title(r["gold_song"]))
        if r.get("gold_artist"):
            n_artist += 1
            artist += bool(pred and clips.artist_match(pred.artist, r["gold_artist"]))
    return {"n": len(rows), "places": _prf(**pl), "provinces": _prf(**pv),
            "partial_accuracy": round(part / n_part, 3) if n_part else None,
            "song_accuracy": round(song / n_song, 3) if n_song else None,
            "artist_accuracy": round(artist / n_artist, 3) if n_artist else None,
            "errors": errors}


def load_gold(path: Path = GOLD) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- Excel round trip for annotation
XLSX = config.ROOT / "tests" / "gold" / "extraction_gold.xlsx"
NONE_MARK = "NONE"          # typed in gold_places when the passage names no Spanish filming place
COLUMNS = ["n", "source_type", "page_title", "text", "gold_places", "gold_partial", "gold_song",
           "gold_artist", "notes", "source_url", "evidence_id"]


def to_excel(rows: list[dict], path: Path = XLSX) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "gold"
    ws.append(COLUMNS)
    for i, r in enumerate(rows, 1):
        gp = r.get("gold_places")
        places = None if gp is None else ("; ".join(gp) if gp else NONE_MARK)
        partial = None if r.get("gold_partial") is None else ("yes" if r["gold_partial"] else "no")
        ws.append([i, r["source_type"], r.get("page_title"), r["text"], places, partial, r.get("gold_song"),
                   r.get("gold_artist"), r.get("notes") or None, r["source_url"], r["evidence_id"]])
    widths = {"n": 5, "source_type": 11, "page_title": 35, "text": 90, "gold_places": 28, "gold_partial": 12,
              "gold_song": 24, "gold_artist": 24, "notes": 30, "source_url": 40, "evidence_id": 14}
    fill = PatternFill("solid", fgColor="FFF2CC")
    for j, col in enumerate(COLUMNS, 1):
        letter = ws.cell(1, j).column_letter
        ws.column_dimensions[letter].width = widths[col]
        ws.cell(1, j).font = Font(bold=True)
        for row in ws.iter_rows(min_row=2, min_col=j, max_col=j):
            row[0].alignment = Alignment(wrap_text=True, vertical="top")
            if col.startswith("gold_"):
                row[0].fill = fill
    dv = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"F2:F{len(rows) + 1}")
    ws.freeze_panes = "E2"

    help_ws = wb.create_sheet("how to annotate")
    for line in [
        "Fill the yellow columns of the 'gold' sheet. Leave a row's gold_places empty to skip it.",
        "gold_places: Spanish places where the music video was filmed, written as in the text or page title, "
        "separated by ';'. Type NONE if the passage names no Spanish filming place.",
        "gold_partial: yes if only part of the video was filmed there or several places are named; else no.",
        "gold_song / gold_artist: as stated in the text or title; leave empty if not stated.",
        "Then run: vmap gold-import  (writes tests/gold/extraction_gold.jsonl) and vmap eval-extract.",
    ]:
        help_ws.append([line])
    help_ws.column_dimensions["A"].width = 140
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def from_excel(path: Path = XLSX) -> list[dict]:
    from openpyxl import load_workbook

    ws = load_workbook(path, read_only=True)["gold"]
    it = ws.iter_rows(values_only=True)
    header = [str(h) for h in next(it)]
    out = []
    for vals in it:
        r = dict(zip(header, vals))
        if not r.get("evidence_id"):
            continue

        def text(v):
            return str(v).strip() if v is not None and str(v).strip() else None

        gp = text(r.get("gold_places"))
        places = None if gp is None else ([] if gp.upper() == NONE_MARK else
                                          [p.strip() for p in gp.split(";") if p.strip()])
        partial = text(r.get("gold_partial"))
        out.append({"evidence_id": r["evidence_id"], "source_type": r.get("source_type"),
                    "source_url": r.get("source_url"), "page_title": r.get("page_title"), "text": r.get("text"),
                    "gold_places": places,
                    "gold_partial": None if partial is None else partial.lower() in ("yes", "y", "true", "1", "sí", "si"),
                    "gold_song": text(r.get("gold_song")), "gold_artist": text(r.get("gold_artist")),
                    "notes": text(r.get("notes")) or ""})
    return out
