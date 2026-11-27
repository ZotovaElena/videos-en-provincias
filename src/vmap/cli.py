"""Command-line entry point: `vmap <stage>`."""
from __future__ import annotations

import typer

from vmap import config

app = typer.Typer(help="Most-viewed music video filmed in each Spanish province / island.",
                  no_args_is_help=True)


@app.callback()
def main() -> None:
    """Pipeline stages, run in order."""


@app.command()
def check() -> None:
    """Check environment: secrets present, folders exist (no network calls)."""
    ok = True
    for name in ("YOUTUBE_KEY", "CONTACT_EMAIL"):
        present = bool(config.secret(name, required=False))
        ok &= present
        typer.echo(f"{name:15s} {'set' if present else 'MISSING'}")
    for p in (config.RAW, config.CACHE, config.INTERIM, config.PROCESSED, config.MANUAL):
        p.mkdir(parents=True, exist_ok=True)
        typer.echo(f"{str(p.relative_to(config.ROOT)):15s} ok")
    raise typer.Exit(0 if ok else 1)


@app.command()
def units() -> None:
    """IGN boundaries → 61 units (provinces, Ceuta, Melilla, 12 islands) → data/interim/units.gpkg."""
    from vmap import units as un

    gdf = un.build()
    un.write(gdf)
    muni = un.build_municipalities(gdf)
    muni.to_csv(un.MUNI_CSV, index=False)
    typer.echo(f"{len(muni)} municipalities → {un.MUNI_CSV.relative_to(config.ROOT)}")
    typer.echo(f"{len(gdf)} units ({gdf['kind'].value_counts().to_dict()}), "
               f"all valid: {bool(gdf.is_valid.all())} → {un.UNITS_GPKG.relative_to(config.ROOT)}")
    isl = gdf[gdf["kind"] == "island"][["unit_id", "name", "area_km2"]]
    typer.echo(isl.to_string(index=False))


@app.command()
def wikidata(refresh: bool = typer.Option(False, help="Re-download instead of using the cache.")) -> None:
    """Music items with a filming location (P915) in Spain → data/interim/evidence/wikidata.jsonl."""
    from vmap.extract import schema
    from vmap.sources import wikidata as wd

    records = wd.run(refresh=refresh)
    path = schema.write("wikidata", records)
    items = {r.wikidata_qid for r in records}
    places = {q for r in records for q in r.location_qids}
    no_coords = sum(1 for r in records if not r.coords)
    typer.echo(f"{len(records)} evidence records, {len(items)} items, {len(places)} places "
               f"({no_coords} records without coordinates) → {path.relative_to(config.ROOT)}")


@app.command()
def wikipedia(refresh: bool = typer.Option(False, help="Re-run searches instead of using the cache.")) -> None:
    """es/en Wikipedia passages about music videos → data/interim/evidence/wikipedia.jsonl."""
    from collections import Counter

    from vmap.extract import schema
    from vmap.sources import wikipedia as wp

    records = wp.run(refresh=refresh, log=typer.echo)
    path = schema.write("wikipedia", records)
    pages = {(r.lang, r.page_title) for r in records}
    kinds = Counter(r.page_kind for r in records)
    in_sec = sum(1 for r in records if r.section != "(intro)")
    typer.echo(f"{len(records)} passages from {len(pages)} pages; by page kind {dict(kinds)}; "
               f"{sum(1 for r in records if r.youtube_ids)} with YouTube IDs → {path.relative_to(config.ROOT)}")
    typer.echo(f"({in_sec} passages outside the intro)")


@app.command()
def seed(lookup: bool = typer.Option(True, help="Search Wikipedia song articles for videos without evidence.")) -> None:
    """Top-down: official channels of seed artists → high-view music videos + evidence status."""
    from vmap.sources import seed as sd
    from vmap.youtube.api import QuotaExceeded

    try:
        res = sd.run(lookup=lookup, log=typer.echo)
    except QuotaExceeded as e:
        typer.echo(f"STOPPED: {e}")
        raise typer.Exit(2)
    typer.echo(f"videos scanned: {res['videos']}; excluded: {res['excluded']}")
    typer.echo(f"top videos: {res['top']} → {sd.TOP_CSV.relative_to(config.ROOT)}")
    typer.echo(f"evidence status: {res['status']}")
    typer.echo(f"needs location research: {res['research']} → {sd.RESEARCH_CSV.relative_to(config.ROOT)}")
    typer.echo(f"YouTube quota used today: {res['quota']}")


@app.command()
def manual() -> None:
    """Import data/manual/manual_evidence.csv → data/interim/evidence/manual.jsonl (validated)."""
    from vmap.extract import schema
    from vmap.sources import manual as mn

    records, errors = mn.load_rows()
    for e in errors:
        typer.echo(f"SKIPPED {e}")
    path = schema.write("manual", records)
    typer.echo(f"{len(records)} manual evidence records → {path.relative_to(config.ROOT)}")


@app.command()
def places() -> None:
    """Municipalities (IGN) + population and capitals (Wikidata) → data/interim/places.csv."""
    from vmap import places as pl

    df = pl.build()
    typer.echo(f"{len(df)} municipalities, {int(df['is_capital'].sum())} province capitals, "
               f"{int(df['population'].notna().sum())} with population → {pl.PLACES_CSV.relative_to(config.ROOT)}")


@app.command()
def press(max_queries: int = typer.Option(None, help="Real (non-cached) queries allowed in this run."),
          dry_run: bool = typer.Option(False, help="Only show the query plan; no API calls."),
          cached_only: bool = typer.Option(False, help="Use cached queries only; spend no search quota.")) -> None:
    """Press search 'videoclip rodado en {localidad}' → data/interim/evidence/press.jsonl."""
    from vmap.extract import schema
    from vmap.sources import press as pr

    res = pr.run(max_queries=max_queries, dry_run=dry_run, cached_only=cached_only, log=typer.echo)
    if dry_run:
        typer.echo(f"{res['planned']} queries planned; by tier (0 capitals, 1 towns, 2 rest): {res['by_tier']}")
        for q in res["first"]:
            typer.echo(f"  {q}")
        return
    path = schema.write("press", res.pop("records"))
    typer.echo(f"{res} → {path.relative_to(config.ROOT)}")


@app.command()
def extract() -> None:
    """Rule extractor over all evidence → data/interim/extracted.jsonl (place mentions near filming verbs)."""
    from collections import Counter

    from vmap.extract import gazetteer, rules, schema

    g, artists = gazetteer.load(), rules.load_artists()
    evidence = schema.load_all()
    out = [r for ev in evidence if (r := rules.extract(ev, g, artists))]
    path = rules.write(out)
    places = [p for r in out for p in r.places]
    by_src = Counter(r.source_type for r in out)
    kinds = Counter((p["candidates"][0]["kind"] if p["candidates"] else
                     ("wikidata" if p["field"] == "wikidata" else "unresolved")) for p in places)
    amb = Counter(a for p in places for a in (p.get("ambiguous") or []))
    typer.echo(f"{len(evidence)} evidence records → {len(out)} with a filming place {dict(by_src)}")
    typer.echo(f"{len(places)} place mentions by kind {dict(kinds)}; ambiguous {dict(amb)}")
    typer.echo(f"partial: {sum(1 for r in out if r.partial)}; with song: {sum(1 for r in out if r.song_title)}; "
               f"with artist: {sum(1 for r in out if r.artist)} → {path.relative_to(config.ROOT)}")


@app.command()
def resolve() -> None:
    """Phase 5: link records to clips, places to units → clip_locations.csv, clips.csv, review_queue.csv."""
    from vmap.resolve import pipeline

    res = pipeline.run(log=typer.echo)
    for k, v in res.items():
        typer.echo(f"{k:22s} {v}")


@app.command()
def youtube(max_queries: int = typer.Option(None, help="Serper queries allowed in this run (default: daily budget)."),
            search: bool = typer.Option(True, help="Fall back to YouTube search.list (its own daily budget)."),
            refresh: bool = typer.Option(True, help="Re-download statistics for one snapshot date.")) -> None:
    """Phase 6: YouTube video + fresh statistics for every clip with a place → youtube_videos.csv."""
    from vmap.youtube import ids

    res = ids.run(max_queries=max_queries, youtube_search=search, refresh=refresh, log=typer.echo)
    for k, v in res.items():
        typer.echo(f"{k:18s} {v}")
    typer.echo(f"→ {ids.VIDEOS_CSV.relative_to(config.ROOT)}, {ids.NOT_FOUND_CSV.relative_to(config.ROOT)}")


@app.command("rank")
def rank_cmd() -> None:
    """Phase 7: overrides + winners per unit → winners.csv, candidates/clips by unit and by place."""
    from vmap import rank as rk

    for k, v in rk.run(log=typer.echo).items():
        typer.echo(f"{k:20s} {v}")


@app.command()
def review() -> None:
    """Phase 7: static review page → data/processed/review.html + review.csv."""
    from vmap import review as rv

    for k, v in rv.run().items():
        typer.echo(f"{k:8s} {v}")


@app.command()
def plot() -> None:
    """Phase 7: static map → data/processed/mapa.svg, mapa.png, results.geojson."""
    from vmap import plot_map

    for k, v in plot_map.run().items():
        typer.echo(f"{k:12s} {v}")


@app.command()
def explore() -> None:
    """Phase 7b: interactive explorer (all videos per unit and town) → data/processed/explorer.html."""
    from vmap import explore as ex

    for k, v in ex.run().items():
        typer.echo(f"{k:8s} {v}")


@app.command("gold-sample")
def gold_sample(n: int = typer.Option(60, help="Passages to sample."),
                force: bool = typer.Option(False, help="Overwrite an existing to-do file.")) -> None:
    """Sample passages to annotate → tests/gold/extraction_gold_todo.jsonl (rename to extraction_gold.jsonl)."""
    from vmap.extract import evaluate as ev
    from vmap.extract import rules, schema

    if ev.TODO.exists() and not force:
        typer.echo(f"{ev.TODO.relative_to(config.ROOT)} exists (may hold annotations); use --force to overwrite.")
        raise typer.Exit(1)
    rows = ev.sample(schema.load_all(), rules.load(), n)
    path = ev.write_sample(rows)
    typer.echo(f"{len(rows)} passages → {path.relative_to(config.ROOT)}")


@app.command("gold-export")
def gold_export() -> None:
    """Gold to-do JSONL → tests/gold/extraction_gold.xlsx for annotation in a spreadsheet."""
    from vmap.extract import evaluate as ev

    src = ev.GOLD if ev.GOLD.exists() else ev.TODO
    if ev.XLSX.exists():
        typer.echo(f"{ev.XLSX.relative_to(config.ROOT)} exists (may hold annotations); delete it first to re-export.")
        raise typer.Exit(1)
    path = ev.to_excel(ev.load_gold(src))
    typer.echo(f"{src.relative_to(config.ROOT)} → {path.relative_to(config.ROOT)}")


@app.command("gold-import")
def gold_import() -> None:
    """Annotated tests/gold/extraction_gold.xlsx → tests/gold/extraction_gold.jsonl."""
    from vmap.extract import evaluate as ev

    rows = ev.from_excel()
    path = ev.write_sample(rows, ev.GOLD)
    done = sum(1 for r in rows if r["gold_places"] is not None)
    typer.echo(f"{len(rows)} rows, {done} annotated → {path.relative_to(config.ROOT)}")


@app.command("eval-extract")
def eval_extract(min_conf: float = typer.Option(0.5, help="Confidence needed for a place to count."),
                 show_errors: bool = typer.Option(True, help="List passages where places differ.")) -> None:
    """Rule extractor vs tests/gold/extraction_gold.jsonl → precision / recall / F1."""
    from vmap.extract import evaluate as ev
    from vmap.extract import gazetteer, rules, schema

    if not ev.GOLD.exists():
        typer.echo(f"{ev.GOLD.relative_to(config.ROOT)} not found: run `vmap gold-sample`, annotate, rename.")
        raise typer.Exit(1)
    gold = ev.load_gold()
    g, artists = gazetteer.load(), rules.load_artists()
    wanted = {r["evidence_id"] for r in gold}
    by_id = {e.evidence_id: e for e in schema.load_all() if e.evidence_id in wanted}
    res = ev.evaluate(gold, lambda i: rules.extract(by_id[i], g, artists) if i in by_id else None, g, min_conf)
    errors = res.pop("errors")
    for k, v in res.items():
        typer.echo(f"{k:17s} {v}")
    if show_errors:
        for e in errors:
            typer.echo(f"  {e['evidence_id'][:10]} gold={e['gold']} pred={e['pred']}")


if __name__ == "__main__":
    app()
