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


if __name__ == "__main__":
    app()
