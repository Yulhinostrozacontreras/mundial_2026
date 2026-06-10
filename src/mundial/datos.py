"""Carga de partidos y utilidades compartidas."""
from pathlib import Path

import polars as pl

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def cargar_partidos() -> pl.DataFrame:
    """Todos los partidos (incluye fixtures futuros con score nulo)."""
    return pl.read_parquet(PROC_DIR / "partidos.parquet")


def cargar_jugados() -> pl.DataFrame:
    """Solo partidos con resultado, ordenados por fecha."""
    return (
        cargar_partidos()
        .filter(pl.col("home_score").is_not_null())
        .sort("date")
    )


def peso_torneo(tournament: str) -> float:
    """Importancia del partido (estilo World Football Elo)."""
    t = tournament.lower()
    if "friendly" in t:
        return 20.0
    if "world cup" in t and "qualification" not in t:
        return 60.0
    if "qualification" in t:
        return 40.0
    finales = ("uefa euro", "copa am", "african cup", "asian cup",
               "gold cup", "nations league", "confederations")
    if any(f in t for f in finales):
        return 50.0
    return 30.0
