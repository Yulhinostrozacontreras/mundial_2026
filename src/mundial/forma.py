"""Sugerencia Estadistica: marcador aproximado por FORMA RECIENTE (no es un modelo).

A diferencia del Elo/Poisson (que miran toda la historia ponderada), aqui se usan
solo los ULTIMOS N partidos OFICIALES de cada equipo (sin amistosos), ponderados por
recencia y por la calidad del rival (Elo), y se modulan por el valor de plantel (peso
de las estrellas) y la localia de anfitrion. Captura el momento actual del equipo.
"""
from pathlib import Path

import numpy as np
import polars as pl

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"

N_REC = 10            # ventana de forma reciente (partidos oficiales)
DECAY = 0.85          # peso por recencia (el mas reciente pesa mas)
K_VAL = 0.18          # sensibilidad al valor de plantel (peso de las estrellas)
K_RIVAL = 0.45        # ajuste por calidad del rival (Elo)
LOCALIA_STAT = 1.12   # multiplicador de goles del anfitrion jugando en casa
GOL_MAX = 4.0
ANFITRIONES = {"United States", "Mexico", "Canada"}

_CACHE: dict = {}


def _cargar() -> dict:
    """Carga (una vez) el historial oficial por equipo, Elo de rivales y valores."""
    if _CACHE:
        return _CACHE
    elo = pl.read_parquet(PROC / "elo_ratings.parquet")
    elo_map = dict(zip(elo["equipo"], elo["elo"]))
    elo_medio = float(np.mean(list(elo_map.values())))
    vp = pl.read_csv(PROC.parent / "valor_plantel_2026.csv")
    vmap = dict(zip(vp["equipo"], vp["valor_mln"]))
    media_lv = float(np.mean([np.log(v) for v in vmap.values()]))

    ofi = (pl.read_parquet(PROC / "partidos.parquet")
           .filter(pl.col("home_score").is_not_null() & (pl.col("tournament") != "Friendly")))
    hist: dict = {}
    for d, h, a, hs, as_ in ofi.select(
            "date", "home_team", "away_team", "home_score", "away_score").iter_rows():
        hist.setdefault(h, []).append((d, int(hs), int(as_), elo_map.get(a, elo_medio)))
        hist.setdefault(a, []).append((d, int(as_), int(hs), elo_map.get(h, elo_medio)))
    for e in hist:
        hist[e].sort(key=lambda x: x[0])

    _CACHE.update(hist=hist, elo_medio=elo_medio, vmap=vmap, media_lv=media_lv)
    return _CACHE


def _indices(eq: str, antes, c: dict):
    """(ataque, defensa, n) reciente: goles marcados/recibidos ajustados por rival."""
    ps = [p for p in c["hist"].get(eq, []) if antes is None or p[0] < antes][-N_REC:]
    if not ps:
        return 1.3, 1.3, 0  # fallback neutro (~promedio de goles por equipo)
    ps = ps[::-1]  # mas reciente primero
    w = np.array([DECAY ** k for k in range(len(ps))])
    w /= w.sum()
    gf = np.array([p[1] for p in ps], dtype=float)
    gc = np.array([p[2] for p in ps], dtype=float)
    fac = (np.array([p[3] for p in ps], dtype=float) - c["elo_medio"]) / 400.0
    atk = float(np.sum(w * gf * (1.0 + K_RIVAL * fac)))  # marcar a rival fuerte vale mas
    dfn = float(np.sum(w * gc * (1.0 - K_RIVAL * fac)))  # recibir de rival debil pesa mas
    return max(atk, 0.1), max(dfn, 0.1), len(ps)


def _logv(eq: str, c: dict) -> float:
    return float(np.log(c["vmap"].get(eq, np.exp(c["media_lv"]))))


def sugerencia(home: str, away: str, fecha=None, country=None):
    """Marcador sugerido por forma reciente: (gol_home, gol_away, n_home, n_away).

    fecha: si se pasa, solo usa partidos anteriores a esa fecha (forma pre-partido).
    country: pais sede; activa la localia si el local/visita es anfitrion en casa.
    """
    c = _cargar()
    of_h, df_h, nh = _indices(home, fecha, c)
    of_a, df_a, na = _indices(away, fecha, c)
    gh = 0.5 * (of_h + df_a)  # ataque propio + debilidad defensiva del rival
    ga = 0.5 * (of_a + df_h)
    dv = _logv(home, c) - _logv(away, c)  # diferencia de valor de plantel
    gh *= np.exp(K_VAL * dv / 2.0)
    ga *= np.exp(-K_VAL * dv / 2.0)
    if country == home and home in ANFITRIONES:
        gh *= LOCALIA_STAT
    if country == away and away in ANFITRIONES:
        ga *= LOCALIA_STAT
    return min(gh, GOL_MAX), min(ga, GOL_MAX), nh, na
