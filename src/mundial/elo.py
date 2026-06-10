"""Modelo Elo ponderado para selecciones + calibracion ordinal de W/D/L."""
import numpy as np
import polars as pl
from scipy.optimize import minimize

from .datos import peso_torneo

VENTAJA_LOCAL = 100.0  # puntos Elo de localia (0 en campo neutral)


def correr_elo(df: pl.DataFrame, rating_inicial: float = 1500.0):
    """Recorre los partidos cronologicamente y devuelve (ratings, historial).

    historial: lista de (dr_efectivo, resultado) por partido, util para calibrar.
    resultado: 0=gana local, 1=empate, 2=gana visitante.
    """
    ratings: dict[str, float] = {}
    historial = []
    rows = df.select(
        "home_team", "away_team", "home_score", "away_score",
        "tournament", "neutral"
    ).rows()

    for home, away, hs, as_, torneo, neutral in rows:
        rh = ratings.get(home, rating_inicial)
        ra = ratings.get(away, rating_inicial)
        ventaja = 0.0 if neutral else VENTAJA_LOCAL
        dr = (rh + ventaja) - ra

        esperado_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        if hs > as_:
            real_h, outcome = 1.0, 0
        elif hs == as_:
            real_h, outcome = 0.5, 1
        else:
            real_h, outcome = 0.0, 2
        historial.append((dr, outcome))

        # K base por importancia, amplificado por margen de goles
        k = peso_torneo(torneo)
        gd = abs(hs - as_)
        if gd == 2:
            k *= 1.5
        elif gd == 3:
            k *= 1.75
        elif gd >= 4:
            k *= 1.75 + (gd - 3) / 8.0

        ajuste = k * (real_h - esperado_h)
        ratings[home] = rh + ajuste
        ratings[away] = ra - ajuste

    return ratings, historial


def calibrar_ordinal(historial) -> tuple[float, float]:
    """Ajusta (s, theta) del modelo ordinal logistico sobre dr -> P(W/D/L)."""
    dr = np.array([h[0] for h in historial], dtype=float)
    out = np.array([h[1] for h in historial], dtype=int)

    def nll(params):
        log_s, log_theta = params
        s, theta = np.exp(log_s), np.exp(log_theta)
        z = dr / s
        pH = _sig(z - theta)
        p_no_loss = _sig(z + theta)
        pD = np.clip(p_no_loss - pH, 1e-9, 1.0)
        pA = np.clip(1.0 - p_no_loss, 1e-9, 1.0)
        pH = np.clip(pH, 1e-9, 1.0)
        p = np.where(out == 0, pH, np.where(out == 1, pD, pA))
        return -np.sum(np.log(p))

    res = minimize(nll, x0=[np.log(280.0), np.log(0.9)], method="Nelder-Mead")
    s, theta = np.exp(res.x[0]), np.exp(res.x[1])
    return s, theta


def prob_wdl(dr: float, s: float, theta: float) -> tuple[float, float, float]:
    """(P local, P empate, P visita) dada la diferencia de rating efectiva."""
    z = dr / s
    pH = _sig(z - theta)
    p_no_loss = _sig(z + theta)
    pD = p_no_loss - pH
    pA = 1.0 - p_no_loss
    return pH, pD, pA


def _sig(u):
    return 1.0 / (1.0 + np.exp(-u))
