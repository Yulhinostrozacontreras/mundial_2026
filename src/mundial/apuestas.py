"""Mercados de apuesta derivados del modelo de goles Dixon-Coles.

A partir de los goles esperados de cada equipo se arma la matriz de marcadores
(Poisson independiente + correccion tau de Dixon-Coles en los marcadores bajos) y
se derivan los mercados clasicos: 1X2, doble oportunidad, over/under, ambos marcan
y marcador exacto. Cada mercado entrega su seleccion mas probable, ordenadas de la
jugada mas segura a la mas arriesgada.
"""
from math import factorial

import numpy as np

KMAX = 8  # goles maximos por equipo en la matriz (cola despreciable mas alla)


def _poisson(lam: float, kmax: int = KMAX) -> np.ndarray:
    k = np.arange(kmax + 1)
    fact = np.array([factorial(int(x)) for x in k], dtype=float)
    return np.exp(-lam) * lam ** k / fact


def matriz_marcadores(lam_h: float, lam_a: float, rho: float, kmax: int = KMAX) -> np.ndarray:
    """Probabilidad de cada marcador (filas=goles local, cols=goles visita)."""
    M = np.outer(_poisson(lam_h, kmax), _poisson(lam_a, kmax))
    # correccion Dixon-Coles: corrige la dependencia en marcadores de pocos goles
    M[0, 0] *= 1.0 - lam_h * lam_a * rho
    M[0, 1] *= 1.0 + lam_h * rho
    M[1, 0] *= 1.0 + lam_a * rho
    M[1, 1] *= 1.0 - rho
    M = np.clip(M, 0.0, None)
    return M / M.sum()


def _nivel(p: float) -> tuple:
    """Etiqueta de confianza de una jugada segun su probabilidad."""
    if p >= 0.65:
        return "Segura", "🟢"
    if p >= 0.50:
        return "Probable", "🟡"
    return "Arriesgada", "🔴"


def mercados(lam_h: float, lam_a: float, rho: float,
             home: str = "Local", away: str = "Visita") -> dict:
    """Devuelve los mercados de apuesta y las jugadas ordenadas por confianza."""
    M = matriz_marcadores(lam_h, lam_a, rho)
    ii, jj = np.indices(M.shape)

    pH = float(M[ii > jj].sum())
    pX = float(M[ii == jj].sum())
    pA = float(M[ii < jj].sum())
    over = float(M[(ii + jj) >= 3].sum())   # over 2.5
    under = 1.0 - over
    btts = float(M[(ii >= 1) & (jj >= 1)].sum())
    no_btts = 1.0 - btts

    # candidatas: la seleccion favorita de cada mercado
    res_pick = max([(pH, f"Gana {home}"), (pX, "Empate"), (pA, f"Gana {away}")])
    do_pick = max([(pH + pX, f"{home} o empate"), (pH + pA, f"{home} o {away} (no empate)"),
                   (pX + pA, f"Empate o {away}")])
    tot_pick = (over, "Mas de 2.5 goles") if over >= under else (under, "Menos de 2.5 goles")
    btt_pick = (btts, "Ambos marcan: Si") if btts >= no_btts else (no_btts, "Ambos marcan: No")

    jugadas = []
    for mercado, (p, pick) in [("Resultado (1X2)", res_pick),
                               ("Doble oportunidad", do_pick),
                               ("Total de goles", tot_pick),
                               ("Ambos marcan", btt_pick)]:
        nivel, emoji = _nivel(p)
        jugadas.append(dict(mercado=mercado, pick=pick, prob=p, nivel=nivel, emoji=emoji))
    jugadas.sort(key=lambda x: -x["prob"])

    # marcadores exactos mas probables (top 3)
    plano = [(int(i), int(j), float(M[i, j])) for i in range(M.shape[0]) for j in range(M.shape[1])]
    plano.sort(key=lambda x: -x[2])
    marcadores_top = [dict(marcador=f"{i}-{j}", prob=p) for i, j, p in plano[:3]]

    return dict(jugadas=jugadas, marcadores_top=marcadores_top,
                p_home=pH, p_draw=pX, p_away=pA, over25=over, btts=btts)
