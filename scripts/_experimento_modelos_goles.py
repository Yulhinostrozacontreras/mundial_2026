"""EXPERIMENTO (no toca produccion): compara modelos de goles vs Dixon-Coles.

Mismo backtesting out-of-sample del script 05 (entrena con datos previos a cada
Mundial, predice 2014/2018/2022). Compara variantes que solo usan goles:
  - Dixon-Coles (actual, referencia)
  - Poisson simple (sin la correccion tau)
  - Dixon-Coles + regularizacion ridge (shrinkage para equipos con pocos datos)
Metricas: W/D/L (log-loss, brier, acierto) y MARCADOR (log-loss del score exacto).
Ademas mide sobreajuste: gap log-loss in-sample vs out-of-sample.
NO modifica dixon_coles.py ni produccion.
"""
import _bootstrap  # noqa: F401
from datetime import date

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import gammaln

from mundial.datos import cargar_jugados
from mundial import dixon_coles as dc

MUNDIALES = {2014: date(2014, 6, 12), 2018: date(2018, 6, 14), 2022: date(2022, 11, 20)}
MAXG = 10
df = cargar_jugados()


def ajustar(prep, usar_tau=True, ridge=0.0, maxiter=400):
    """Variante generica del ajuste DC: usar_tau=False -> Poisson; ridge>0 -> shrinkage."""
    n = prep["n"]
    h, a, hs, as_, w = prep["h"], prep["a"], prep["hs"], prep["as_"], prep["w"]
    neut = prep["neutral"].astype(float)
    cte_h, cte_a = gammaln(hs + 1.0), gammaln(as_ + 1.0)

    def nll(p):
        base, home, rho = p[0], p[1], p[2]
        att = p[3:3 + n].copy(); deff = p[3 + n:3 + 2 * n].copy()
        att -= att.mean(); deff -= deff.mean()
        log_lh = base + home * (1.0 - neut) + att[h] - deff[a]
        log_la = base + att[a] - deff[h]
        lam, mu = np.exp(log_lh), np.exp(log_la)
        ll = (hs * log_lh - lam - cte_h) + (as_ * log_la - mu - cte_a)
        if usar_tau:
            ll += np.log(dc._tau(hs, as_, lam, mu, rho))
        val = -np.sum(w * ll)
        if ridge > 0:
            val += ridge * (np.sum(att ** 2) + np.sum(deff ** 2))
        return val

    x0 = np.concatenate([[0.0, 0.25, -0.03], np.zeros(2 * n)])
    bounds = [(None, None), (0.0, 1.0), (-0.2, 0.2)] + [(-3, 3)] * (2 * n)
    if not usar_tau:
        bounds[2] = (0.0, 0.0)
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=maxiter))
    p = res.x
    att = p[3:3 + n] - p[3:3 + n].mean()
    deff = p[3 + n:3 + 2 * n] - p[3 + n:3 + 2 * n].mean()
    return dict(base=float(p[0]), home=float(p[1]), rho=float(p[2] if usar_tau else 0.0),
                att=att, deff=deff, equipos=prep["equipos"], idx=prep["idx"])


def _matriz(modelo, home, away, neutral):
    lam, mu = dc.tasas(modelo, home, away, neutral)
    g = np.arange(MAXG + 1)
    ph = np.exp(g * np.log(lam) - lam - gammaln(g + 1.0))
    pa = np.exp(g * np.log(mu) - mu - gammaln(g + 1.0))
    m = np.outer(ph, pa)
    rho = modelo["rho"]
    if rho != 0.0:
        m[0, 0] *= 1.0 - lam * mu * rho
        m[0, 1] *= 1.0 + lam * rho
        m[1, 0] *= 1.0 + mu * rho
        m[1, 1] *= 1.0 - rho
    return m / m.sum()


def prob_wdl(modelo, home, away, neutral):
    m = _matriz(modelo, home, away, neutral)
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()


def ll_marcador(modelo, home, away, neutral, hs, as_):
    m = _matriz(modelo, home, away, neutral)
    i, j = min(hs, MAXG), min(as_, MAXG)
    return -np.log(max(m[i, j], 1e-12))


def metricas_wdl(probs, reales):
    probs, reales = np.array(probs), np.array(reales)
    p_real = probs[np.arange(len(reales)), reales]
    logloss = -np.mean(np.log(np.clip(p_real, 1e-12, 1)))
    y = np.zeros_like(probs); y[np.arange(len(reales)), reales] = 1.0
    brier = np.mean(np.sum((probs - y) ** 2, axis=1))
    acc = np.mean(np.argmax(probs, axis=1) == reales)
    return logloss, brier, acc


MODELOS = {
    "Dixon-Coles": dict(usar_tau=True, ridge=0.0),
    "Poisson simple": dict(usar_tau=False, ridge=0.0),
    "DC + ridge 0.5": dict(usar_tau=True, ridge=0.5),
    "DC + ridge 2.0": dict(usar_tau=True, ridge=2.0),
}

acum = {k: dict(ll=[], br=[], ac=[], llm=[], llm_in=[]) for k in MODELOS}

for anio, inicio in MUNDIALES.items():
    test = df.filter((pl.col("tournament") == "FIFA World Cup")
                     & (pl.col("date") >= inicio) & (pl.col("date") < date(anio + 1, 1, 1)))
    previos = df.filter(pl.col("date") < inicio)
    prep = dc.preparar(previos, desde=date(anio - 12, 1, 1), fecha_ref=inicio,
                       half_life_dias=730, min_partidos=20)
    # muestra in-sample: ultimos 600 partidos de entrenamiento con cobertura
    insample = previos.filter(pl.col("date") >= date(anio - 1, 1, 1)).tail(600)

    print(f"\n=== Mundial {anio} ===")
    print(f"  {'modelo':<16}{'WDL-LL':>9}{'brier':>8}{'acc':>7}{'SCORE-LL':>10}{'(in)':>8}{'gap':>7}")
    for nombre, cfg in MODELOS.items():
        modelo = ajustar(prep, **cfg)
        pw, reales, llm = [], [], []
        for home, away, hs, as_, neutral in test.select(
                "home_team", "away_team", "home_score", "away_score", "neutral").rows():
            if home not in modelo["idx"] or away not in modelo["idx"]:
                continue
            pw.append(prob_wdl(modelo, home, away, neutral))
            reales.append(0 if hs > as_ else (1 if hs == as_ else 2))
            llm.append(ll_marcador(modelo, home, away, neutral, hs, as_))
        # in-sample score log-loss
        llm_in = []
        for home, away, hs, as_, neutral in insample.select(
                "home_team", "away_team", "home_score", "away_score", "neutral").rows():
            if home in modelo["idx"] and away in modelo["idx"]:
                llm_in.append(ll_marcador(modelo, home, away, neutral, hs, as_))
        ll, br, ac = metricas_wdl(pw, reales)
        sm, smin = float(np.mean(llm)), float(np.mean(llm_in))
        print(f"  {nombre:<16}{ll:>9.4f}{br:>8.4f}{ac:>7.1%}{sm:>10.4f}{smin:>8.4f}{sm - smin:>7.4f}")
        acum[nombre]["ll"].append(ll); acum[nombre]["br"].append(br); acum[nombre]["ac"].append(ac)
        acum[nombre]["llm"].append(sm); acum[nombre]["llm_in"].append(smin)

print("\n" + "=" * 64)
print("PROMEDIO (3 Mundiales) - menor log-loss es mejor")
print(f"  {'modelo':<16}{'WDL-LL':>9}{'brier':>8}{'acc':>7}{'SCORE-LL':>10}{'gap(over)':>11}")
for nombre in MODELOS:
    d = acum[nombre]
    ll, br, ac = np.mean(d["ll"]), np.mean(d["br"]), np.mean(d["ac"])
    sm, gap = np.mean(d["llm"]), np.mean(d["llm"]) - np.mean(d["llm_in"])
    print(f"  {nombre:<16}{ll:>9.4f}{br:>8.4f}{ac:>7.1%}{sm:>10.4f}{gap:>11.4f}")
print("\nSCORE-LL = log-loss del marcador exacto (calidad prediciendo el score).")
print("gap(over) = cuanto peor predice fuera-de-muestra que dentro (mayor = mas sobreajuste).")
