"""Modelo de goles Poisson (ataque/defensa por equipo, ventaja de localia y
ponderacion temporal). Originalmente Dixon-Coles; la correccion tau para
marcadores bajos se desactivo (rho=0) por sobreajustar en selecciones — ver
ajustar(). Se conserva _tau() y rho en la API por compatibilidad."""
from datetime import date

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import gammaln


def preparar(df: pl.DataFrame, desde: date, fecha_ref: date,
             half_life_dias: float = 730.0, min_partidos: int = 20):
    """Filtra ventana de entrenamiento, mapea equipos a indices y calcula pesos."""
    d = df.filter((pl.col("date") >= desde) & (pl.col("date") < fecha_ref))

    # equipos con suficientes partidos en la ventana
    apar = pl.concat([d.select(pl.col("home_team").alias("t")),
                      d.select(pl.col("away_team").alias("t"))])
    validos = (apar.group_by("t").len()
               .filter(pl.col("len") >= min_partidos)["t"].to_list())
    vset = set(validos)
    d = d.filter(pl.col("home_team").is_in(vset) & pl.col("away_team").is_in(vset))

    equipos = sorted(vset)
    idx = {e: i for i, e in enumerate(equipos)}

    h = np.array([idx[t] for t in d["home_team"]], dtype=np.int64)
    a = np.array([idx[t] for t in d["away_team"]], dtype=np.int64)
    hs = d["home_score"].to_numpy().astype(np.int64)
    as_ = d["away_score"].to_numpy().astype(np.int64)
    neutral = d["neutral"].to_numpy()

    dias = np.array([(fecha_ref - dd).days for dd in d["date"]], dtype=float)
    xi = np.log(2.0) / half_life_dias
    w = np.exp(-xi * dias)

    return dict(equipos=equipos, idx=idx, h=h, a=a, hs=hs, as_=as_,
                neutral=neutral, w=w, n=len(equipos))


def _tau(hs, as_, lam, mu, rho):
    """Correccion Dixon-Coles para los marcadores 0-0, 1-0, 0-1, 1-1."""
    t = np.ones_like(lam)
    m00 = (hs == 0) & (as_ == 0)
    m01 = (hs == 0) & (as_ == 1)
    m10 = (hs == 1) & (as_ == 0)
    m11 = (hs == 1) & (as_ == 1)
    t[m00] = 1.0 - lam[m00] * mu[m00] * rho
    t[m01] = 1.0 + lam[m01] * rho
    t[m10] = 1.0 + mu[m10] * rho
    t[m11] = 1.0 - rho
    return np.clip(t, 1e-9, None)


def ajustar(prep: dict, maxiter: int = 400) -> dict:
    """Estima parametros por maxima verosimilitud ponderada (Poisson simple).

    Se usa Poisson simple (rho=0, SIN la correccion tau de Dixon-Coles): en
    selecciones internacionales la correccion tau no mejora y ademas sobreajusta
    (validado out-of-sample en los Mundiales 2014/2018/2022). El dict mantiene
    rho=0.0 por compatibilidad (apuestas/torneo lo leen y la correccion se anula).
    """
    n = prep["n"]
    h, a, hs, as_, w = prep["h"], prep["a"], prep["hs"], prep["as_"], prep["w"]
    neut = prep["neutral"].astype(float)
    cte_h = gammaln(hs + 1.0)
    cte_a = gammaln(as_ + 1.0)

    def nll(p):
        base, home = p[0], p[1]
        att = p[2:2 + n].copy()
        deff = p[2 + n:2 + 2 * n].copy()
        att -= att.mean()
        deff -= deff.mean()
        log_lh = base + home * (1.0 - neut) + att[h] - deff[a]
        log_la = base + att[a] - deff[h]
        lam, mu = np.exp(log_lh), np.exp(log_la)
        ll = (hs * log_lh - lam - cte_h) + (as_ * log_la - mu - cte_a)
        return -np.sum(w * ll)

    x0 = np.concatenate([[0.0, 0.25], np.zeros(2 * n)])
    bounds = [(None, None), (0.0, 1.0)] + [(-3, 3)] * (2 * n)
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=maxiter))

    p = res.x
    att = p[2:2 + n] - p[2:2 + n].mean()
    deff = p[2 + n:2 + 2 * n] - p[2 + n:2 + 2 * n].mean()
    return dict(base=float(p[0]), home=float(p[1]), rho=0.0,
                att=att, deff=deff, equipos=prep["equipos"], idx=prep["idx"])


def tasas(modelo: dict, home: str, away: str, neutral: bool):
    """Lambda (goles esperados) de local y visitante."""
    i, j = modelo["idx"][home], modelo["idx"][away]
    home_adv = 0.0 if neutral else modelo["home"]
    lam = np.exp(modelo["base"] + home_adv + modelo["att"][i] - modelo["deff"][j])
    mu = np.exp(modelo["base"] + modelo["att"][j] - modelo["deff"][i])
    return lam, mu


def prob_wdl(modelo: dict, home: str, away: str, neutral: bool,
             max_goals: int = 10):
    """(P local, P empate, P visita) integrando la matriz de marcadores."""
    lam, mu = tasas(modelo, home, away, neutral)
    gh = np.arange(max_goals + 1)
    ph = np.exp(gh * np.log(lam) - lam - gammaln(gh + 1.0))
    pa = np.exp(gh * np.log(mu) - mu - gammaln(gh + 1.0))
    m = np.outer(ph, pa)
    rho = modelo["rho"]
    m[0, 0] *= 1.0 - lam * mu * rho
    m[0, 1] *= 1.0 + lam * rho
    m[1, 0] *= 1.0 + mu * rho
    m[1, 1] *= 1.0 - rho
    m /= m.sum()
    pH = np.tril(m, -1).sum()
    pD = np.trace(m)
    pA = np.triu(m, 1).sum()
    return pH, pD, pA
