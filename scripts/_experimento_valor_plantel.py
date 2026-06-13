"""EXPERIMENTO (no toca produccion): valida si el VALOR DE PLANTEL mejora la prediccion.

Compara dos modelos de resultado W/D/L con validacion cruzada leave-one-World-Cup-out
(entrena en 2 Mundiales, evalua en el 3ro):
  - Solo Elo         : z = w_elo * dif_elo
  - Elo + valor      : z = w_elo * dif_elo + w_val * dif_log_valor
Modelo ordinal: pH=sig(z-theta), p_no_loss=sig(z+theta). Metricas: log-loss y acierto.
Valores de plantel por Mundial (Transfermarkt / theScore / Boardroom).
"""
import _bootstrap  # noqa: F401
from datetime import date

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit as sig

from mundial.datos import cargar_jugados
from mundial.elo import correr_elo, VENTAJA_LOCAL

MUNDIALES = {2014: date(2014, 6, 12), 2018: date(2018, 6, 14), 2022: date(2022, 11, 20)}

VALORES = {
    2014: {  # USD mln (theScore)
        "Brazil": 718.3, "Croatia": 258.76, "Mexico": 94.49, "Cameroon": 184.25,
        "Spain": 673.57, "Netherlands": 248.78, "Chile": 200.03, "Australia": 61.12,
        "Colombia": 229.40, "Greece": 120.77, "Ivory Coast": 207.63, "Japan": 167.66,
        "Uruguay": 260.50, "Costa Rica": 51.75, "England": 493.23, "Italy": 448.80,
        "Switzerland": 194.77, "Ecuador": 114.99, "France": 555.07, "Honduras": 45.05,
        "Argentina": 654.48, "Bosnia and Herzegovina": 177.10, "Iran": 54.27, "Nigeria": 138.46,
        "Germany": 621.82, "Portugal": 399.52, "Ghana": 150.74, "United States": 77.46,
        "Belgium": 467.86, "Algeria": 104.93, "Russia": 261.97, "South Korea": 83.33,
    },
    2018: {  # EUR mln (Transfermarkt)
        "France": 1520, "England": 1360, "Spain": 1220, "Portugal": 1010, "Germany": 947,
        "Brazil": 928.2, "Argentina": 807.5, "Belgium": 547.5, "Senegal": 478.1, "Morocco": 447.7,
        "Sweden": 406.08, "Croatia": 387.3, "Denmark": 365, "Uruguay": 359.3, "Switzerland": 332.5,
        "Colombia": 302.35, "Japan": 270.85, "Russia": 242.7, "Poland": 231.6, "Serbia": 209.5,
        "Mexico": 191.85, "Nigeria": 172.05, "South Korea": 139.05, "Egypt": 116.48, "Iceland": 89.10,
        "Australia": 77.45, "Tunisia": 69.95, "Saudi Arabia": 40.68, "Panama": 34.55, "Iran": 32.05,
        "Peru": 30.63, "Costa Rica": 28.20,
    },
    2022: {  # EUR mln (Boardroom)
        "England": 1260, "Brazil": 1140, "France": 1080, "Portugal": 937, "Spain": 902,
        "Germany": 885.5, "Argentina": 633.2, "Netherlands": 587.25, "Belgium": 563.2, "Uruguay": 449.7,
        "Croatia": 377, "Serbia": 359.5, "Denmark": 353, "Senegal": 288, "Switzerland": 281,
        "United States": 277.4, "Poland": 255.6, "Morocco": 251.1, "Ghana": 216.9, "Canada": 187.3,
        "Mexico": 176.1, "South Korea": 164.48, "Wales": 160.15, "Cameroon": 155, "Japan": 154,
        "Ecuador": 146.5, "Tunisia": 62.4, "Iran": 59.53, "Australia": 38.34, "Saudi Arabia": 25.2,
        "Qatar": 14.9, "Costa Rica": 23,
    },
}

df = cargar_jugados()


def features(anio, inicio):
    """Para cada partido del Mundial: [dif_elo/400, dif_log_valor, outcome]."""
    previos = df.filter(pl.col("date") < inicio)
    ratings, _ = correr_elo(previos)
    test = df.filter((pl.col("tournament") == "FIFA World Cup")
                     & (pl.col("date") >= inicio) & (pl.col("date") < date(anio + 1, 1, 1)))
    vals = VALORES[anio]
    rows, skip = [], 0
    for home, away, hs, as_, neutral in test.select(
            "home_team", "away_team", "home_score", "away_score", "neutral").rows():
        if home not in vals or away not in vals:
            skip += 1
            continue
        ventaja = 0.0 if neutral else VENTAJA_LOCAL
        delo = (ratings.get(home, 1500.0) + ventaja - ratings.get(away, 1500.0)) / 400.0
        dlv = np.log(vals[home]) - np.log(vals[away])
        out = 0 if hs > as_ else (1 if hs == as_ else 2)
        rows.append((delo, dlv, out))
    return np.array(rows), skip


def nll(params, X, y, usar_val):
    if usar_val:
        z = params[0] * X[:, 0] + params[1] * X[:, 1]; theta = abs(params[2])
    else:
        z = params[0] * X[:, 0]; theta = abs(params[1])
    pH = sig(z - theta); pnl = sig(z + theta)
    pD = np.clip(pnl - pH, 1e-9, 1); pA = np.clip(1 - pnl, 1e-9, 1); pH = np.clip(pH, 1e-9, 1)
    p = np.where(y == 0, pH, np.where(y == 1, pD, pA))
    return -np.sum(np.log(p))


def entrenar(X, y, usar_val):
    x0 = [1.0, 1.0, 0.9] if usar_val else [1.0, 0.9]
    return minimize(nll, x0, args=(X, y, usar_val), method="Nelder-Mead").x


def evaluar(params, X, y, usar_val):
    if usar_val:
        z = params[0] * X[:, 0] + params[1] * X[:, 1]; theta = abs(params[2])
    else:
        z = params[0] * X[:, 0]; theta = abs(params[1])
    pH = sig(z - theta); pnl = sig(z + theta)
    P = np.clip(np.stack([pH, pnl - pH, 1 - pnl], 1), 1e-9, 1)
    P /= P.sum(1, keepdims=True)
    ll = -np.mean(np.log(P[np.arange(len(y)), y]))
    acc = np.mean(np.argmax(P, 1) == y)
    return ll, acc


data = {a: features(a, MUNDIALES[a]) for a in MUNDIALES}
print("Cobertura (partidos con valor de ambos / saltados):")
for a in MUNDIALES:
    print(f"  Mundial {a}: {len(data[a][0])} partidos usados, {data[a][1]} saltados")

print("\nValidacion cruzada leave-one-World-Cup-out:")
print(f"  {'test':<8}{'Elo LL':>9}{'Elo acc':>9}{'+Val LL':>10}{'+Val acc':>10}")
res = []
for test_anio in MUNDIALES:
    Xtr = np.vstack([data[a][0][:, :2] for a in MUNDIALES if a != test_anio])
    ytr = np.concatenate([data[a][0][:, 2].astype(int) for a in MUNDIALES if a != test_anio])
    Xte = data[test_anio][0][:, :2]; yte = data[test_anio][0][:, 2].astype(int)
    p_elo = entrenar(Xtr, ytr, False); p_val = entrenar(Xtr, ytr, True)
    lle, ae = evaluar(p_elo, Xte, yte, False)
    llv, av = evaluar(p_val, Xte, yte, True)
    print(f"  {test_anio:<8}{lle:>9.4f}{ae:>9.1%}{llv:>10.4f}{av:>10.1%}")
    res.append((lle, ae, llv, av))

m = np.mean(res, 0)
print("  " + "-" * 46)
print(f"  {'PROM':<8}{m[0]:>9.4f}{m[1]:>9.1%}{m[2]:>10.4f}{m[3]:>10.1%}")
print(f"\nValor de plantel {'MEJORA' if m[2] < m[0] else 'NO mejora'} el log-loss "
      f"({m[2]:.4f} vs {m[0]:.4f}); acierto {m[3]:.1%} vs {m[1]:.1%}.")
# peso del valor entrenado con todo
allX = np.vstack([data[a][0][:, :2] for a in MUNDIALES])
allY = np.concatenate([data[a][0][:, 2].astype(int) for a in MUNDIALES])
pv = entrenar(allX, allY, True)
print(f"Pesos (todo): w_elo={pv[0]:.3f}  w_valor={pv[1]:.3f}  -> el valor "
      f"{'aporta' if abs(pv[1]) > 0.05 else 'casi no aporta'} senial propia.")
