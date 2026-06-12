"""05 - Compara Elo vs Poisson prediciendo Mundiales pasados (out-of-sample).

Para cada Mundial objetivo se entrena cada modelo SOLO con datos previos y se
predice el resultado (gana/empata/pierde) de todos sus partidos. Metricas:
log-loss (menor mejor), Brier (menor mejor) y acierto de la clase mas probable.
"""
import _bootstrap  # noqa: F401
from datetime import date

import numpy as np
import polars as pl

from mundial.datos import cargar_jugados
from mundial.elo import correr_elo, calibrar_ordinal, prob_wdl, VENTAJA_LOCAL
from mundial import dixon_coles as dc

MUNDIALES = {2014: date(2014, 6, 12), 2018: date(2018, 6, 14), 2022: date(2022, 11, 20)}
df = cargar_jugados()


def partidos_mundial(anio: int, inicio: date) -> pl.DataFrame:
    return df.filter(
        (pl.col("tournament") == "FIFA World Cup")
        & (pl.col("date") >= inicio)
        & (pl.col("date") < date(anio + 1, 1, 1))
    )


def outcome(hs, as_):
    return 0 if hs > as_ else (1 if hs == as_ else 2)


def metricas(probs, reales):
    probs = np.array(probs)
    reales = np.array(reales)
    eps = 1e-12
    p_real = probs[np.arange(len(reales)), reales]
    logloss = -np.mean(np.log(np.clip(p_real, eps, 1)))
    y = np.zeros_like(probs)
    y[np.arange(len(reales)), reales] = 1.0
    brier = np.mean(np.sum((probs - y) ** 2, axis=1))
    acc = np.mean(np.argmax(probs, axis=1) == reales)
    return logloss, brier, acc


resumen = []
for anio, inicio in MUNDIALES.items():
    test = partidos_mundial(anio, inicio)
    previos = df.filter(pl.col("date") < inicio)

    # --- Elo ---
    ratings, hist = correr_elo(previos)
    s, theta = calibrar_ordinal(hist)

    # --- Poisson ---
    prep = dc.preparar(previos, desde=date(anio - 12, 1, 1), fecha_ref=inicio,
                       half_life_dias=730, min_partidos=20)
    modelo = dc.ajustar(prep, maxiter=400)

    pe, pd_, reales = [], [], []
    saltados = 0
    for home, away, hs, as_, _, neutral in test.select(
            "home_team", "away_team", "home_score", "away_score",
            "tournament", "neutral").rows():
        if home not in modelo["idx"] or away not in modelo["idx"]:
            saltados += 1
            continue
        ventaja = 0.0 if neutral else VENTAJA_LOCAL
        dr = ratings.get(home, 1500.0) + ventaja - ratings.get(away, 1500.0)
        pe.append(prob_wdl(dr, s, theta))
        pd_.append(dc.prob_wdl(modelo, home, away, neutral))
        reales.append(outcome(hs, as_))

    le, be, ae = metricas(pe, reales)
    ld, bd, ad = metricas(pd_, reales)
    n = len(reales)
    print(f"\n=== Mundial {anio}  ({n} partidos, {saltados} sin cobertura DC) ===")
    print(f"  {'modelo':<14}{'log-loss':>10}{'brier':>9}{'acierto':>9}")
    print(f"  {'Elo':<14}{le:>10.4f}{be:>9.4f}{ae:>9.1%}")
    print(f"  {'Poisson':<14}{ld:>10.4f}{bd:>9.4f}{ad:>9.1%}")
    resumen.append((anio, le, be, ae, ld, bd, ad))

# promedio global
arr = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in resumen])
m = arr.mean(axis=0)
print("\n" + "=" * 48)
print("PROMEDIO (3 Mundiales)")
print(f"  {'modelo':<14}{'log-loss':>10}{'brier':>9}{'acierto':>9}")
print(f"  {'Elo':<14}{m[0]:>10.4f}{m[1]:>9.4f}{m[2]:>9.1%}")
print(f"  {'Poisson':<14}{m[3]:>10.4f}{m[4]:>9.4f}{m[5]:>9.1%}")
ganador = "Poisson" if m[3] < m[0] else "Elo"
print(f"\nMejor por log-loss: {ganador}")

# guardar resumen para el Streamlit
filas = []
for anio, le, be, ae, ld, bd, ad in resumen:
    filas.append({"mundial": str(anio), "motor": "Elo", "log_loss": le, "brier": be, "acierto": ae})
    filas.append({"mundial": str(anio), "motor": "Poisson", "log_loss": ld, "brier": bd, "acierto": ad})
filas.append({"mundial": "Promedio", "motor": "Elo", "log_loss": m[0], "brier": m[1], "acierto": m[2]})
filas.append({"mundial": "Promedio", "motor": "Poisson", "log_loss": m[3], "brier": m[4], "acierto": m[5]})
PROC = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "processed"
pl.DataFrame(filas).write_parquet(PROC / "backtesting.parquet")
print(f"Guardado: backtesting.parquet")
