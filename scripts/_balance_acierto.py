"""Balance de acierto del modelo sobre los partidos del Mundial 2026 ya jugados.

Honesto pre-partido: el Elo se reentrena SOLO con partidos anteriores al 11-jun
(no ve ningun resultado del Mundial). El Poisson ya se entrena pre-torneo
(REF=2026-06-11). Se evaluan dos variantes para medir la correccion del 13-jun:
  A) PROD   = Elo pre-torneo + valor de plantel + localia anfitriones (modelo actual)
  B) NEUTRO = Elo pre-torneo, campo neutral, SIN valor (modelo antes del 13-jun)
"""
import _bootstrap  # noqa: F401
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from mundial.datos import cargar_jugados
from mundial.elo import correr_elo, calibrar_ordinal

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
REF = date(2026, 6, 11)
GOL_TARGET = 2.60
GOL_MAX = 3.6
FACTOR_VALOR = 147.0
FACTOR_VALOR_GOL = 0.28
VENTAJA_ANFITRION = 80.0
ANFITRIONES = {"United States", "Mexico", "Canada"}


def _sig(u):
    return 1.0 / (1.0 + np.exp(-u))


# --- 1. Elo PRE-TORNEO (solo partidos antes del 11-jun) ---
df = cargar_jugados().filter(pl.col("date") < REF)
ratings, historial = correr_elo(df)
S, theta = calibrar_ordinal(historial)

# --- 2. Poisson (dc_params.json ya es pre-torneo) ---
dc = json.loads((PROC / "dc_params.json").read_text(encoding="utf-8"))
dc_idx = {e: i for i, e in enumerate(dc["equipos"])}

# --- 3. equipos del torneo y fixtures ---
grupos_df = pl.read_parquet(PROC / "grupos_2026.parquet")
fixtures = pl.read_parquet(PROC / "fixtures_2026.parquet")
equipos = grupos_df["equipo"].to_list()
idx = {e: i for i, e in enumerate(equipos)}

att = np.array([dc["att"][dc_idx[e]] for e in equipos])
deff = np.array([dc["deff"][dc_idx[e]] for e in equipos])
elo_base = np.array([ratings.get(e, 1500.0) for e in equipos])

# valor de plantel
vp = pl.read_csv(PROC.parent / "valor_plantel_2026.csv")
vmap = dict(zip(vp["equipo"], vp["valor_mln"]))
media_lv = float(np.mean([np.log(v) for v in vmap.values()]))
logv = np.array([np.log(vmap.get(e, np.exp(media_lv))) for e in equipos])

# variante A (PROD): valor + localia
elo_prod = elo_base + FACTOR_VALOR * (logv - logv.mean())
att_prod = att + FACTOR_VALOR_GOL * (logv - logv.mean())
deff_prod = deff + FACTOR_VALOR_GOL * (logv - logv.mean())

# localia por fixture
localia = {}
for h, a, pais in fixtures.select("home_team", "away_team", "country").rows():
    if pais == h and h in ANFITRIONES:
        localia[(idx[h], idx[a])] = VENTAJA_ANFITRION
    elif pais == a and a in ANFITRIONES:
        localia[(idx[h], idx[a])] = -VENTAJA_ANFITRION

# calibracion del nivel de goles (base) para cada variante de att/deff
home_f = float(dc["home"])
pares = [(idx[h], idx[a]) for h, a in fixtures.select("home_team", "away_team").rows()]


def calibrar_base(att_v, deff_v):
    prom = float(np.mean([np.exp(dc["base"] + att_v[ih] - deff_v[ia]) +
                          np.exp(dc["base"] + att_v[ia] - deff_v[ih]) for ih, ia in pares]))
    return dc["base"] + float(np.log(GOL_TARGET / prom))


base_prod = calibrar_base(att_prod, deff_prod)
base_neu = calibrar_base(att, deff)

# --- 4. partidos jugados ---
partidos = pl.read_parquet(PROC / "partidos.parquet")
fix_set = set(fixtures.select("home_team", "away_team").rows())
jug = partidos.filter(
    (pl.col("tournament") == "FIFA World Cup")
    & (pl.col("date") >= pl.date(2026, 1, 1))
    & pl.col("home_score").is_not_null())
jugados = []
for r in jug.select("date", "home_team", "away_team", "home_score", "away_score").iter_rows(named=True):
    if (r["home_team"], r["away_team"]) in fix_set:
        jugados.append((r["date"], r["home_team"], r["away_team"],
                        int(r["home_score"]), int(r["away_score"])))
jugados.sort(key=lambda x: x[0])


def wdl(elo_v, ih, ia, usar_localia):
    bonus = localia.get((ih, ia), 0.0) if usar_localia else 0.0
    z = (elo_v[ih] + bonus - elo_v[ia]) / S
    pH, pnl = _sig(z - theta), _sig(z + theta)
    return pH, pnl - pH, 1.0 - pnl


def goles(att_v, deff_v, base_v, ih, ia, usar_localia):
    lc = localia.get((ih, ia), 0.0) if usar_localia else 0.0
    hb_h = home_f if lc > 0 else 0.0
    hb_a = home_f if lc < 0 else 0.0
    gh = min(np.exp(base_v + hb_h + att_v[ih] - deff_v[ia]), GOL_MAX)
    ga = min(np.exp(base_v + hb_a + att_v[ia] - deff_v[ih]), GOL_MAX)
    return gh, ga


def signo(hs, as_):
    return "L" if hs > as_ else ("E" if hs == as_ else "V")


def evaluar(nombre, elo_v, att_v, deff_v, base_v, usar_localia):
    print(f"\n{'='*84}\n  VARIANTE {nombre}\n{'='*84}")
    print(f"  {'Partido':<42}{'real':>7}  {'pred1X2':>8}  {'estim':>6}  {'logloss':>8}")
    print("  " + "-"*80)
    n_ok_signo = 0
    n_ok_marcador = 0
    suma_ll = 0.0
    suma_brier = 0.0
    err_dif = []
    for fecha, h, a, hs, as_ in jugados:
        ih, ia = idx[h], idx[a]
        pH, pD, pA = wdl(elo_v, ih, ia, usar_localia)
        probs = {"L": pH, "E": pD, "V": pA}
        pred = max(probs, key=probs.get)
        real = signo(hs, as_)
        ok = pred == real
        n_ok_signo += ok
        gh, ga = goles(att_v, deff_v, base_v, ih, ia, usar_localia)
        mh, ma = round(gh), round(ga)
        ok_m = (mh == hs and ma == as_)
        n_ok_marcador += ok_m
        p_real = max(probs[real], 1e-9)
        ll = -np.log(p_real)
        suma_ll += ll
        suma_brier += sum((probs[k] - (1.0 if k == real else 0.0))**2 for k in probs)
        err_dif.append(abs((mh - ma) - (hs - as_)))
        nom = f"{h[:18]} {hs}-{as_} {a[:18]}".encode("ascii", "replace").decode()
        marca = "OK " if ok else "x  "
        print(f"  {nom:<42}{real:>7}  {pred:>5}{marca}  {mh}-{ma:<4}  {ll:>8.3f}")
    n = len(jugados)
    print("  " + "-"*80)
    print(f"  Acierto 1X2 (signo): {n_ok_signo}/{n} = {100*n_ok_signo/n:.0f}%")
    print(f"  Marcador exacto    : {n_ok_marcador}/{n}")
    print(f"  Log-loss medio     : {suma_ll/n:.3f}  (menor=mejor; azar 3 clases=1.099)")
    print(f"  Brier medio        : {suma_brier/n:.3f}  (menor=mejor)")
    print(f"  Error medio dif gol: {np.mean(err_dif):.2f}")
    return n_ok_signo, suma_ll / n


print(f"Partidos evaluados: {len(jugados)} (Elo entrenado con datos < {REF})")
a_ok, a_ll = evaluar("A: PROD (Elo + valor + localia, modelo actual)",
                     elo_prod, att_prod, deff_prod, base_prod, True)
b_ok, b_ll = evaluar("B: NEUTRO (Elo solo, campo neutral, sin valor)",
                     elo_base, att, deff, base_neu, False)

print(f"\n{'='*84}\n  RESUMEN COMPARATIVO\n{'='*84}")
print(f"  PROD   : {a_ok}/{len(jugados)} aciertos 1X2, log-loss {a_ll:.3f}")
print(f"  NEUTRO : {b_ok}/{len(jugados)} aciertos 1X2, log-loss {b_ll:.3f}")
print(f"  Mejora correccion 13-jun: {a_ok - b_ok:+d} aciertos, "
      f"{b_ll - a_ll:+.3f} log-loss")
