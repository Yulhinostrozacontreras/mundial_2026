"""Prototipo: Sugerencia Estadistica basada en forma reciente (NO un modelo).

Para cada equipo toma sus ULTIMOS 10 partidos OFICIALES (sin amistosos) previos a
la fecha del encuentro, pondera por recencia y calidad del rival, y aproxima un
marcador. Se modula por valor de plantel (peso de las estrellas) y localia de
anfitrion. Se valida sobre los 8 partidos ya jugados comparando con el modelo y lo
real.
"""
import _bootstrap  # noqa: F401
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from mundial.elo import correr_elo
from mundial.datos import cargar_jugados

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
N_REC = 10           # ventana de forma reciente (oficiales)
DECAY = 0.85         # peso por recencia (0 = mas reciente)
K_VAL = 0.18         # sensibilidad al valor de plantel
K_RIVAL = 0.45       # ajuste por calidad del rival (Elo)
LOCALIA_STAT = 1.12  # multiplicador de goles del anfitrion en casa
GOL_MAX = 4.0
ANFITRIONES = {"United States", "Mexico", "Canada"}

# --- insumos ---
todos = pl.read_parquet(PROC / "partidos.parquet")
ofi = todos.filter(pl.col("home_score").is_not_null() & (pl.col("tournament") != "Friendly"))

# Elo pre-torneo para medir calidad de rival (datos < 11-jun)
ratings, _ = correr_elo(cargar_jugados().filter(pl.col("date") < date(2026, 6, 11)))
ELO_MEDIO = float(np.mean(list(ratings.values())))


def elo_de(eq):
    return ratings.get(eq, 1500.0)


# valor de plantel
vp = pl.read_csv(PROC.parent / "valor_plantel_2026.csv")
vmap = dict(zip(vp["equipo"], vp["valor_mln"]))
MEDIA_LV = float(np.mean([np.log(v) for v in vmap.values()]))


def logv(eq):
    return np.log(vmap.get(eq, np.exp(MEDIA_LV)))


# historial por equipo como lista de (fecha, gf, gc, elo_rival), ordenado por fecha
hist = {}
for r in ofi.select("date", "home_team", "away_team", "home_score", "away_score").iter_rows(named=True):
    h, a, hs, as_ = r["home_team"], r["away_team"], int(r["home_score"]), int(r["away_score"])
    hist.setdefault(h, []).append((r["date"], hs, as_, elo_de(a)))
    hist.setdefault(a, []).append((r["date"], as_, hs, elo_de(h)))
for e in hist:
    hist[e].sort(key=lambda x: x[0])


def indices_forma(eq, antes):
    """(ataque, defensa) reciente del equipo, ajustado por calidad del rival.

    ataque > 1 marca mas de lo normal; defensa < 1 recibe menos.
    """
    partidos = [p for p in hist.get(eq, []) if p[0] < antes][-N_REC:]
    if not partidos:
        return 1.3, 1.3, 0  # fallback neutro (~promedio de goles)
    partidos = partidos[::-1]  # mas reciente primero
    w = np.array([DECAY ** k for k in range(len(partidos))])
    w /= w.sum()
    gf = np.array([p[1] for p in partidos], dtype=float)
    gc = np.array([p[2] for p in partidos], dtype=float)
    # ajuste por calidad de rival: marcarle a un rival fuerte (Elo alto) cuenta mas;
    # recibirle goles a un rival debil cuenta mas (peor defensa).
    er = np.array([p[3] for p in partidos], dtype=float)
    fac = (er - ELO_MEDIO) / 400.0
    gf_aj = gf * (1.0 + K_RIVAL * fac)          # gol a rival fuerte vale mas
    gc_aj = gc * (1.0 - K_RIVAL * fac)          # gol recibido de rival debil pesa mas
    atk = float(np.sum(w * gf_aj))
    dfn = float(np.sum(w * gc_aj))
    return max(atk, 0.1), max(dfn, 0.1), len(partidos)


def sugerencia(home, away, anfitrion_local=False, anfitrion_visita=False):
    of_h, df_h, nh = indices_forma(home, FECHA_REF)
    of_a, df_a, na = indices_forma(away, FECHA_REF)
    # goles base = mezcla del ataque propio con la debilidad defensiva del rival
    gh = 0.5 * (of_h + df_a)
    ga = 0.5 * (of_a + df_h)
    # modulador por valor de plantel (peso de las estrellas)
    dv = logv(home) - logv(away)
    gh *= np.exp(K_VAL * dv / 2.0)
    ga *= np.exp(-K_VAL * dv / 2.0)
    # localia de anfitrion
    if anfitrion_local:
        gh *= LOCALIA_STAT
    if anfitrion_visita:
        ga *= LOCALIA_STAT
    gh = min(gh, GOL_MAX)
    ga = min(ga, GOL_MAX)
    return gh, ga, nh, na


# --- validacion sobre los 8 jugados ---
fixtures = pl.read_parquet(PROC / "fixtures_2026.parquet")
fix_set = set(fixtures.select("home_team", "away_team").rows())
pais_fix = {(h, a): p for h, a, p in fixtures.select("home_team", "away_team", "country").rows()}
jug = todos.filter((pl.col("tournament") == "FIFA World Cup")
                   & (pl.col("date") >= pl.date(2026, 1, 1))
                   & pl.col("home_score").is_not_null())
jugados = []
for r in jug.select("date", "home_team", "away_team", "home_score", "away_score").iter_rows(named=True):
    if (r["home_team"], r["away_team"]) in fix_set:
        jugados.append((r["date"], r["home_team"], r["away_team"], int(r["home_score"]), int(r["away_score"])))
jugados.sort(key=lambda x: x[0])

FECHA_REF = date(2026, 6, 11)  # forma reciente congelada al inicio del torneo

print(f"{'Partido':<40}{'real':>6}{'SUGER':>8}{'sig':>4}")
print("-" * 60)
ok_sig = 0
ok_exact = 0
for fecha, h, a, hs, as_ in jugados:
    pais = pais_fix.get((h, a))
    gh, ga, nh, na = sugerencia(h, a, anfitrion_local=(pais == h and h in ANFITRIONES),
                                anfitrion_visita=(pais == a and a in ANFITRIONES))
    mh, ma = round(gh), round(ga)
    real_sig = "L" if hs > as_ else ("E" if hs == as_ else "V")
    sug_sig = "L" if mh > ma else ("E" if mh == ma else "V")
    ok = sug_sig == real_sig
    ok_sig += ok
    ok_exact += (mh == hs and ma == as_)
    nom = f"{h[:17]} {hs}-{as_} {a[:17]}".encode("ascii", "replace").decode()
    print(f"{nom:<40}{real_sig:>6}{f'{mh}-{ma}':>8}{('OK' if ok else 'x'):>4}  (raw {gh:.2f}-{ga:.2f}, n={nh}/{na})")
print("-" * 60)
print(f"Acierto signo: {ok_sig}/{len(jugados)}   Marcador exacto: {ok_exact}/{len(jugados)}")
