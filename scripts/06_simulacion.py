"""06 - Simulacion Monte Carlo del Mundial 2026 (hasta el campeon).

Se simula el torneo completo N veces con DOS motores:
  - Dixon-Coles: marcadores via Poisson -> tabla de grupos con desempate por
    diferencia y goles a favor; knockout con penales si hay empate.
  - Elo: resultado W/D/L muestreado de la calibracion ordinal; desempate de
    grupos por puntos y rating Elo.
Reglas Mundial 2026: 12 grupos de 4, avanzan 1o, 2o y los 8 mejores 3os (32 a
knockout). Bracket sembrado por fuerza (limitacion documentada: no usa la tabla
oficial de cruce de terceros de FIFA).
"""
import _bootstrap  # noqa: F401
import json
from pathlib import Path

import numpy as np
import polars as pl

N_SIMS = 20000
SEED = 20260611
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
rng = np.random.default_rng(SEED)

# ---- cargar insumos ----
dc = json.loads((PROC / "dc_params.json").read_text(encoding="utf-8"))
grupos_df = pl.read_parquet(PROC / "grupos_2026.parquet")
fixtures = pl.read_parquet(PROC / "fixtures_2026.parquet")
elo_df = pl.read_parquet(PROC / "elo_ratings.parquet")
cal = pl.read_parquet(PROC / "elo_calibracion.parquet")
S, THETA = cal["s"][0], cal["theta"][0]

# indice de los 48 equipos del Mundial
equipos = grupos_df["equipo"].to_list()
idx = {e: i for i, e in enumerate(equipos)}
NT = len(equipos)

# parametros DC alineados al indice WC
dc_idx = {e: i for i, e in enumerate(dc["equipos"])}
att = np.array([dc["att"][dc_idx[e]] for e in equipos])
deff = np.array([dc["deff"][dc_idx[e]] for e in equipos])
BASE = dc["base"]
elo_map = dict(zip(elo_df["equipo"], elo_df["elo"]))
elo = np.array([elo_map[e] for e in equipos])

# grupos: letra -> [idx de 4 equipos]
grupos = {}
for letra, sub in grupos_df.group_by("grupo", maintain_order=True):
    grupos[letra[0]] = [idx[e] for e in sub["equipo"].to_list()]

# fixtures de grupo por grupo (pares de idx)
team2grupo = {idx[e]: l for l, miembros in grupos.items() for e in
              [equipos[m] for m in miembros]}
fix_por_grupo = {l: [] for l in grupos}
for h, a in fixtures.select("home_team", "away_team").rows():
    ih, ia = idx[h], idx[a]
    fix_por_grupo[team2grupo[ih]].append((ih, ia))


def _sig(u):
    return 1.0 / (1.0 + np.exp(-u))


def lam_dc(i, j):
    """Goles esperados de i vs j en campo neutral."""
    return np.exp(BASE + att[i] - deff[j])


def prob_elo(i, j):
    dr = elo[i] - elo[j]
    z = dr / S
    pH = _sig(z - THETA)
    pnl = _sig(z + THETA)
    return pH, pnl - pH, 1.0 - pnl  # pH, pD, pA


# ---------- FASE DE GRUPOS ----------
def grupos_dc():
    """Devuelve standings: para cada grupo, (winner, runner, third, third_score)."""
    res = {}
    for l, miembros in grupos.items():
        loc = {g: k for k, g in enumerate(miembros)}  # idx global -> 0..3
        pts = np.zeros((N_SIMS, 4)); gf = np.zeros((N_SIMS, 4)); ga = np.zeros((N_SIMS, 4))
        for ih, ia in fix_por_grupo[l]:
            ph, pa = loc[ih], loc[ia]
            hg = rng.poisson(lam_dc(ih, ia), N_SIMS)
            ag = rng.poisson(lam_dc(ia, ih), N_SIMS)
            pts[:, ph] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
            pts[:, pa] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
            gf[:, ph] += hg; ga[:, ph] += ag
            gf[:, pa] += ag; ga[:, pa] += hg
        comp = pts * 1e6 + (gf - ga) * 1e3 + gf
        comp += rng.random((N_SIMS, 4)) * 1e-3  # romper empates exactos al azar
        orden = np.argsort(-comp, axis=1)
        miemb = np.array(miembros)
        res[l] = dict(w=miemb[orden[:, 0]], r=miemb[orden[:, 1]],
                      t=miemb[orden[:, 2]],
                      ts=np.take_along_axis(comp, orden[:, 2:3], 1)[:, 0])
    return res


def grupos_elo():
    res = {}
    for l, miembros in grupos.items():
        loc = {g: k for k, g in enumerate(miembros)}
        pts = np.zeros((N_SIMS, 4))
        for ih, ia in fix_por_grupo[l]:
            ph, pa = loc[ih], loc[ia]
            pH, pD, _ = prob_elo(ih, ia)
            r = rng.random(N_SIMS)
            pts[:, ph] += np.where(r < pH, 3, np.where(r < pH + pD, 1, 0))
            pts[:, pa] += np.where(r >= pH + pD, 3, np.where(r < pH + pD, 1, 0))
        # desempate por rating Elo (deterministico) + ruido minimo
        comp = pts * 1e6 + np.array([elo[m] for m in miembros])[None, :]
        comp += rng.random((N_SIMS, 4)) * 1e-3
        orden = np.argsort(-comp, axis=1)
        miemb = np.array(miembros)
        res[l] = dict(w=miemb[orden[:, 0]], r=miemb[orden[:, 1]],
                      t=miemb[orden[:, 2]],
                      ts=np.take_along_axis(comp, orden[:, 2:3], 1)[:, 0])
    return res


def clasificados(res):
    """Arma los 32 clasificados (12 1os + 12 2os + 8 mejores 3os)."""
    letras = list(grupos)
    winners = np.stack([res[l]["w"] for l in letras], axis=1)
    runners = np.stack([res[l]["r"] for l in letras], axis=1)
    thirds = np.stack([res[l]["t"] for l in letras], axis=1)
    tscore = np.stack([res[l]["ts"] for l in letras], axis=1)
    mejores = np.argsort(-tscore, axis=1)[:, :8]
    thirds8 = np.take_along_axis(thirds, mejores, axis=1)
    return np.concatenate([winners, runners, thirds8], axis=1)  # (N,32)


# ---------- KNOCKOUT ----------
def bracket_order(n):
    o = [1, 2]
    while len(o) < n:
        m = len(o) * 2
        o = [x for v in o for x in (v, m + 1 - v)]
    return np.array(o)


ORDER = bracket_order(32) - 1  # indices 0..31 en orden de llave


def match_dc(A, B):
    lamA = np.exp(BASE + att[A] - deff[B])
    lamB = np.exp(BASE + att[B] - deff[A])
    gA = rng.poisson(lamA); gB = rng.poisson(lamB)
    pen = elo[A] - elo[B]
    coinA = rng.random(A.shape) < 1.0 / (1.0 + 10.0 ** (-pen / 400.0))
    win_emp = np.where(coinA, A, B)
    return np.where(gA > gB, A, np.where(gB > gA, B, win_emp))


def match_elo(A, B):
    z = (elo[A] - elo[B]) / S
    pH = _sig(z - THETA); pnl = _sig(z + THETA)
    r = rng.random(A.shape)
    coinA = rng.random(A.shape) < 1.0 / (1.0 + 10.0 ** (-(elo[A] - elo[B]) / 400.0))
    win_emp = np.where(coinA, A, B)
    return np.where(r < pH, A, np.where(r < pnl, win_emp, B))


def knockout(qual, match_fn):
    """Corre la llave y devuelve conteos por etapa (idx equipo -> #sims)."""
    strength = elo[qual]
    sortq = np.take_along_axis(qual, np.argsort(-strength, axis=1), axis=1)
    cur = sortq[:, ORDER]  # (N,32) en orden de llave
    etapas = {32: cur}
    while cur.shape[1] > 1:
        pares = cur.reshape(cur.shape[0], cur.shape[1] // 2, 2)
        cur = match_fn(pares[:, :, 0], pares[:, :, 1])
        etapas[cur.shape[1]] = cur
    conteo = {k: np.bincount(v.ravel(), minlength=NT) for k, v in etapas.items()}
    return conteo


# ---------- CORRER ----------
print(f"Simulando {N_SIMS:,} torneos con cada motor...")
res_dc = grupos_dc(); qual_dc = clasificados(res_dc); cnt_dc = knockout(qual_dc, match_dc)
res_el = grupos_elo(); qual_el = clasificados(res_el); cnt_el = knockout(qual_el, match_elo)


def pct(cnt, etapa):
    return cnt[etapa] / N_SIMS


tabla = pl.DataFrame({
    "equipo": equipos,
    "grupo": [team2grupo[i] for i in range(NT)],
    "elo": elo,
    "avanza_DC": pct(cnt_dc, 32), "final_DC": pct(cnt_dc, 2), "campeon_DC": pct(cnt_dc, 1),
    "avanza_Elo": pct(cnt_el, 32), "final_Elo": pct(cnt_el, 2), "campeon_Elo": pct(cnt_el, 1),
}).with_columns(((pl.col("campeon_DC") + pl.col("campeon_Elo")) / 2).alias("campeon_prom")
                ).sort("campeon_prom", descending=True)

tabla.write_parquet(PROC / "simulacion_2026.parquet")

print("\n" + "=" * 78)
print(f"{'PRONOSTICO MUNDIAL 2026':^78}")
print("=" * 78)
print(f"{'#':>2} {'equipo':<22}{'gpo':>4}{'campeon DC':>12}{'campeon Elo':>13}{'prom':>8}")
print("-" * 78)
for r, row in enumerate(tabla.head(16).iter_rows(named=True), 1):
    print(f"{r:>2} {row['equipo']:<22}{row['grupo']:>4}"
          f"{row['campeon_DC']:>11.1%}{row['campeon_Elo']:>12.1%}{row['campeon_prom']:>8.1%}")
print("-" * 78)
print(f"Suma campeon DC={tabla['campeon_DC'].sum():.3f}  Elo={tabla['campeon_Elo'].sum():.3f} (deben ~1.0)")
print(f"\nGuardado: simulacion_2026.parquet")
