"""Logica reutilizable de simulacion del Mundial 2026 (usada por el script 06
y por el Streamlit). Permite overrides de fuerza para escenarios 'que pasaria si'."""
import json
from pathlib import Path

import numpy as np
import polars as pl

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"
ETAPAS = {32: "avanza", 16: "R16", 8: "cuartos", 4: "semis", 2: "final", 1: "campeon"}


def cargar_insumos() -> dict:
    """Carga parametros DC, Elo, grupos y fixtures alineados a un indice de 48."""
    dc = json.loads((PROC / "dc_params.json").read_text(encoding="utf-8"))
    grupos_df = pl.read_parquet(PROC / "grupos_2026.parquet")
    fixtures = pl.read_parquet(PROC / "fixtures_2026.parquet")
    elo_df = pl.read_parquet(PROC / "elo_ratings.parquet")
    cal = pl.read_parquet(PROC / "elo_calibracion.parquet")

    equipos = grupos_df["equipo"].to_list()
    idx = {e: i for i, e in enumerate(equipos)}
    dc_idx = {e: i for i, e in enumerate(dc["equipos"])}

    att = np.array([dc["att"][dc_idx[e]] for e in equipos])
    deff = np.array([dc["deff"][dc_idx[e]] for e in equipos])
    elo_map = dict(zip(elo_df["equipo"], elo_df["elo"]))
    elo = np.array([elo_map[e] for e in equipos])

    grupos = {}
    for letra, sub in grupos_df.group_by("grupo", maintain_order=True):
        grupos[letra[0]] = [idx[e] for e in sub["equipo"].to_list()]
    team2grupo = {i: l for l, ms in grupos.items() for i in ms}

    fix_por_grupo = {l: [] for l in grupos}
    for h, a in fixtures.select("home_team", "away_team").rows():
        fix_por_grupo[team2grupo[idx[h]]].append((idx[h], idx[a]))

    return dict(equipos=equipos, idx=idx, att=att, deff=deff, base=dc["base"],
                rho=dc["rho"], elo=elo, S=float(cal["s"][0]), theta=float(cal["theta"][0]),
                grupos=grupos, team2grupo=team2grupo, fix_por_grupo=fix_por_grupo,
                nt=len(equipos))


def _sig(u):
    return 1.0 / (1.0 + np.exp(-u))


def _bracket_order(n=32):
    o = [1, 2]
    while len(o) < n:
        m = len(o) * 2
        o = [x for v in o for x in (v, m + 1 - v)]
    return np.array(o) - 1


ORDER = _bracket_order(32)


def simular(ins: dict, n_sims: int = 20000, seed: int = 20260611,
            elo_delta: dict | None = None) -> dict:
    """Corre ambos motores. elo_delta: {equipo: +/- puntos} para escenarios.

    Devuelve conteos por etapa (proporciones) para 'dc' y 'elo'.
    """
    rng = np.random.default_rng(seed)
    att, deff, elo = ins["att"].copy(), ins["deff"].copy(), ins["elo"].copy()
    base, S, theta, NT = ins["base"], ins["S"], ins["theta"], ins["nt"]

    if elo_delta:
        for eq, d in elo_delta.items():
            i = ins["idx"][eq]
            elo[i] += d
            att[i] += d / 250.0   # reflejar el ajuste tambien en Dixon-Coles
            deff[i] += d / 250.0

    grupos, fix_por_grupo = ins["grupos"], ins["fix_por_grupo"]

    def standings_dc():
        res = {}
        for l, ms in grupos.items():
            loc = {g: k for k, g in enumerate(ms)}
            pts = np.zeros((n_sims, 4)); gf = np.zeros((n_sims, 4)); ga = np.zeros((n_sims, 4))
            for ih, ia in fix_por_grupo[l]:
                ph, pa = loc[ih], loc[ia]
                hg = rng.poisson(np.exp(base + att[ih] - deff[ia]), n_sims)
                ag = rng.poisson(np.exp(base + att[ia] - deff[ih]), n_sims)
                pts[:, ph] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
                pts[:, pa] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
                gf[:, ph] += hg; ga[:, ph] += ag; gf[:, pa] += ag; ga[:, pa] += hg
            comp = pts * 1e6 + (gf - ga) * 1e3 + gf + rng.random((n_sims, 4)) * 1e-3
            res[l] = _ranking(comp, ms)
        return res

    def standings_elo():
        res = {}
        for l, ms in grupos.items():
            loc = {g: k for k, g in enumerate(ms)}
            pts = np.zeros((n_sims, 4))
            for ih, ia in fix_por_grupo[l]:
                ph, pa = loc[ih], loc[ia]
                z = (elo[ih] - elo[ia]) / S
                pH, pnl = _sig(z - theta), _sig(z + theta)
                r = rng.random(n_sims)
                pts[:, ph] += np.where(r < pH, 3, np.where(r < pnl, 1, 0))
                pts[:, pa] += np.where(r >= pnl, 3, np.where(r < pnl, 1, 0))
            comp = pts * 1e6 + np.array([elo[m] for m in ms])[None, :] + rng.random((n_sims, 4)) * 1e-3
            res[l] = _ranking(comp, ms)
        return res

    def _ranking(comp, ms):
        orden = np.argsort(-comp, axis=1)
        mi = np.array(ms)
        return dict(w=mi[orden[:, 0]], r=mi[orden[:, 1]], t=mi[orden[:, 2]],
                    ts=np.take_along_axis(comp, orden[:, 2:3], 1)[:, 0])

    def clasificados(res):
        ls = list(grupos)
        W = np.stack([res[l]["w"] for l in ls], 1)
        R = np.stack([res[l]["r"] for l in ls], 1)
        T = np.stack([res[l]["t"] for l in ls], 1)
        TS = np.stack([res[l]["ts"] for l in ls], 1)
        best = np.argsort(-TS, axis=1)[:, :8]
        return np.concatenate([W, R, np.take_along_axis(T, best, 1)], 1)

    def match_dc(A, B):
        gA = rng.poisson(np.exp(base + att[A] - deff[B]))
        gB = rng.poisson(np.exp(base + att[B] - deff[A]))
        coinA = rng.random(A.shape) < 1.0 / (1.0 + 10.0 ** (-(elo[A] - elo[B]) / 400.0))
        return np.where(gA > gB, A, np.where(gB > gA, B, np.where(coinA, A, B)))

    def match_elo(A, B):
        z = (elo[A] - elo[B]) / S
        pH, pnl = _sig(z - theta), _sig(z + theta)
        r = rng.random(A.shape)
        coinA = rng.random(A.shape) < 1.0 / (1.0 + 10.0 ** (-(elo[A] - elo[B]) / 400.0))
        return np.where(r < pH, A, np.where(r < pnl, np.where(coinA, A, B), B))

    def knockout(qual, match_fn):
        sortq = np.take_along_axis(qual, np.argsort(-elo[qual], axis=1), 1)
        cur = sortq[:, ORDER]
        etapas = {32: cur}
        while cur.shape[1] > 1:
            p = cur.reshape(cur.shape[0], cur.shape[1] // 2, 2)
            cur = match_fn(p[:, :, 0], p[:, :, 1])
            etapas[cur.shape[1]] = cur
        return {k: np.bincount(v.ravel(), minlength=NT) / n_sims for k, v in etapas.items()}

    g_dc = standings_dc()
    g_el = standings_elo()
    cnt_dc = knockout(clasificados(g_dc), match_dc)
    cnt_el = knockout(clasificados(g_el), match_elo)
    # prob de quedar 1ro / 2do de grupo (motor Elo, el mejor calibrado)
    prim = np.zeros(NT); seg = np.zeros(NT)
    for l in grupos:
        prim += np.bincount(g_el[l]["w"], minlength=NT)
        seg += np.bincount(g_el[l]["r"], minlength=NT)
    return dict(dc=cnt_dc, elo=cnt_el, primero=prim / n_sims, segundo=seg / n_sims,
                n_sims=n_sims)


def tabla_resultados(ins: dict, sim: dict) -> pl.DataFrame:
    """DataFrame con probabilidades por etapa y motor, por equipo."""
    eq = ins["equipos"]
    filas = {"equipo": eq, "grupo": [ins["team2grupo"][i] for i in range(ins["nt"])],
             "elo": ins["elo"]}
    for et, nom in ETAPAS.items():
        filas[f"{nom}_DC"] = sim["dc"][et]
        filas[f"{nom}_Elo"] = sim["elo"][et]
    if "primero" in sim:
        filas["primero_Elo"] = sim["primero"]
        filas["segundo_Elo"] = sim["segundo"]
    df = pl.DataFrame(filas).with_columns(
        ((pl.col("campeon_DC") + pl.col("campeon_Elo")) / 2).alias("campeon_prom"))
    return df.sort("campeon_prom", descending=True)


# ---------- proyeccion determinista (para bracket y tablas de grupo) ----------
def _elo_wdl(ins, i, j):
    z = (ins["elo"][i] - ins["elo"][j]) / ins["S"]
    pH, pnl = _sig(z - ins["theta"]), _sig(z + ins["theta"])
    return pH, pnl - pH, 1.0 - pnl


def _gol_esperado(ins, i, j):
    """Goles esperados de i vs j en campo neutral (Dixon-Coles)."""
    return float(np.exp(ins["base"] + ins["att"][i] - ins["deff"][j]))


def tabla_grupos(ins: dict) -> dict:
    """Por grupo: lista ordenada de (idx, pts_esperados) segun Elo."""
    out = {}
    for l, ms in ins["grupos"].items():
        filas = []
        for t in ms:
            pts = 0.0
            for ih, ia in ins["fix_por_grupo"][l]:
                if t == ih:
                    pH, pD, _ = _elo_wdl(ins, ih, ia); pts += 3 * pH + pD
                elif t == ia:
                    pH, pD, pA = _elo_wdl(ins, ih, ia); pts += 3 * pA + pD
            filas.append((t, pts))
        filas.sort(key=lambda x: -x[1])
        out[l] = filas
    return out


def partidos_grupos(ins: dict) -> list:
    """Los 72 partidos de grupos con prob (Elo) y marcador estimado (DC)."""
    fixtures = pl.read_parquet(PROC / "fixtures_2026.parquet")
    idx = ins["idx"]
    out = []
    for row in fixtures.iter_rows(named=True):
        h, a = row["home_team"], row["away_team"]
        ih, ia = idx[h], idx[a]
        pH, pD, pA = _elo_wdl(ins, ih, ia)
        out.append(dict(grupo=ins["team2grupo"][ih], fecha=row["date"], home=h, away=a,
                        p_home=pH, p_draw=pD, p_away=pA,
                        gol_home=_gol_esperado(ins, ih, ia),
                        gol_away=_gol_esperado(ins, ia, ih)))
    return out


def bracket_proyectado(ins: dict) -> list:
    """Bracket determinista (favorito por Elo en cada llave).

    Devuelve [nombres_R32(32), R16(16), QF(8), SF(4), F(2), campeon(1)]
    en orden de llave; las dos mitades solo se cruzan en la final.
    """
    tg = tabla_grupos(ins)
    ls = list(ins["grupos"])
    winners = [tg[l][0][0] for l in ls]
    runners = [tg[l][1][0] for l in ls]
    thirds = sorted([(tg[l][2][0], tg[l][2][1]) for l in ls], key=lambda x: -x[1])[:8]
    qual = np.array(winners + runners + [t[0] for t in thirds])

    sembrado = qual[np.argsort(-ins["elo"][qual])]
    slots = sembrado[ORDER]
    rondas = [slots.tolist()]
    cur = slots
    while len(cur) > 1:
        nxt = [cur[k] if ins["elo"][cur[k]] >= ins["elo"][cur[k + 1]] else cur[k + 1]
               for k in range(0, len(cur), 2)]
        cur = np.array(nxt)
        rondas.append(cur.tolist())
    eq = ins["equipos"]
    return [[eq[i] for i in r] for r in rondas]
