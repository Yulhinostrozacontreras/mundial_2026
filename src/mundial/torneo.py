"""Logica reutilizable de simulacion del Mundial 2026 (usada por el script 06
y por el Streamlit). Permite overrides de fuerza para escenarios 'que pasaria si'."""
import json
from pathlib import Path

import numpy as np
import polars as pl

from . import geo
from . import forma

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"
ETAPAS = {32: "avanza", 16: "R16", 8: "cuartos", 4: "semis", 2: "final", 1: "campeon"}

_CLAUDE_CACHE: dict = {}


def _claude_scores() -> dict:
    """Score Claude (juicio de experto) por partido, desde data/claude_scores.csv.

    {(home, away): (claude_home, claude_away, nota)}. Vacio si no existe el CSV.
    """
    if _CLAUDE_CACHE:
        return _CLAUDE_CACHE
    p = PROC.parent / "claude_scores.csv"
    if p.exists():
        df = pl.read_csv(p)
        for h, a, ch, ca, n in df.select(
                "home_team", "away_team", "claude_home", "claude_away", "nota").iter_rows():
            _CLAUDE_CACHE[(h, a)] = (int(ch), int(ca), n)
    return _CLAUDE_CACHE

# --- Bracket OFICIAL del Mundial 2026 (estructura FIFA, letras de grupo oficiales) ---
# Ronda de 32: cada partido es (slot, slot). slot = ('W', letra) ganador de grupo,
# ('R', letra) segundo, ('T', k) k-esimo mejor tercero (k=0..7).
R32_BRACKET = [
    (("R", "A"), ("R", "B")), (("W", "E"), ("T", 0)), (("W", "F"), ("R", "C")),
    (("W", "C"), ("R", "F")), (("W", "I"), ("T", 1)), (("R", "E"), ("R", "I")),
    (("W", "A"), ("T", 2)), (("W", "L"), ("T", 3)), (("W", "D"), ("T", 4)),
    (("W", "G"), ("T", 5)), (("R", "K"), ("R", "L")), (("W", "H"), ("R", "J")),
    (("W", "B"), ("T", 6)), (("W", "J"), ("R", "H")), (("W", "K"), ("T", 7)),
    (("R", "D"), ("R", "G")),
]
# pares (por indice 0..15 del R32) que se enfrentan en cada ronda siguiente
R16_PAIRS = [(1, 4), (0, 2), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)]
QF_PAIRS = [(0, 1), (4, 5), (2, 3), (6, 7)]
SF_PAIRS = [(0, 1), (2, 3)]


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

    # ajuste por VALOR DE PLANTEL (validado out-of-sample): equipos con plantel caro
    # respecto a su Elo suben, y viceversa. 147 = w_val/w_elo*400 puntos Elo por unidad
    # de log-valor (pesos del backtesting). Corrige casos como USA-Paraguay.
    vp = pl.read_csv(PROC.parent / "valor_plantel_2026.csv")
    vmap = dict(zip(vp["equipo"], vp["valor_mln"]))
    media_lv = float(np.mean([np.log(v) for v in vmap.values()]))
    logv = np.array([np.log(vmap.get(e, np.exp(media_lv))) for e in equipos])
    elo = elo + 147.0 * (logv - logv.mean())
    # propagar el valor al modelo de goles (att/deff) para que los marcadores sean
    # coherentes con las probabilidades: equipos caros marcan mas y reciben menos.
    # 0.28 calibrado para que la jerarquia de goles coincida con la de las prob (1X2).
    att = att + 0.28 * (logv - logv.mean())
    deff = deff + 0.28 * (logv - logv.mean())

    grupos = {}
    for letra, sub in grupos_df.group_by("grupo", maintain_order=True):
        grupos[letra[0]] = [idx[e] for e in sub["equipo"].to_list()]
    team2grupo = {i: l for l, ms in grupos.items() for i in ms}

    fix_por_grupo = {l: [] for l in grupos}
    fix_set = set()
    for h, a in fixtures.select("home_team", "away_team").rows():
        fix_por_grupo[team2grupo[idx[h]]].append((idx[h], idx[a]))
        fix_set.add((h, a))

    # resultados REALES de los partidos de grupo ya jugados (condicionan la simulacion)
    jugados = {}
    partidos = pl.read_parquet(PROC / "partidos.parquet")
    wc_jug = partidos.filter(
        (pl.col("tournament") == "FIFA World Cup")
        & (pl.col("date") >= pl.date(2026, 1, 1))
        & pl.col("home_score").is_not_null())
    for r in wc_jug.select("home_team", "away_team", "home_score", "away_score").iter_rows(named=True):
        if (r["home_team"], r["away_team"]) in fix_set:
            jugados[(idx[r["home_team"]], idx[r["away_team"]])] = (int(r["home_score"]), int(r["away_score"]))

    # mapeo de cada grupo interno (arbitrario) a su letra OFICIAL FIFA
    oficial_de_grupo = {l: geo.INFO[equipos[ms[0]]][3] for l, ms in grupos.items()}

    # ventaja de localia para los anfitriones (USA/Mexico/Canada) en sus partidos en
    # casa: el modelo asume campo neutral y los subestimaba (ej. USA 4-1 Paraguay).
    # localia[(ih,ia)] = +V si el local es anfitrion en su pais, -V si lo es el visita.
    ANFITRIONES = {"United States", "Mexico", "Canada"}
    VENTAJA_ANFITRION = 80.0
    localia = {}
    for h, a, pais in fixtures.select("home_team", "away_team", "country").rows():
        if pais == h and h in ANFITRIONES:
            localia[(idx[h], idx[a])] = VENTAJA_ANFITRION
        elif pais == a and a in ANFITRIONES:
            localia[(idx[h], idx[a])] = -VENTAJA_ANFITRION

    # calibracion del nivel de goles: en campo neutral el modelo subestima
    # (~2.37 vs ~2.67 real en Mundiales). Se sube el 'base' para que el promedio
    # de goles esperados del torneo sea GOL_TARGET (no cambia quien es favorito,
    # solo el nivel general; las proporciones att/def se mantienen).
    GOL_TARGET = 2.60
    base = dc["base"]
    pares = [p for ms in fix_por_grupo.values() for p in ms]
    prom = float(np.mean([np.exp(base + att[ih] - deff[ia]) +
                          np.exp(base + att[ia] - deff[ih]) for ih, ia in pares]))
    base += float(np.log(GOL_TARGET / prom))

    return dict(equipos=equipos, idx=idx, att=att, deff=deff, base=base,
                rho=dc["rho"], home=float(dc["home"]), elo=elo,
                S=float(cal["s"][0]), theta=float(cal["theta"][0]),
                grupos=grupos, team2grupo=team2grupo, fix_por_grupo=fix_por_grupo,
                oficial_de_grupo=oficial_de_grupo, jugados=jugados, localia=localia,
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
            att[i] += d / 250.0   # reflejar el ajuste tambien en Poisson
            deff[i] += d / 250.0

    grupos, fix_por_grupo = ins["grupos"], ins["fix_por_grupo"]
    jugados = ins.get("jugados", {})
    localia = ins.get("localia", {})
    home_f = ins.get("home", 0.0)

    def standings_dc():
        res = {}
        for l, ms in grupos.items():
            loc = {g: k for k, g in enumerate(ms)}
            pts = np.zeros((n_sims, 4)); gf = np.zeros((n_sims, 4)); ga = np.zeros((n_sims, 4))
            for ih, ia in fix_por_grupo[l]:
                ph, pa = loc[ih], loc[ia]
                lc = localia.get((ih, ia), 0.0)  # ventaja de anfitrion local en goles
                hb_h = home_f if lc > 0 else 0.0
                hb_a = home_f if lc < 0 else 0.0
                if (ih, ia) in jugados:  # marcador real fijo
                    sh, sa = jugados[(ih, ia)]
                    hg = np.full(n_sims, sh); ag = np.full(n_sims, sa)
                else:
                    hg = rng.poisson(np.minimum(np.exp(base + hb_h + att[ih] - deff[ia]), GOL_MAX), n_sims)
                    ag = rng.poisson(np.minimum(np.exp(base + hb_a + att[ia] - deff[ih]), GOL_MAX), n_sims)
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
                if (ih, ia) in jugados:  # resultado real fijo (W/D/L)
                    sh, sa = jugados[(ih, ia)]
                    pts[:, ph] += 3 if sh > sa else (1 if sh == sa else 0)
                    pts[:, pa] += 3 if sa > sh else (1 if sh == sa else 0)
                else:
                    z = (elo[ih] + localia.get((ih, ia), 0.0) - elo[ia]) / S
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
        gA = rng.poisson(np.minimum(np.exp(base + att[A] - deff[B]), GOL_MAX))
        gB = rng.poisson(np.minimum(np.exp(base + att[B] - deff[A]), GOL_MAX))
        coinA = rng.random(A.shape) < 1.0 / (1.0 + 10.0 ** (-(elo[A] - elo[B]) / 400.0))
        return np.where(gA > gB, A, np.where(gB > gA, B, np.where(coinA, A, B)))

    def match_elo(A, B):
        z = (elo[A] - elo[B]) / S
        pH, pnl = _sig(z - theta), _sig(z + theta)
        r = rng.random(A.shape)
        coinA = rng.random(A.shape) < 1.0 / (1.0 + 10.0 ** (-(elo[A] - elo[B]) / 400.0))
        return np.where(r < pH, A, np.where(r < pnl, np.where(coinA, A, B), B))

    oficial = ins["oficial_de_grupo"]
    ls = list(grupos)

    def knockout(res, match_fn):
        """Bracket OFICIAL FIFA: emparejamientos fijos de 1ros/2dos por letra de
        grupo; los 8 mejores terceros se asignan a sus slots por ranking de tercero
        (aproxima la tabla oficial de 495 combinaciones; no afecta a 1ros/2dos)."""
        W_of = {oficial[l]: res[l]["w"] for l in ls}
        R_of = {oficial[l]: res[l]["r"] for l in ls}
        T = np.stack([res[l]["t"] for l in ls], 1)
        TS = np.stack([res[l]["ts"] for l in ls], 1)
        thirds = np.take_along_axis(T, np.argsort(-TS, axis=1)[:, :8], 1)  # (n,8) por TS

        def slot(s):
            tipo, key = s
            return W_of[key] if tipo == "W" else (R_of[key] if tipo == "R" else thirds[:, key])

        r32 = [match_fn(slot(s1), slot(s2)) for s1, s2 in R32_BRACKET]
        r16 = [match_fn(r32[i], r32[j]) for i, j in R16_PAIRS]
        qf = [match_fn(r16[i], r16[j]) for i, j in QF_PAIRS]
        sf = [match_fn(qf[i], qf[j]) for i, j in SF_PAIRS]
        champ = match_fn(sf[0], sf[1])

        qual32 = np.concatenate([np.stack([res[l]["w"] for l in ls], 1),
                                 np.stack([res[l]["r"] for l in ls], 1), thirds], 1).ravel()
        etapas = {32: qual32, 16: np.concatenate(r32), 8: np.concatenate(r16),
                  4: np.concatenate(qf), 2: np.concatenate(sf), 1: champ}
        return {k: np.bincount(v, minlength=NT) / n_sims for k, v in etapas.items()}

    g_dc = standings_dc()
    g_el = standings_elo()
    cnt_dc = knockout(g_dc, match_dc)
    cnt_el = knockout(g_el, match_elo)
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
    og = ins["oficial_de_grupo"]
    filas = {"equipo": eq, "grupo": [og[ins["team2grupo"][i]] for i in range(ins["nt"])],
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
    bonus = ins.get("localia", {}).get((i, j), 0.0)
    z = (ins["elo"][i] + bonus - ins["elo"][j]) / ins["S"]
    pH, pnl = _sig(z - ins["theta"]), _sig(z + ins["theta"])
    return pH, pnl - pH, 1.0 - pnl


GOL_MAX = 3.6  # tope realista de goles esperados por equipo (evita goleadas irreales)


def _gol_esperado(ins, i, j, home_bonus=0.0):
    """Goles esperados de i vs j (home_bonus = ventaja de localia si i es anfitrion)."""
    return float(min(np.exp(ins["base"] + home_bonus + ins["att"][i] - ins["deff"][j]), GOL_MAX))


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
    """Los 72 partidos de grupos con prob (Elo), marcador estimado (DC), sede y
    score real (si el partido ya se jugo)."""
    fixtures = pl.read_parquet(PROC / "fixtures_2026.parquet")
    idx = ins["idx"]
    jugados = ins.get("jugados", {})
    out = []
    home_f = ins.get("home", 0.0)
    for row in fixtures.iter_rows(named=True):
        h, a = row["home_team"], row["away_team"]
        ih, ia = idx[h], idx[a]
        pH, pD, pA = _elo_wdl(ins, ih, ia)
        lc = ins.get("localia", {}).get((ih, ia), 0.0)  # ventaja de anfitrion local
        hb_h = home_f if lc > 0 else 0.0
        hb_a = home_f if lc < 0 else 0.0
        sg_h, sg_a, _, _ = forma.sugerencia(h, a, row["date"], row.get("country"))
        cl = _claude_scores().get((h, a))
        out.append(dict(grupo=ins["oficial_de_grupo"][ins["team2grupo"][ih]], fecha=row["date"], home=h, away=a,
                        p_home=pH, p_draw=pD, p_away=pA,
                        gol_home=_gol_esperado(ins, ih, ia, hb_h),
                        gol_away=_gol_esperado(ins, ia, ih, hb_a),
                        sug_home=sg_h, sug_away=sg_a,
                        claude_home=cl[0] if cl else None,
                        claude_away=cl[1] if cl else None,
                        claude_nota=cl[2] if cl else None,
                        city=row.get("city"), country=row.get("country"),
                        score_real=jugados.get((ih, ia))))
    return out


# orden de las llaves del R32 segun el arbol oficial (para dibujar el bracket lineal)
_LAYOUT_R32 = [1, 4, 0, 2, 10, 11, 8, 9, 3, 5, 6, 7, 13, 15, 12, 14]


def bracket_proyectado(ins: dict) -> list:
    """Bracket determinista OFICIAL (favorito por Elo en cada llave).

    Devuelve [nombres_R32(32), R16(16), QF(8), SF(4), F(2), campeon(1)]
    en orden de llave; usa la estructura oficial FIFA (no la siembra por Elo).
    """
    tg = tabla_grupos(ins)
    oficial = ins["oficial_de_grupo"]
    W = {oficial[l]: tg[l][0][0] for l in tg}
    R = {oficial[l]: tg[l][1][0] for l in tg}
    thirds = [t[0] for t in sorted([(tg[l][2][0], tg[l][2][1]) for l in tg],
                                   key=lambda x: -x[1])[:8]]
    elo = ins["elo"]

    def slot(s):
        tipo, key = s
        return W[key] if tipo == "W" else (R[key] if tipo == "R" else thirds[key])

    # equipos del R32 en orden de layout (los pares que se unen quedan consecutivos)
    cur = [t for i in _LAYOUT_R32 for t in (slot(R32_BRACKET[i][0]), slot(R32_BRACKET[i][1]))]
    rondas = [cur]
    while len(cur) > 1:
        cur = [cur[k] if elo[cur[k]] >= elo[cur[k + 1]] else cur[k + 1]
               for k in range(0, len(cur), 2)]
        rondas.append(cur)
    eq = ins["equipos"]
    return [[eq[i] for i in r] for r in rondas]


def _standings_reales(ins: dict):
    """Posiciones REALES por grupo a partir de los partidos ya jugados.

    Devuelve (tabla, n_completos) donde tabla[letra_oficial] = lista ordenada de
    (idx, pts, dg, gf, pj) por puntos/dif.goles/goles a favor (criterio FIFA,
    sin head-to-head). n_completos = cuantos grupos tienen sus 3 fechas jugadas.
    """
    jug = ins.get("jugados", {})
    og = ins["oficial_de_grupo"]
    tabla, n_completos = {}, 0
    for l, ms in ins["grupos"].items():
        ac = {t: [0, 0, 0, 0] for t in ms}  # pts, gf, ga, pj
        for ih, ia in ins["fix_por_grupo"][l]:
            if (ih, ia) not in jug:
                continue
            sh, sa = jug[(ih, ia)]
            ac[ih][3] += 1; ac[ia][3] += 1
            ac[ih][1] += sh; ac[ih][2] += sa
            ac[ia][1] += sa; ac[ia][2] += sh
            if sh > sa:   ac[ih][0] += 3
            elif sh < sa: ac[ia][0] += 3
            else:         ac[ih][0] += 1; ac[ia][0] += 1
        filas = sorted(((t, ac[t][0], ac[t][1] - ac[t][2], ac[t][1], ac[t][3]) for t in ms),
                       key=lambda x: (-x[1], -x[2], -x[3]))
        tabla[og[l]] = filas
        if all(f[4] == 3 for f in filas):
            n_completos += 1
    return tabla, n_completos


def _lab_match(i: int) -> str:
    """Etiqueta de slots del match i del bracket: '1E - Mejor 3o', '2A - 2B', etc."""
    def lab(s):
        tipo, key = s
        return f"1{key}" if tipo == "W" else (f"2{key}" if tipo == "R" else "Mejor 3o")
    s1, s2 = R32_BRACKET[i]
    return f"{lab(s1)} - {lab(s2)}"


def _wc_ko(ins: dict):
    """Partidos de eliminatoria (WC desde 28-jun que NO son de grupos). Lista de
    (ih, ia, hs, as, fecha); hs/as None si pendiente."""
    idx = ins["idx"]
    fixtures = pl.read_parquet(PROC / "fixtures_2026.parquet")
    grupos_set = set(fixtures.select("home_team", "away_team").rows())
    partidos = pl.read_parquet(PROC / "partidos.parquet")
    wc = partidos.filter((pl.col("tournament") == "FIFA World Cup")
                         & (pl.col("date") >= pl.date(2026, 6, 28)))
    out = []
    for d, h, a, hs, as_ in wc.select("date", "home_team", "away_team",
                                      "home_score", "away_score").rows():
        if (h, a) not in grupos_set and h in idx and a in idx:
            out.append((idx[h], idx[a], hs, as_, d))
    return out


def _shootouts(ins: dict) -> dict:
    """Ganador por PENALES de los KO 2026. {frozenset({i,j}): idx_ganador}."""
    idx, out = ins["idx"], {}
    p = PROC / "shootouts.parquet"
    if p.exists():
        sh = pl.read_parquet(p).filter(pl.col("date") >= pl.date(2026, 6, 28))
        for h, a, w in sh.select("home_team", "away_team", "winner").rows():
            if h in idx and a in idx and w in idx:
                out[frozenset({idx[h], idx[a]})] = idx[w]
    return out


def _ko_resultados(ins: dict) -> dict:
    """Resultado de cada KO jugado: {frozenset({i,j}): (idx_ganador, por_penales)}.
    En empates (90'+prorroga) el ganador sale de la tanda de penales (shootouts)."""
    sho = _shootouts(ins)
    ko = {}
    for ih, ia, hs, as_, _ in _wc_ko(ins):
        if hs is None:
            continue
        key = frozenset({ih, ia})
        if hs > as_:
            ko[key] = (ih, False)
        elif as_ > hs:
            ko[key] = (ia, False)
        elif key in sho:
            ko[key] = (sho[key], True)  # empate -> penales
    return ko


def _fixture_r32_real(ins: dict) -> dict:
    """Mapea cada match de 16avos (idx 0-15 = R32_BRACKET[i]) a su partido REAL del
    dataset, identificandolo por el equipo del slot fijo (1X/2X). Devuelve
    {i: (ih, ia, hs, as, fecha)} o {} si el dataset aun no tiene los 16 partidos."""
    tabla, _ = _standings_reales(ins)
    W = {g: tabla[g][0][0] for g in tabla}
    R = {g: tabla[g][1][0] for g in tabla}
    # solo 16avos: KO con fecha hasta el 3-jul (octavos arrancan el 4-jul)
    kos = [m for m in _wc_ko(ins) if m[4] <= __import__("datetime").date(2026, 7, 3)]
    if len(kos) < 16:
        return {}
    res, usados = {}, set()
    for i, (s1, s2) in enumerate(R32_BRACKET):
        anchor = next((W[s[1]] if s[0] == "W" else R[s[1]]
                       for s in (s1, s2) if s[0] in ("W", "R")), None)
        for k, (ih, ia, hs, as_, d) in enumerate(kos):
            if k not in usados and anchor in (ih, ia):
                res[i] = (ih, ia, hs, as_, d)
                usados.add(k)
                break
    return res


def bracket_real_r32(ins: dict):
    """Las 16 llaves REALES de 16avos. Si el dataset ya tiene los enfrentamientos de
    16avos, usa esos (fixture OFICIAL); si no, los deriva de los resultados de grupos
    (1ros/2dos exactos + 8 mejores terceros por ranking, aproximacion).

    Devuelve (llaves, n_completos) con llaves = 16 tuplas (etiqueta, idxA, idxB) en
    orden de layout del bracket.
    """
    tabla, n_completos = _standings_reales(ins)
    real = _fixture_r32_real(ins)
    if real:  # fixture oficial disponible
        return [(_lab_match(i), real[i][0], real[i][1]) for i in _LAYOUT_R32], n_completos
    # derivacion aproximada (antes de que el dataset publique los 16avos)
    W = {g: tabla[g][0][0] for g in tabla}
    R = {g: tabla[g][1][0] for g in tabla}
    terceros = sorted(((g, tabla[g][2]) for g in tabla),
                      key=lambda x: (-x[1][1], -x[1][2], -x[1][3]))
    best8 = {k: terceros[k][1][0] for k in range(8)}

    def slot(s):
        tipo, key = s
        return W[key] if tipo == "W" else (R[key] if tipo == "R" else best8[key])

    llaves = []
    for i in _LAYOUT_R32:
        s1, s2 = R32_BRACKET[i]
        llaves.append((_lab_match(i), slot(s1), slot(s2)))
    return llaves, n_completos


def bracket_real_arbol(ins: dict):
    """El bracket REAL en formato arbol (mismas rondas que bracket_proyectado).

    Devuelve ([R32(32), R16(16), QF(8), SF(4), F(2), campeon(1)], n_completos)
    con NOMBRES; las celdas aun no definidas van como cadena vacia "". Los 16avos
    salen del fixture real; las rondas siguientes se llenan con los ganadores
    reales (incluido el desempate por penales) conforme se juegan.
    """
    llaves, n_completos = bracket_real_r32(ins)
    eq = ins["equipos"]
    ko = _ko_resultados(ins)
    cur = [t for _, a, b in llaves for t in (a, b)]
    rondas = [cur]
    while len(cur) > 1:
        nxt = []
        for k in range(0, len(cur), 2):
            x, y = cur[k], cur[k + 1]
            g = ko.get(frozenset({x, y})) if (x is not None and y is not None) else None
            nxt.append(g[0] if g else None)
        cur = nxt
        rondas.append(cur)
    return [[eq[i] if i is not None else "" for i in r] for r in rondas], n_completos


_CLA_CACHE: dict = {}


def _claude_csv(nombre: str) -> dict:
    """Score Claude (juicio de experto) desde data/<nombre>. {(home,away):(ch,ca,nota)}."""
    if nombre in _CLA_CACHE:
        return _CLA_CACHE[nombre]
    d = {}
    p = PROC.parent / nombre
    if p.exists():
        for h, a, ch, ca, n in pl.read_csv(p).select(
                "home_team", "away_team", "claude_home", "claude_away", "nota").iter_rows():
            d[(h, a)] = (int(ch), int(ca), n)
    _CLA_CACHE[nombre] = d
    return d


def _claude_16avos() -> dict:
    return _claude_csv("claude_16avos.csv")


def _claude_octavos() -> dict:
    return _claude_csv("claude_octavos.csv")


# Calendario OFICIAL de 16avos (Round of 32). El indice i corresponde a
# R32_BRACKET[i] = Match (73+i). (fecha, hora_local, offset_utc, sede). Fuente:
# calendario FIFA / Wikipedia 2026 FIFA World Cup knockout stage.
_FECHAS_R32 = [
    ("2026-06-28", "12:00", -7, "SoFi Stadium, Los Angeles"),
    ("2026-06-29", "16:30", -4, "Gillette Stadium, Boston"),
    ("2026-06-29", "19:00", -6, "Estadio BBVA, Monterrey"),
    ("2026-06-29", "12:00", -5, "NRG Stadium, Houston"),
    ("2026-06-30", "17:00", -4, "MetLife Stadium, Nueva York"),
    ("2026-06-30", "12:00", -5, "AT&T Stadium, Dallas"),
    ("2026-06-30", "19:00", -6, "Estadio Azteca, Ciudad de Mexico"),
    ("2026-07-01", "12:00", -4, "Mercedes-Benz Stadium, Atlanta"),
    ("2026-07-01", "13:00", -7, "Levi's Stadium, San Francisco"),
    ("2026-07-01", "13:00", -7, "Lumen Field, Seattle"),
    ("2026-07-02", "19:00", -4, "BMO Field, Toronto"),
    ("2026-07-02", "12:00", -7, "SoFi Stadium, Los Angeles"),
    ("2026-07-02", "20:00", -7, "BC Place, Vancouver"),
    ("2026-07-03", "18:00", -4, "Hard Rock Stadium, Miami"),
    ("2026-07-03", "20:30", -5, "Arrowhead Stadium, Kansas City"),
    ("2026-07-03", "13:00", -5, "AT&T Stadium, Dallas"),
]


def _peru_dt(fecha: str, hora: str, off: int):
    """Convierte la hora local de la sede (offset UTC) a hora de Peru (UTC-5)."""
    import datetime as _dt
    loc = _dt.datetime.strptime(f"{fecha} {hora}", "%Y-%m-%d %H:%M")
    return loc - _dt.timedelta(hours=off) - _dt.timedelta(hours=5)


def _enriquecer_ko(ins: dict, items: list, cla: dict) -> list:
    """Enriquece una lista de cruces de eliminatoria con el mismo detalle que la
    fase de grupos. items = [(idxA, idxB, fecha_peru, sede), ...]. Devuelve dicts
    con forma (info estadistica), modelo (prediccion), Claude, prob de avance y de
    penales, y -si ya se jugo- score real, ganador y si fue por penales."""
    import datetime as _dt
    eq, elo = ins["equipos"], ins["elo"]
    ko = _ko_resultados(ins)
    score = {frozenset({ih, ia}): (hs, as_) for ih, ia, hs, as_, _ in _wc_ko(ins) if hs is not None}
    fref = _dt.date(2026, 6, 30)  # forma sobre lo ya jugado
    out = []
    for a, b, fecha_peru, sede in items:
        _, pD, _ = _elo_wdl(ins, a, b)  # prob de empate en 90' -> prorroga/penales
        pa = 1.0 / (1.0 + 10.0 ** (-(elo[a] - elo[b]) / 400.0))  # prob de AVANZAR
        sh, sa, _, _ = forma.sugerencia(eq[a], eq[b], fref)
        cl = cla.get((eq[a], eq[b]))
        gana = ko.get(frozenset({a, b}))  # (idx_ganador, por_penales) o None
        out.append(dict(
            home=eq[a], away=eq[b], p_home=pa, p_away=1.0 - pa, p_penales=pD,
            gol_home=_gol_esperado(ins, a, b), gol_away=_gol_esperado(ins, b, a),
            sug_home=sh, sug_away=sa,
            claude_home=cl[0] if cl else None, claude_away=cl[1] if cl else None,
            claude_nota=cl[2] if cl else None, sede=sede, fecha_peru=fecha_peru,
            score_real=score.get(frozenset({a, b})),
            ganador=eq[gana[0]] if gana else None,
            por_penales=gana[1] if gana else False))
    out.sort(key=lambda m: m["fecha_peru"])
    return out


def partidos_16avos(ins: dict):
    """16avos con el detalle de la fase de grupos (forma/modelo/Claude/prob).
    Devuelve (lista, n_completos)."""
    llaves, n_completos = bracket_real_r32(ins)
    items = []
    for j, (_, a, b) in enumerate(llaves):
        midx = _LAYOUT_R32[j]  # llave j en pantalla = R32_BRACKET[midx] = Match 73+midx
        fecha, hora, off, sede = _FECHAS_R32[midx]
        items.append((a, b, _peru_dt(fecha, hora, off), sede))
    return _enriquecer_ko(ins, items, _claude_16avos()), n_completos


# Octavos (Round of 16): hora oficial de inicio en GMT por enfrentamiento. El
# dataset aporta los equipos, la fecha y la sede; aqui solo la hora de comienzo.
_HORAS_R16 = {
    ("Canada", "Morocco"): "2026-07-04 17:00",
    ("Paraguay", "France"): "2026-07-04 21:00",
    ("Brazil", "Norway"): "2026-07-05 20:00",
    ("Mexico", "England"): "2026-07-06 02:00",
    ("Argentina", "Egypt"): "2026-07-06 16:00",
    ("Portugal", "Spain"): "2026-07-06 22:00",
    ("Switzerland", "Colombia"): "2026-07-06 23:00",
    ("United States", "Belgium"): "2026-07-07 03:00",
}


# Cuartos (QF): hora oficial de inicio en GMT por enfrentamiento (equipos, fecha y
# sede vienen del dataset). France-Morocco 9-jul, Spain-Belgium 10-jul, Norway-
# England y Argentina-Switzerland 11-jul.
_HORAS_QF = {
    ("France", "Morocco"): "2026-07-09 20:00",
    ("Spain", "Belgium"): "2026-07-10 19:00",
    ("Norway", "England"): "2026-07-11 21:00",
    ("Argentina", "Switzerland"): "2026-07-12 01:00",
}


def _partidos_ko_dataset(ins: dict, horas: dict, cla: dict):
    """Cruces de una ronda KO tomados del dataset (enfrentamientos reales), con el
    mismo detalle que 16avos. `horas` = {(home,away): 'YYYY-MM-DD HH:MM' GMT} define
    que partidos entran y a que hora. Vacio si el dataset aun no los tiene."""
    import datetime as _dt
    idx = ins["idx"]
    fixtures = pl.read_parquet(PROC / "fixtures_2026.parquet")
    gset = set(fixtures.select("home_team", "away_team").rows())
    partidos = pl.read_parquet(PROC / "partidos.parquet")
    wc = partidos.filter((pl.col("tournament") == "FIFA World Cup")
                         & (pl.col("date") >= pl.date(2026, 7, 4)))
    items = []
    for h, a, city, country in wc.select("home_team", "away_team", "city", "country").rows():
        gmt = horas.get((h, a))
        if gmt is None or (h, a) in gset or h not in idx or a not in idx:
            continue
        fp = _dt.datetime.strptime(gmt, "%Y-%m-%d %H:%M") - _dt.timedelta(hours=5)  # GMT->Peru
        sede = ", ".join(x for x in (city, country) if x) or "-"
        items.append((idx[h], idx[a], fp, sede))
    return _enriquecer_ko(ins, items, cla)


def partidos_octavos(ins: dict):
    """Octavos (Round of 16), enfrentamientos reales del dataset."""
    return _partidos_ko_dataset(ins, _HORAS_R16, _claude_octavos())


def partidos_cuartos(ins: dict):
    """Cuartos de final, enfrentamientos reales del dataset."""
    return _partidos_ko_dataset(ins, _HORAS_QF, _claude_csv("claude_cuartos.csv"))
