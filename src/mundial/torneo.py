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


def bracket_real_r32(ins: dict):
    """Las 16 llaves REALES de 16avos segun los resultados ya jugados de grupos.

    Devuelve (llaves, n_completos) con llaves = lista de 16 tuplas
    (etiqueta_slot, idxA, idxB) en orden de layout del bracket. Los 8 mejores
    terceros se asignan por ranking (misma aproximacion que bracket_proyectado).
    Mientras haya grupos sin cerrar, las posiciones de esos grupos son
    provisionales (con lo jugado hasta el momento).
    """
    tabla, n_completos = _standings_reales(ins)
    W = {g: tabla[g][0][0] for g in tabla}
    R = {g: tabla[g][1][0] for g in tabla}
    terceros = sorted(((g, tabla[g][2]) for g in tabla),
                      key=lambda x: (-x[1][1], -x[1][2], -x[1][3]))
    best8 = {k: terceros[k][1][0] for k in range(8)}

    def slot(s):
        tipo, key = s
        return W[key] if tipo == "W" else (R[key] if tipo == "R" else best8[key])

    def lab(s):
        tipo, key = s
        return f"1{key}" if tipo == "W" else (f"2{key}" if tipo == "R" else "Mejor 3o")

    llaves = []
    for i in _LAYOUT_R32:
        s1, s2 = R32_BRACKET[i]
        llaves.append((f"{lab(s1)} - {lab(s2)}", slot(s1), slot(s2)))
    return llaves, n_completos


def _ko_resultados(ins: dict) -> dict:
    """Ganadores REALES de los partidos de eliminatoria ya jugados (los WC
    jugados que NO son de la fase de grupos). {frozenset({i,j}): idx_ganador}.
    Empates (definidos por penales, sin dato de marcador) quedan sin resolver."""
    idx = ins["idx"]
    fixtures = pl.read_parquet(PROC / "fixtures_2026.parquet")
    grupos_set = set(fixtures.select("home_team", "away_team").rows())
    partidos = pl.read_parquet(PROC / "partidos.parquet")
    wc = partidos.filter((pl.col("tournament") == "FIFA World Cup")
                         & (pl.col("date") >= pl.date(2026, 6, 1))
                         & pl.col("home_score").is_not_null())
    ko = {}
    for h, a, hs, as_ in wc.select("home_team", "away_team", "home_score", "away_score").rows():
        if (h, a) in grupos_set or h not in idx or a not in idx:
            continue
        if hs > as_:
            ko[frozenset({idx[h], idx[a]})] = idx[h]
        elif as_ > hs:
            ko[frozenset({idx[h], idx[a]})] = idx[a]
    return ko


def bracket_real_arbol(ins: dict):
    """El bracket REAL en formato arbol (mismas rondas que bracket_proyectado).

    Devuelve ([R32(32), R16(16), QF(8), SF(4), F(2), campeon(1)], n_completos)
    con NOMBRES; las celdas aun no definidas (eliminatorias sin jugar) van como
    cadena vacia "". Los 16avos salen de los resultados de grupos; las rondas
    siguientes se llenan con los ganadores reales conforme se jueguen.
    """
    llaves, n_completos = bracket_real_r32(ins)
    eq = ins["equipos"]
    ko = _ko_resultados(ins)
    cur = [t for _, a, b in llaves for t in (a, b)]  # 32 idx en orden de layout
    rondas = [cur]
    while len(cur) > 1:
        nxt = []
        for k in range(0, len(cur), 2):
            x, y = cur[k], cur[k + 1]
            nxt.append(ko.get(frozenset({x, y})) if (x is not None and y is not None) else None)
        cur = nxt
        rondas.append(cur)
    return [[eq[i] if i is not None else "" for i in r] for r in rondas], n_completos


_CLA16_CACHE: dict = {}

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


def _claude_16avos() -> dict:
    """Score Claude para 16avos, desde data/claude_16avos.csv. {(home,away):(ch,ca,nota)}."""
    if _CLA16_CACHE:
        return _CLA16_CACHE
    p = PROC.parent / "claude_16avos.csv"
    if p.exists():
        for h, a, ch, ca, n in pl.read_csv(p).select(
                "home_team", "away_team", "claude_home", "claude_away", "nota").iter_rows():
            _CLA16_CACHE[(h, a)] = (int(ch), int(ca), n)
    return _CLA16_CACHE


def partidos_16avos(ins: dict):
    """Para cada llave real de 16avos, con el mismo detalle que la fase de grupos:
    info estadistica (forma), prediccion del modelo (Poisson) y score Claude, mas
    la prob de AVANCE (Elo, incluye desempate por penales).

    Devuelve (lista, n_completos). Cada item: home, away, p_home, p_away (suman 1),
    gol_home, gol_away (modelo), sug_home, sug_away (forma), claude_home/away/nota,
    ganador (si la llave ya se jugo, sino None).
    """
    import datetime as _dt
    llaves, n_completos = bracket_real_r32(ins)
    eq, elo = ins["equipos"], ins["elo"]
    ko = _ko_resultados(ins)
    cla = _claude_16avos()
    fref = _dt.date(2026, 6, 30)  # tras los grupos: forma sobre lo ya jugado
    out = []
    for j, (_, a, b) in enumerate(llaves):
        midx = _LAYOUT_R32[j]  # llave j en pantalla = R32_BRACKET[midx] = Match 73+midx
        fecha, hora, off, sede = _FECHAS_R32[midx]
        pa = 1.0 / (1.0 + 10.0 ** (-(elo[a] - elo[b]) / 400.0))
        sh, sa, _, _ = forma.sugerencia(eq[a], eq[b], fref)
        cl = cla.get((eq[a], eq[b]))
        out.append(dict(
            home=eq[a], away=eq[b], p_home=pa, p_away=1.0 - pa,
            gol_home=_gol_esperado(ins, a, b), gol_away=_gol_esperado(ins, b, a),
            sug_home=sh, sug_away=sa,
            claude_home=cl[0] if cl else None, claude_away=cl[1] if cl else None,
            claude_nota=cl[2] if cl else None, sede=sede,
            fecha_peru=_peru_dt(fecha, hora, off),
            ganador=(eq[ko[frozenset({a, b})]] if frozenset({a, b}) in ko else None)))
    out.sort(key=lambda m: m["fecha_peru"])
    return out, n_completos
