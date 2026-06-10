"""Streamlit - Pronostico Mundial 2026 (Dixon-Coles vs Elo + bracket + simulador).

Correr en su propio puerto, independiente de otras apps:
    uv run streamlit run app.py --server.port 8502
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import altair as alt
import polars as pl
import streamlit as st

from mundial import apuestas, torneo

st.set_page_config(page_title="Mundial 2026 - Pronostico", page_icon="🏆", layout="wide")

PROC = Path(__file__).resolve().parent / "data" / "processed"
ETAPAS_NOM = ["avanza", "R16", "cuartos", "semis", "final", "campeon"]
ETIQ_RONDA = {"avanza": "Avanza de grupo", "R16": "Octavos", "cuartos": "Cuartos",
              "semis": "Semifinal", "final": "Final", "campeon": "Campeon"}
ROJO, AZUL, NARANJA, ORO = "#a01a45", "#3b5bdb", "#e8590c", "#f1b305"

# ---------- estilos globales ----------
st.markdown("""
<style>
.hero {background:linear-gradient(100deg,#a01a45 0%,#6d1030 60%,#3a0a1c 100%);
       padding:22px 28px;border-radius:14px;color:#fff;margin-bottom:6px;}
.hero h1 {margin:0;font-size:30px;font-weight:800;letter-spacing:.5px;}
.hero p {margin:4px 0 0;opacity:.85;font-size:14px;}
.kpi {background:#fff;border:1px solid #eee;border-left:6px solid #a01a45;border-radius:12px;
      padding:14px 18px;box-shadow:0 2px 8px rgba(0,0,0,.05);}
.kpi .eq {font-size:18px;font-weight:700;color:#222;}
.kpi .pc {font-size:30px;font-weight:800;color:#a01a45;line-height:1;}
.kpi .sub {font-size:12px;color:#888;}
.kpi.oro {border-left-color:#f1b305;} .kpi.oro .pc{color:#c98a00;}
.kpi.azul{border-left-color:#3b5bdb;} .kpi.azul .pc{color:#3b5bdb;}

/* ---- responsive: en celular las columnas se apilan y todo se compacta ---- */
@media (max-width: 640px){
  .block-container{padding:1rem .8rem !important;}
  [data-testid="stHorizontalBlock"]{flex-wrap:wrap;gap:.5rem !important;}
  [data-testid="stColumn"],[data-testid="column"]{min-width:100% !important;flex:1 1 100% !important;}
  .hero{padding:16px 18px;}
  .hero h1{font-size:22px;}
  .hero p{font-size:12px;}
  .kpi .pc{font-size:26px;}
  .stTabs [data-baseweb="tab"]{padding:6px 8px;font-size:13px;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_insumos():
    return torneo.cargar_insumos()


@st.cache_data
def get_sim(n_sims: int, seed: int, delta_items: tuple):
    ins = get_insumos()
    elo_delta = {k: v for k, v in delta_items} if delta_items else None
    sim = torneo.simular(ins, n_sims=n_sims, seed=seed, elo_delta=elo_delta)
    return torneo.tabla_resultados(ins, sim)


@st.cache_data
def get_bracket():
    return torneo.bracket_proyectado(get_insumos())


@st.cache_data
def get_tabla_grupos():
    ins = get_insumos()
    return {l: [(ins["equipos"][i], pts) for i, pts in v]
            for l, v in torneo.tabla_grupos(ins).items()}


@st.cache_data
def get_partidos():
    ins = get_insumos()
    ps = torneo.partidos_grupos(ins)
    for m in ps:
        m["apuestas"] = apuestas.mercados(m["gol_home"], m["gol_away"], ins["rho"],
                                          m["home"], m["away"])
    return ps


@st.cache_data
def get_backtest():
    p = PROC / "backtesting.parquet"
    return pl.read_parquet(p) if p.exists() else None


ins = get_insumos()
equipos = ins["equipos"]

# ---------- header ----------
st.markdown(
    '<div class="hero"><h1>🏆 MUNDIAL 2026 — Pronostico predictivo</h1>'
    '<p>Modelo de fuerza de selecciones (Dixon-Coles + Elo) y simulacion Monte Carlo de 48 equipos. '
    'Elo es el motor mejor calibrado segun backtesting.</p></div>',
    unsafe_allow_html=True)

with st.sidebar:
    st.header("Parametros")
    n_sims = st.select_slider("Simulaciones", [5000, 10000, 20000, 50000], value=20000)
    seed = st.number_input("Semilla", value=20260611, step=1)
    st.divider()
    st.caption("App en puerto **8502**. Tu app de descuento sigue viva en 8501 sin interferencia.")

base = get_sim(n_sims, int(seed), tuple())

# ---------- KPIs de favoritos ----------
top3 = base.head(3).to_dicts()
clases = ["oro", "", "azul"]
cols = st.columns(3)
for col, row, cls in zip(cols, top3, clases):
    col.markdown(
        f'<div class="kpi {cls}"><div class="sub">Grupo {row["grupo"]}</div>'
        f'<div class="eq">{row["equipo"]}</div>'
        f'<div class="pc">{row["campeon_Elo"]:.0%}</div>'
        f'<div class="sub">probabilidad de campeon (Elo) · final {row["final_Elo"]:.0%}</div></div>',
        unsafe_allow_html=True)
st.write("")

tab_pron, tab_brk, tab_grp, tab_camino, tab_esc, tab_comp = st.tabs(
    ["📊 Pronostico", "🗺️ Bracket", "🏟️ Fase de grupos", "🎯 Camino de un equipo",
     "🔮 Que pasaria si", "⚖️ Motores"])

# ================= PRONOSTICO =================
with tab_pron:
    st.subheader("Favoritos al titulo")
    top = base.head(16).select(
        "equipo", "grupo", "campeon_Elo", "campeon_DC", "campeon_prom", "final_Elo", "semis_Elo")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(
            top.rename({"campeon_Elo": "Campeon (Elo)", "campeon_DC": "Campeon (DC)",
                        "campeon_prom": "Campeon (prom)", "final_Elo": "Llega a final",
                        "semis_Elo": "Llega a semis", "grupo": "Gpo"}),
            hide_index=True, width="stretch",
            column_config={c: st.column_config.ProgressColumn(c, format="percent", min_value=0,
                           max_value=float(top[col].max()))
                           for c, col in [("Campeon (Elo)", "campeon_Elo"), ("Campeon (DC)", "campeon_DC"),
                                          ("Campeon (prom)", "campeon_prom"), ("Llega a final", "final_Elo"),
                                          ("Llega a semis", "semis_Elo")]})
    with c2:
        ch = (alt.Chart(top.head(10).to_pandas())
              .mark_bar(color=ROJO)
              .encode(x=alt.X("campeon_Elo:Q", title="Prob. campeon (Elo)", axis=alt.Axis(format="%")),
                      y=alt.Y("equipo:N", sort="-x", title=None))
              .properties(height=320))
        st.altair_chart(ch, width="stretch")


# ================= BRACKET =================
def render_bracket(rondas):
    r32, r16, qf, sf, fin = rondas[0], rondas[1], rondas[2], rondas[3], rondas[4]
    champ = rondas[5][0]

    def box(name, cls=""):
        ch_mark = " ★" if cls == "champ" else ""
        return f'<div class="bx {cls}">{name}{ch_mark}</div>'

    def col(label, teams, cls=""):
        cajas = "".join(box(t, cls) for t in teams)
        return f'<div class="col"><div class="lbl">{label}</div><div class="stk">{cajas}</div></div>'

    left = (col("16avos", r32[0:16]) + col("8vos", r16[0:8]) +
            col("4tos", qf[0:4]) + col("Semis", sf[0:2]))
    right = (col("Semis", sf[2:4]) + col("4tos", qf[4:8]) +
             col("8vos", r16[8:16]) + col("16avos", r32[16:32]))
    center = ('<div class="col center"><div class="lbl">FINAL</div>'
              f'{box(fin[0], "fin")}<div class="cup">🏆</div>{box(fin[1], "fin")}'
              f'<div class="lbl">CAMPEON</div>{box(champ, "champ")}</div>')

    css = """<style>
    *{box-sizing:border-box;font-family:'Segoe UI',Arial,sans-serif;}
    body{margin:0;overflow-x:auto;}
    .wrap{background:linear-gradient(135deg,#a01a45,#5c0d28);padding:14px;border-radius:12px;
          display:flex;justify-content:center;gap:4px;min-width:720px;}
    .col{flex:1;display:flex;flex-direction:column;align-items:center;min-width:84px;}
    .lbl{color:#ffd9e4;font-size:10px;font-weight:700;text-transform:uppercase;margin-bottom:6px;letter-spacing:.5px;}
    .stk{flex:1;display:flex;flex-direction:column;justify-content:space-around;width:100%;gap:2px;}
    .bx{background:#fff;border-radius:4px;padding:3px 6px;font-size:10px;font-weight:600;color:#2a2a2a;
        text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
        border-left:3px solid #d44d77;box-shadow:0 1px 2px rgba(0,0,0,.15);}
    .bx.fin{font-size:12px;padding:7px;border-left-color:#3b5bdb;font-weight:800;width:130px;margin:6px 0;}
    .bx.champ{background:linear-gradient(135deg,#ffe07a,#f1b305);border-left-color:#c98a00;
              font-size:13px;font-weight:800;padding:8px;width:140px;color:#5a3d00;}
    .center{justify-content:center;flex:1.3;}
    .cup{font-size:30px;margin:6px 0;}
    </style>"""
    return css + f'<div class="wrap">{left}{center}{right}</div>'


with tab_brk:
    st.subheader("Cronograma proyectado — ruta mas probable")
    st.caption("Bracket determinista: en cada llave avanza el equipo de mayor fuerza (Elo). "
               "Es el escenario 'chalk' (sin sorpresas); las probabilidades reales estan en las otras pestañas.")
    rondas = get_bracket()
    st.iframe(render_bracket(rondas), height=620)
    fin = rondas[4]
    st.success(f"**Final proyectada:** {fin[0]}  vs  {fin[1]}   →   Campeon: **{rondas[5][0]}**")
    st.caption("Limitacion: el cruce usa siembra por Elo, no la tabla oficial FIFA de los 8 mejores terceros.")


# ================= FASE DE GRUPOS =================
def render_jugadas(m):
    """Panel de mercados de apuesta de un partido (dentro de un popover)."""
    mk = m["apuestas"]
    estrella = mk["jugadas"][0]
    st.markdown(f"**{m['home']}** vs **{m['away']}**")
    st.success(f"⭐ Jugada mas segura: **{estrella['pick']}** · {estrella['prob']:.0%}")
    for j in mk["jugadas"]:
        st.markdown(f"{j['emoji']} {j['mercado']} — **{j['pick']}**")
        st.progress(min(j["prob"], 1.0), text=f"{j['prob']:.0%} · {j['nivel']}")
    tops = " · ".join(f"{t['marcador']} ({t['prob']:.0%})" for t in mk["marcadores_top"])
    st.caption(f"🎯 Marcadores mas probables: {tops}")
    st.caption("Mercados derivados del modelo de goles Dixon-Coles (no son cuotas reales de casas de apuesta).")


with tab_grp:
    st.subheader("Tabla final estimada y partidos por grupo")
    tg = get_tabla_grupos()
    partidos = get_partidos()
    clasif = set(get_bracket()[0])  # los 32 que avanzan
    prob = {r["equipo"]: r for r in base.to_dicts()}

    letras = sorted(tg.keys())
    for fila_letras in [letras[i:i + 2] for i in range(0, len(letras), 2)]:
        cols = st.columns(2)
        for col, l in zip(cols, fila_letras):
            with col:
                st.markdown(f"#### Grupo {l}")
                filas = []
                for pos, (eq, pts) in enumerate(tg[l], 1):
                    p = prob[eq]
                    marca = "🟢" if pos <= 2 else ("🟡" if eq in clasif else "⚪")
                    filas.append({"": marca, "Pos": pos, "Equipo": eq,
                                  "Pts est.": round(pts, 2),
                                  "1ro": p["primero_Elo"], "Avanza": p["avanza_Elo"]})
                st.dataframe(pl.DataFrame(filas), hide_index=True, width="stretch",
                             column_config={"1ro": st.column_config.NumberColumn(format="percent"),
                                            "Avanza": st.column_config.NumberColumn(format="percent")})
                pg = [m for m in partidos if m["grupo"] == l]
                with st.expander(f"Partidos del grupo {l} ({len(pg)}) — con jugadas sugeridas"):
                    for m in pg:
                        c1, c2, c3 = st.columns([3.2, 2.3, 1.5], vertical_alignment="center")
                        c1.markdown(f"**{m['home']}** vs **{m['away']}**  \n"
                                    f"<span style='color:#888;font-size:12px'>{m['fecha'].strftime('%d/%m')}</span>",
                                    unsafe_allow_html=True)
                        c2.markdown(
                            f"Marcador est. **{m['gol_home']:.1f}-{m['gol_away']:.1f}**  \n"
                            f"<span style='color:#888;font-size:12px'>Elo: L {m['p_home']:.0%} · "
                            f"E {m['p_draw']:.0%} · V {m['p_away']:.0%}</span>",
                            unsafe_allow_html=True)
                        with c3.popover("🎲 Jugadas", use_container_width=True):
                            render_jugadas(m)
    st.caption("🟢 clasifica directo (1ro/2do) · 🟡 mejor tercero que avanza · ⚪ eliminado (proyeccion)")


# ================= CAMINO DE UN EQUIPO =================
with tab_camino:
    eq = st.selectbox("Selecciona un equipo", sorted(equipos),
                      index=sorted(equipos).index("Argentina"))
    fila = base.filter(pl.col("equipo") == eq).row(0, named=True)
    g = fila["grupo"]
    rivales = [e for e in equipos if ins["team2grupo"][ins["idx"][e]] == g and e != eq]
    st.markdown(f"**Grupo {g}** — rivales: {', '.join(rivales)}")
    cols = st.columns(6)
    for col, nom in zip(cols, ETAPAS_NOM):
        col.metric(ETIQ_RONDA[nom], f"{fila[nom + '_Elo']:.1%}",
                   help=f"Dixon-Coles: {fila[nom + '_DC']:.1%}")
    camino = pl.DataFrame({
        "ronda": [ETIQ_RONDA[n] for n in ETAPAS_NOM],
        "Elo": [fila[n + "_Elo"] for n in ETAPAS_NOM],
        "Dixon-Coles": [fila[n + "_DC"] for n in ETAPAS_NOM],
    }).to_pandas().melt(id_vars="ronda", var_name="motor", value_name="prob")
    orden = [ETIQ_RONDA[n] for n in ETAPAS_NOM]
    ch = (alt.Chart(camino).mark_line(point=True)
          .encode(x=alt.X("ronda:N", sort=orden, title=None),
                  y=alt.Y("prob:Q", axis=alt.Axis(format="%"), title="Probabilidad"),
                  color=alt.Color("motor:N", scale=alt.Scale(range=[NARANJA, AZUL])))
          .properties(height=300))
    st.altair_chart(ch, width="stretch")


# ================= QUE PASARIA SI =================
with tab_esc:
    st.subheader("Ajusta la forma de un equipo y re-simula")
    st.caption("Sube o baja el nivel (en puntos Elo) de uno o mas equipos y compara el pronostico. "
               "+50 ~ equipo en racha; -50 ~ con bajas o lesiones.")
    elegidos = st.multiselect("Equipos a ajustar", sorted(equipos), default=["Brazil"])
    deltas = {e: st.slider(f"Ajuste {e}", -150, 150, 50, step=10, key=f"d_{e}") for e in elegidos}
    if deltas:
        esc = get_sim(n_sims, int(seed), tuple(sorted(deltas.items())))
        comp = (base.select("equipo", base_p="campeon_Elo")
                .join(esc.select("equipo", esc_p="campeon_Elo"), on="equipo")
                .with_columns((pl.col("esc_p") - pl.col("base_p")).alias("delta"))
                .sort("esc_p", descending=True).head(12))
        st.dataframe(
            comp.rename({"equipo": "Equipo", "base_p": "Campeon base",
                         "esc_p": "Campeon escenario", "delta": "Cambio"}),
            hide_index=True, width="stretch",
            column_config={"Campeon base": st.column_config.NumberColumn(format="percent"),
                           "Campeon escenario": st.column_config.NumberColumn(format="percent"),
                           "Cambio": st.column_config.NumberColumn(format="percent")})
    else:
        st.info("Selecciona al menos un equipo para crear un escenario.")


# ================= MOTORES + BACKTESTING =================
with tab_comp:
    st.subheader("Donde discrepan los motores")
    comp = base.head(12).select("equipo", "campeon_DC", "campeon_Elo").to_pandas().melt(
        id_vars="equipo", var_name="motor", value_name="prob")
    comp["motor"] = comp["motor"].map({"campeon_DC": "Dixon-Coles", "campeon_Elo": "Elo"})
    ch = (alt.Chart(comp).mark_bar()
          .encode(x=alt.X("prob:Q", axis=alt.Axis(format="%"), title="Prob. campeon"),
                  y=alt.Y("equipo:N", sort="-x", title=None),
                  color=alt.Color("motor:N", scale=alt.Scale(range=[NARANJA, AZUL])),
                  yOffset="motor:N").properties(height=380))
    st.altair_chart(ch, width="stretch")

    st.subheader("Backtesting out-of-sample (Mundiales 2014 / 2018 / 2022)")
    bt = get_backtest()
    if bt is not None:
        st.caption("Menor log-loss y Brier = mejor; mayor acierto = mejor. Elo gana en el promedio.")
        st.dataframe(bt.rename({"mundial": "Mundial", "motor": "Motor", "log_loss": "Log-loss",
                                "brier": "Brier", "acierto": "Acierto"}),
                     hide_index=True, width="stretch",
                     column_config={"Acierto": st.column_config.NumberColumn(format="percent")})
    else:
        st.info("Corre `uv run scripts/05_backtesting.py` para generar el backtesting.")
