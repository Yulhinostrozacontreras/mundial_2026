"""06 - Simulacion Monte Carlo del Mundial 2026 (hasta el campeon).

Usa la logica compartida en mundial.torneo (la misma que el Streamlit), que:
  - Poisson: marcadores via Poisson -> tabla de grupos con desempate por
    diferencia y goles a favor; knockout con penales si hay empate.
  - Elo: resultado W/D/L muestreado de la calibracion ordinal.
  - FIJA los partidos de grupo YA JUGADOS con su marcador real (las predicciones
    se condicionan a lo ocurrido); solo simula lo que falta.
Reglas Mundial 2026: 12 grupos de 4, avanzan 1o, 2o y los 8 mejores 3os (32 a
knockout). Bracket sembrado por fuerza (limitacion: no usa la tabla oficial de
cruce de terceros de FIFA).
"""
import _bootstrap  # noqa: F401
from pathlib import Path

from mundial import torneo

N_SIMS = 20000
SEED = 20260611
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"

ins = torneo.cargar_insumos()
print(f"Partidos de grupo ya jugados (fijados con marcador real): {len(ins['jugados'])}")
print(f"Simulando {N_SIMS:,} torneos con cada motor...")
sim = torneo.simular(ins, n_sims=N_SIMS, seed=SEED)
tabla = torneo.tabla_resultados(ins, sim)
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
