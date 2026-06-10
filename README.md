# Mundial 2026 - Modelo Predictivo

Proyecto hobby para predecir el Mundial 2026 (USA / Canada / Mexico, 48 equipos).

## Objetivo

Estimar la fuerza de cada seleccion a partir de resultados historicos de partidos
internacionales y simular el torneo completo via Monte Carlo para obtener probabilidades
de: pasar de grupos, llegar a cada ronda y ser campeon.

## Enfoque

Se construyen y comparan DOS motores de prediccion:

1. **Poisson / Dixon-Coles**: modela goles por equipo (ataque/defensa). Interpretable y
   permite simular marcadores.
2. **Elo ponderado**: rating dinamico por equipo, simple y robusto.

Se valida cual predice mejor mediante backtesting sobre Mundiales pasados, y se usa el
ganador (o un ensemble) para la simulacion del bracket 2026.

## Datos

Dataset publico `martj42/international_results`: todos los partidos internacionales
desde 1872. Incluye ya los fixtures del Mundial 2026 (partidos con score nulo).

## Stack

Polars + DuckDB, Parquet como formato intermedio, `uv` para entorno.

## Pipeline (scripts/)

| Script | Que hace |
|--------|----------|
| `01_descargar_datos.py` | Descarga el CSV historico y lo cachea a Parquet |
| `02_explorar_fixtures_2026.py` | Extrae los 12 grupos y el calendario del Mundial 2026 |
| `03_modelo_elo.py` | Entrena Elo ponderado + calibracion ordinal de W/D/L |
| `04_modelo_poisson.py` | Entrena Dixon-Coles (MLE) sobre la ventana reciente |
| `05_backtesting.py` | Compara ambos motores en los Mundiales 2014/2018/2022 |
| `06_simulacion.py` | Monte Carlo del bracket 2026 -> probabilidades de campeon |

Logica compartida en `src/mundial/` (`datos.py`, `elo.py`, `dixon_coles.py`).

## Uso

```bash
uv sync
uv run scripts/01_descargar_datos.py
uv run scripts/02_explorar_fixtures_2026.py
uv run scripts/03_modelo_elo.py
uv run scripts/04_modelo_poisson.py
uv run scripts/05_backtesting.py
uv run scripts/06_simulacion.py
```

## Hallazgos

- **Backtesting (out-of-sample, 3 Mundiales):** Elo predice mejor el resultado
  W/D/L (log-loss 0.992 vs 1.013; acierto 55% vs 52%). Ambos superan al baseline.
- **Favoritos 2026:** Argentina y Espana co-favoritos. Dixon-Coles favorece a
  Argentina (eliminatorias CONMEBOL muy competitivas); Elo favorece a Espana
  (mejor rating actual). Brasil, Francia, Colombia e Inglaterra completan el grupo.

## Limitaciones conocidas

- El bracket de knockout se siembra por fuerza Elo, no usa la tabla oficial de
  cruce de los 8 mejores terceros de FIFA.
- La simulacion muestrea goles con Poisson independiente (ignora la correccion
  tau de Dixon-Coles, que solo afecta marcadores bajos).
