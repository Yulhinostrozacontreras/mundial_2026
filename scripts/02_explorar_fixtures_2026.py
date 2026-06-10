"""
02 - Extrae los fixtures del Mundial 2026 y deriva los 12 grupos.

La fase de grupos son 72 partidos (12 grupos x 4 equipos, round-robin = 6 c/u).
Los grupos no vienen etiquetados, pero se derivan: dos equipos estan en el mismo
grupo si tienen un partido entre si. Se reconstruyen como componentes conexas.
"""
from pathlib import Path

import polars as pl

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
df = pl.read_parquet(PROC_DIR / "partidos.parquet")

# Fase de grupos = los primeros 72 partidos del Mundial 2026 (jugados o no).
# Filtrar solo por home_score nulo romperia los grupos conforme se disputan los
# partidos (el dataset llena los marcadores), por eso se toman todos por fecha.
wc = df.filter(
    (pl.col("tournament") == "FIFA World Cup")
    & (pl.col("date") >= pl.date(2026, 1, 1))
).select("date", "home_team", "away_team", "city", "country").sort("date").head(72)

print(f"Fixtures Mundial 2026 (fase de grupos): {wc.height} partidos\n")

# Construir adyacencia y derivar grupos como componentes conexas
aristas = wc.select("home_team", "away_team").rows()
adj: dict[str, set[str]] = {}
for a, b in aristas:
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)

visitados: set[str] = set()
grupos: list[list[str]] = []
for equipo in sorted(adj):
    if equipo in visitados:
        continue
    # BFS sobre la componente
    comp, pila = set(), [equipo]
    while pila:
        n = pila.pop()
        if n in visitados:
            continue
        visitados.add(n)
        comp.add(n)
        pila.extend(adj[n] - visitados)
    grupos.append(sorted(comp))

grupos.sort()
print(f"Equipos totales: {len(visitados)}  |  Grupos derivados: {len(grupos)}\n")

letras = "ABCDEFGHIJKL"
filas = []
for i, g in enumerate(grupos):
    etiqueta = letras[i] if i < len(letras) else str(i)
    print(f"Grupo {etiqueta} ({len(g)}): {', '.join(g)}")
    for eq in g:
        filas.append({"grupo": etiqueta, "equipo": eq})

# Guardar grupos para el pipeline de simulacion
out = pl.DataFrame(filas)
out.write_parquet(PROC_DIR / "grupos_2026.parquet")
wc.write_parquet(PROC_DIR / "fixtures_2026.parquet")
print(f"\nGuardado: grupos_2026.parquet ({out.height} equipos) y fixtures_2026.parquet")
