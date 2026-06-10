"""03 - Entrena el modelo Elo sobre toda la historia y guarda ratings."""
import _bootstrap  # noqa: F401
from pathlib import Path

import polars as pl

from mundial.datos import cargar_jugados
from mundial.elo import correr_elo, calibrar_ordinal

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"

df = cargar_jugados()
print(f"Partidos para entrenar Elo: {df.height:,}")

ratings, historial = correr_elo(df)
s, theta = calibrar_ordinal(historial)
print(f"Calibracion ordinal -> escala s={s:.1f}, umbral empate theta={theta:.3f}")

tabla = (pl.DataFrame({"equipo": list(ratings.keys()),
                       "elo": list(ratings.values())})
         .sort("elo", descending=True))
tabla.write_parquet(PROC / "elo_ratings.parquet")

# guardar calibracion
pl.DataFrame({"s": [s], "theta": [theta]}).write_parquet(PROC / "elo_calibracion.parquet")

print("\nTop 20 selecciones por Elo:")
for i, (eq, elo) in enumerate(tabla.head(20).rows(), 1):
    print(f"  {i:2d}. {eq:<22} {elo:7.1f}")
print(f"\nGuardado: elo_ratings.parquet ({tabla.height} equipos)")
