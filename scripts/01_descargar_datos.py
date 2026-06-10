"""
01 - Descarga el dataset historico de partidos internacionales y lo cachea a Parquet.

Fuente: https://github.com/martj42/international_results (CC BY 4.0)
Incluye todos los partidos desde 1872, mas los fixtures futuros del Mundial 2026.
"""
from pathlib import Path

import polars as pl
import requests

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

csv_path = RAW_DIR / "results.csv"
parquet_path = PROC_DIR / "partidos.parquet"


def main():
    print(f"Descargando {URL} ...")
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    csv_path.write_bytes(resp.content)
    print(f"  guardado en {csv_path} ({len(resp.content) / 1e6:.1f} MB)")

    # NA -> null en scores; tipar fecha
    df = pl.read_csv(
        csv_path,
        null_values=["NA", ""],
        schema_overrides={"home_score": pl.Int64, "away_score": pl.Int64},
    ).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))

    jugados = df.filter(pl.col("home_score").is_not_null())
    futuros = df.filter(pl.col("home_score").is_null())

    df.write_parquet(parquet_path)

    print(f"\nResumen:")
    print(f"  total partidos     : {df.height:,}")
    print(f"  jugados            : {jugados.height:,}")
    print(f"  fixtures futuros   : {futuros.height:,}")
    print(f"  rango fechas       : {df['date'].min()} -> {df['date'].max()}")
    print(f"  torneos distintos  : {df['tournament'].n_unique():,}")
    print(f"\n  parquet -> {parquet_path}")


if __name__ == "__main__":
    main()
