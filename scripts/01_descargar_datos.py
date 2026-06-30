"""
01 - Descarga el dataset historico de partidos internacionales y lo cachea a Parquet.

Fuente: https://github.com/martj42/international_results (CC BY 4.0)
Incluye todos los partidos desde 1872, mas los fixtures futuros del Mundial 2026.
"""
from pathlib import Path

import polars as pl
import requests

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
URL_SHOOTOUTS = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
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

    # Resultados PROVISIONALES: partidos ya jugados que martj42 aun no publica
    # (lag ~1 dia). Se aplican SOLO a fixtures sin marcador, asi cuando martj42
    # los traiga su marcador real prevalece y el override deja de tener efecto
    # (auto-limpiante). Sirve para ver el bracket al dia sin esperar al dataset.
    prov_path = PROC_DIR.parent / "resultados_provisionales.csv"
    n_prov = 0
    if prov_path.exists():
        prov = (pl.read_csv(prov_path)
                .select("home_team", "away_team",
                        pl.col("home_score").alias("hs_ov"),
                        pl.col("away_score").alias("as_ov")))
        df = df.join(prov, on=["home_team", "away_team"], how="left")
        aplicar = pl.col("hs_ov").is_not_null() & pl.col("home_score").is_null()
        n_prov = int(df.filter(aplicar).height)
        df = df.with_columns(
            pl.when(aplicar).then(pl.col("hs_ov")).otherwise(pl.col("home_score")).alias("home_score"),
            pl.when(aplicar).then(pl.col("as_ov")).otherwise(pl.col("away_score")).alias("away_score"),
        ).drop("hs_ov", "as_ov")
        print(f"  resultados provisionales aplicados: {n_prov}")

    jugados = df.filter(pl.col("home_score").is_not_null())
    futuros = df.filter(pl.col("home_score").is_null())

    df.write_parquet(parquet_path)

    # Penales (define quien avanza en los empates de eliminatoria)
    rs = requests.get(URL_SHOOTOUTS, timeout=60)
    rs.raise_for_status()
    (RAW_DIR / "shootouts.csv").write_bytes(rs.content)
    sh = (pl.read_csv(RAW_DIR / "shootouts.csv")
          .with_columns(pl.col("date").str.to_date("%Y-%m-%d")))
    sh.write_parquet(PROC_DIR / "shootouts.parquet")
    print(f"  shootouts (penales) -> {sh.height:,} registros")

    print(f"\nResumen:")
    print(f"  total partidos     : {df.height:,}")
    print(f"  jugados            : {jugados.height:,}")
    print(f"  fixtures futuros   : {futuros.height:,}")
    print(f"  rango fechas       : {df['date'].min()} -> {df['date'].max()}")
    print(f"  torneos distintos  : {df['tournament'].n_unique():,}")
    print(f"\n  parquet -> {parquet_path}")


if __name__ == "__main__":
    main()
