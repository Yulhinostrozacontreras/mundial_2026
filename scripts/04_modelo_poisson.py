"""04 - Entrena Poisson sobre la ventana reciente y guarda parametros."""
import _bootstrap  # noqa: F401
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from mundial.datos import cargar_jugados
from mundial import dixon_coles as dc

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
DESDE = date(2014, 1, 1)
INICIO_WC = date(2026, 6, 1)  # los partidos del Mundial 2026 reciben peso extra
BOOST_WC = 2.0               # ~2x vs un amistoso/clasificatorio de la misma fecha

df = cargar_jugados()
# ventana hasta el dia siguiente del ultimo resultado: el Poisson YA ve el Mundial
REF = df["date"].max() + timedelta(days=1)
prep = dc.preparar(df, desde=DESDE, fecha_ref=REF, half_life_dias=730, min_partidos=20,
                   torneo_boost=BOOST_WC, boost_desde=INICIO_WC)
print(f"Ventana {DESDE} -> {REF}: {len(prep['hs']):,} partidos, {prep['n']} equipos "
      f"(Mundial 2026 con boost x{BOOST_WC})")

print("Ajustando Poisson (MLE)...")
modelo = dc.ajustar(prep, maxiter=500)
print(f"  base={modelo['base']:.3f}  home_adv={modelo['home']:.3f}  rho={modelo['rho']:.3f}")

# guardar parametros (numpy -> listas)
salida = dict(base=modelo["base"], home=modelo["home"], rho=modelo["rho"],
              equipos=modelo["equipos"],
              att=modelo["att"].tolist(), deff=modelo["deff"].tolist())
(PROC / "dc_params.json").write_text(json.dumps(salida), encoding="utf-8")

# fuerza neta = ataque + defensa
fuerza = modelo["att"] + modelo["deff"]
orden = np.argsort(-fuerza)
print("\nTop 20 por fuerza neta (ataque + defensa):")
for r, i in enumerate(orden[:20], 1):
    print(f"  {r:2d}. {modelo['equipos'][i]:<22} "
          f"fuerza={fuerza[i]:+.2f}  (atq={modelo['att'][i]:+.2f}, def={modelo['deff'][i]:+.2f})")
print(f"\nGuardado: dc_params.json ({prep['n']} equipos)")
