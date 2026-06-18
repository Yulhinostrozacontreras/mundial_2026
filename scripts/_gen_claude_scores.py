"""Genera data/claude_scores.csv: el SCORE CLAUDE, mi juicio de experto por
partido pendiente del Mundial 2026.

No es un modelo: es un pronostico cualitativo que combina (1) las senales del
modelo (Elo, Poisson, forma), (2) el valor de plantel y (3) CONTEXTO que el
modelo no ve -lesiones, suspensiones, escenarios de clasificacion, matchups,
el nivel goleador real del torneo- recogido de prensa al 18-jun-2026. Esta
hecho para DIFERIR del modelo donde el contexto lo amerita.

Las claves se normalizan (sin acentos, minusculas) para casar exacto con los
nombres en ingles de fixtures_2026.parquet (incluye 'Curacao', 'Ivory Coast').
"""
import _bootstrap  # noqa: F401
import unicodedata
from pathlib import Path

import polars as pl

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT = Path(__file__).resolve().parents[1] / "data" / "claude_scores.csv"


def norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


# (home, away, claude_home, claude_away, nota)  -- nota en ASCII, sin acentos
PICKS = [
    # --- Jornada 1 de K y L (17-jun) ---
    ("portugal", "dr congo", 2, 0, "Portugal clase superior; DRC fisico pero inferior"),
    ("uzbekistan", "colombia", 0, 2, "Colombia top-5 Elo, contundente fuera"),
    ("england", "croatia", 2, 1, "Inglaterra superior; Croacia veterana incomoda; torneo goleador"),
    ("ghana", "panama", 1, 0, "Ghana algo superior en un duelo cerrado"),
    # --- Jornada 2 ---
    ("czech republic", "south africa", 2, 1, "Chequia obligada tras perder; SA limitado"),
    ("mexico", "south korea", 1, 1, "DIFIERE: choque de lideres (ambos 3pts); Corea ordenada frena la localia"),
    ("switzerland", "bosnia and herzegovina", 2, 1, "Suiza solida; Bosnia con Dzeko siempre marca"),
    ("canada", "qatar", 2, 0, "DIFIERE a la baja: Canada anfitrion gana claro pero el 3-0 del modelo exagera"),
    ("scotland", "morocco", 1, 2, "Marruecos clase mundial pese al 1-1; Escocia sorprendio pero es inferior"),
    ("brazil", "haiti", 2, 0, "DIFIERE: Brasil gana pero SIN Neymar/Rodrygo/Militao/Estevao no golea como cree el modelo"),
    ("united states", "australia", 2, 1, "USA anfitrion en forma; Pulisic en duda; Australia dura"),
    ("turkey", "paraguay", 1, 1, "DIFIERE: ambos perdieron y se la juegan; duelo mas parejo de lo que da el modelo"),
    ("germany", "ivory coast", 2, 1, "Alemania con confianza tras 7-1; Costa de Marfil fisica gano su debut"),
    ("ecuador", "curacao", 3, 0, "Ecuador muy superior y obligado; Curacao ya encajo 7"),
    ("netherlands", "sweden", 2, 1, "Choque de lideres; Suecia pego 5-1 pero Oranje tiene mas plantel"),
    ("tunisia", "japan", 0, 2, "Japon clase asiatica; Tunez goleado 1-5"),
    ("belgium", "iran", 2, 1, "Belgica plantel; Iran ordenado empato su debut"),
    ("new zealand", "egypt", 1, 1, "Egipto con Salah algo superior pero NZ compite; reparto"),
    ("spain", "saudi arabia", 3, 0, "Espana obligada a golear tras el 0-0; Yamal disponible; Arabia goleable"),
    ("uruguay", "cape verde", 2, 0, "Uruguay clase sudamericana; Cabo Verde se atrinchera pero cede"),
    ("france", "iraq", 3, 0, "Francia plantel brutal con Saliba ok; Irak muy inferior"),
    ("norway", "senegal", 2, 1, "Noruega-Haaland en racha (4-1); Senegal fisico pero perdio; duelo por el liderato"),
    ("argentina", "austria", 2, 0, "Argentina campeona solida, Otamendi vuelve; Austria gano pero es inferior"),
    ("jordan", "algeria", 0, 2, "Argelia muy superior pese al 0-3 con Argentina; obligada a ganar"),
    # --- Jornada 3 (entran escenarios de clasificacion / rotaciones) ---
    ("portugal", "uzbekistan", 3, 0, "Portugal golea para asegurar primer puesto"),
    ("colombia", "dr congo", 2, 0, "Colombia clase, cierra como lider"),
    ("england", "ghana", 2, 0, "Inglaterra superior y enchufada"),
    ("panama", "croatia", 0, 2, "Croacia clase y oficio"),
    ("mexico", "czech republic", 2, 1, "Mexico anfitrion; Chequia jugandose la vida arriesga"),
    ("south africa", "south korea", 0, 2, "Corea superior; SA probablemente ya eliminado"),
    ("canada", "switzerland", 1, 1, "DIFIERE: duelo directo por clasificar; reparto plausible"),
    ("bosnia and herzegovina", "qatar", 2, 1, "Bosnia obligada; Qatar flojo defensivamente"),
    ("scotland", "brazil", 0, 2, "Brasil con bajas pero superior; Escocia se la juega y se expone"),
    ("morocco", "haiti", 3, 0, "Marruecos golea a un Haiti eliminado"),
    ("united states", "turkey", 2, 1, "USA anfitrion cierra fuerte en casa"),
    ("paraguay", "australia", 1, 1, "Ambos jugandose el pase; duelo parejo y tenso"),
    ("curacao", "ivory coast", 0, 3, "Costa de Marfil muy superior ante el colista"),
    ("ecuador", "germany", 1, 1, "DIFIERE: Ecuador solido atras; Alemania ya clasificada puede rotar"),
    ("japan", "sweden", 1, 1, "Choque por el liderato; dos equipos en forma, reparto"),
    ("tunisia", "netherlands", 0, 2, "Paises Bajos superior cierra primero"),
    ("egypt", "iran", 1, 1, "Parejo y ambos jugandose el pase"),
    ("new zealand", "belgium", 0, 2, "Belgica superior sentencia"),
    ("cape verde", "saudi arabia", 1, 1, "Duelo parejo entre dos que pelean el tercer puesto"),
    ("uruguay", "spain", 1, 2, "Espana clase y necesita el primer puesto; Uruguay vende cara la derrota"),
    ("norway", "france", 1, 2, "DIFIERE menos: Francia superior pero Noruega-Haaland en casa incomoda; duelo por liderato"),
    ("senegal", "iraq", 2, 0, "Senegal superior y obligado a ganar"),
    ("algeria", "austria", 1, 1, "Austria quiza ya clasificada; Argelia empuja; reparto"),
    ("jordan", "argentina", 0, 3, "Argentina golea pese a posible rotacion; Jordan eliminado"),
    ("colombia", "portugal", 1, 1, "Choque de lideres por el primer puesto, ambos quiza clasificados; reparto"),
    ("dr congo", "uzbekistan", 1, 1, "Dos selecciones probablemente eliminadas; duelo plano"),
    ("panama", "england", 0, 3, "Inglaterra golea para cerrar como lider"),
    ("croatia", "ghana", 2, 0, "Croacia clase cierra su grupo"),
]


def main():
    pick = {(norm(h), norm(a)): (ch, ca, n) for h, a, ch, ca, n in PICKS}
    fx = pl.read_parquet(PROC / "fixtures_2026.parquet")
    part = pl.read_parquet(PROC / "partidos.parquet")
    jugados = set(part.filter((pl.col("tournament") == "FIFA World Cup")
                              & (pl.col("date") >= pl.date(2026, 1, 1))
                              & pl.col("home_score").is_not_null())
                  .select("home_team", "away_team").rows())

    rows, faltan, usados = [], [], set()
    for h, a in fx.select("home_team", "away_team").rows():
        if (h, a) in jugados:
            continue  # ya jugado: no necesita score Claude
        k = (norm(h), norm(a))
        if k in pick:
            ch, ca, n = pick[k]
            rows.append({"home_team": h, "away_team": a,
                         "claude_home": ch, "claude_away": ca, "nota": n})
            usados.add(k)
        else:
            faltan.append((h, a))

    if faltan:
        print("SIN PICK (revisar nombres):")
        for h, a in faltan:
            print(f"  {h!r} vs {a!r}  -> norm {norm(h)!r}/{norm(a)!r}")
    sobran = set(pick) - usados
    if sobran:
        print("PICKS NO USADOS:", sobran)

    pl.DataFrame(rows).write_csv(OUT, include_header=True)
    print(f"\nEscrito {OUT} con {len(rows)} partidos (pendientes con score Claude).")


if __name__ == "__main__":
    main()
