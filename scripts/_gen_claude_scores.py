"""Genera data/claude_scores.csv: el SCORE CLAUDE, mi juicio de experto por
partido pendiente del Mundial 2026.

No es un modelo: es un pronostico cualitativo que combina (1) las senales del
modelo (Elo, Poisson, forma), (2) el valor de plantel y (3) CONTEXTO que el
modelo no ve -lesiones, suspensiones, escenarios de clasificacion, matchups,
el nivel goleador real del torneo- recogido de prensa al 24-jun-2026. Esta
hecho para DIFERIR del modelo donde el contexto lo amerita.

Actualizado al 24-jun (cerrada la J2): los picks de la J3 (ultima fecha de
grupos) incorporan los PUNTOS reales de cada grupo y los escenarios de
clasificacion -ya clasificados (Argentina, Colombia, Francia, Alemania,
Noruega, Mexico, USA) que pueden ROTAR; equipos a vida o muerte que arriesgan;
y eliminados sin nada en juego-. Formato 48: avanzan 2 por grupo + los 8
mejores terceros, asi que varios con 3 pts siguen peleando.

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
    # --- Jornada 3 (ultima fecha de grupos) -- con PUNTOS reales tras J2 y
    #     escenarios de clasificacion al 24-jun (avanzan 2 + 8 mejores terceros) ---
    # Grupo A (Mexico 6 clasif / Corea 3 / Chequia 1 / SA 1)
    ("mexico", "czech republic", 2, 1, "Mexico ya clasificado y lider; rota algo pero la localia del Azteca pesa; Chequia (1pt) obligada se expone"),
    ("south africa", "south korea", 1, 2, "Corea (3pts) avanza con un punto y es superior; Sudafrica (1pt) necesita ganar pero no le alcanza el plantel"),
    # Grupo B (Canada 4 / Suiza 4 / Bosnia 1 / Qatar 1)
    ("canada", "switzerland", 1, 1, "Duelo por el 1er puesto; a ambos (4pts) un empate los deja comodos rumbo a octavos; reparto plausible"),
    ("bosnia and herzegovina", "qatar", 2, 0, "Bosnia (1pt) obligada y con Dzeko marca; Qatar (1pt, DG-6) ya goleado y sin opciones"),
    # Grupo C (Brasil 4 / Marruecos 4 / Escocia 3 / Haiti 0)
    ("scotland", "brazil", 1, 2, "DIFIERE: Brasil DIEZMADO (Neymar/Rodrygo/Estevao fuera) no golea pero su oficio alcanza; Escocia obligada arriesga y se expone"),
    ("morocco", "haiti", 2, 0, "Marruecos (4pts) gana y asegura el grupo; Haiti (0pts) ya eliminado"),
    # Grupo D (USA 6 clasif / Australia 3 / Paraguay 3 / Turquia 0)
    ("united states", "turkey", 2, 1, "USA ya clasificado y anfitrion cierra en casa aun rotando; Turquia (0pts) eliminada descuenta"),
    ("paraguay", "australia", 1, 1, "DIFIERE: final por el 2do puesto (ambos 3pts); el empate deja a los dos vivos como terceros; duelo tenso y parejo"),
    # Grupo E (Alemania 6 clasif / Costa Marfil 3 / Ecuador 1 / Curacao 1)
    ("curacao", "ivory coast", 0, 2, "Costa de Marfil (3pts) gana y clasifica; Curacao (1pt, DG-6) colista ya casi sin opciones"),
    ("ecuador", "germany", 1, 1, "DIFIERE: Alemania ya clasificada como lider ROTA (sin Schlotterbeck); Ecuador solido atras araña el punto que pelea por tercero"),
    # Grupo F (Holanda 4 / Japon 4 / Suecia 3 / Tunez 0)
    ("japan", "sweden", 2, 1, "Choque por el liderato entre dos goleadores; Japon (4pts) en forma se impone; Suecia (3pts) cae a pelear tercero"),
    ("tunisia", "netherlands", 0, 2, "Paises Bajos (4pts) cierra arriba aun sin Timber; Tunez (0pts, DG-8) eliminado y goleado"),
    # Grupo G (Egipto 4 / Iran 2 / Belgica 2 / NZ 1 -- el mas apretado)
    ("egypt", "iran", 1, 1, "Grupo apretado: a Egipto (4pts) con Salah el empate le basta para avanzar; Iran (2pts) empuja y se reparten"),
    ("new zealand", "belgium", 1, 2, "Belgica (2pts) OBLIGADA a ganar para no firmar un fracaso; NZ (1pt) pelea pero el plantel belga resuelve"),
    # Grupo H (Espana 4 / Uruguay 2 / Cabo Verde 2 / Arabia 1)
    ("cape verde", "saudi arabia", 1, 0, "Final por el tercer puesto; Cabo Verde (2pts) se atrinchera y golpea; Arabia (1pt, DG-4) necesitaba mas"),
    ("uruguay", "spain", 1, 2, "Espana (4pts) con Yamal apto va por el 1er puesto y su clase pesa; Uruguay (2pts) vende cara su ultima opcion"),
    # Grupo I (Francia 6 clasif / Noruega 6 clasif / Senegal 0 / Iraq 0)
    ("norway", "france", 1, 2, "Ambos ya clasificados: duelo por el 1er puesto (Haaland vs Mbappe); Francia tiene mas plantel y se queda el grupo"),
    ("senegal", "iraq", 2, 0, "Ambos eliminados (0pts) juegan por orgullo; Senegal es muy superior"),
    # Grupo J (Argentina 6 clasif / Austria 3 / Argelia 3 / Jordania 0)
    ("algeria", "austria", 1, 1, "DIFIERE: final por el 2do puesto (ambos 3pts); duelo tenso, el empate los deja a ambos peleando como terceros"),
    ("jordan", "argentina", 0, 2, "Argentina ya clasificada ROTA (Messi puede descansar); gana con comodidad sin golear; Jordania (0pts) eliminada"),
    # Grupo K (Colombia 6 clasif / Portugal 4 / DR Congo 1 / Uzbekistan 0)
    ("colombia", "portugal", 1, 1, "Choque por el 1er puesto entre dos ya practicamente clasificados; ambos pueden especular; reparto plausible"),
    ("dr congo", "uzbekistan", 1, 0, "DR Congo (1pt) mantiene una minima chance de tercero y gana ajustado; Uzbekistan (0pts, DG-7) eliminado"),
    # Grupo L (Inglaterra 4 / Ghana 4 / Croacia 3 / Panama 0)
    ("panama", "england", 0, 2, "Inglaterra (4pts) cierra como lider; Panama (0pts) ya eliminado"),
    ("croatia", "ghana", 2, 1, "Croacia (3pts) OBLIGADA a ganar saca su clase y oficio; Ghana (4pts) avanzaba con el empate pero cae; Modric manda"),
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
