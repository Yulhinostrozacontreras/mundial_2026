"""Mapeo de las 48 selecciones del Mundial 2026 a codigos ISO-3 (para el mapa).

England y Scotland comparten GBR (no tienen codigo ISO propio): en el mapa
mundial se pinta GBR con el valor del mas probable de los dos.
"""

ISO3 = {
    "Algeria": "DZA", "Argentina": "ARG", "Australia": "AUS", "Austria": "AUT",
    "Belgium": "BEL", "Bosnia and Herzegovina": "BIH", "Brazil": "BRA", "Canada": "CAN",
    "Cape Verde": "CPV", "Colombia": "COL", "Croatia": "HRV", "Curaçao": "CUW",
    "Czech Republic": "CZE", "DR Congo": "COD", "Ecuador": "ECU", "Egypt": "EGY",
    "England": "GBR", "France": "FRA", "Germany": "DEU", "Ghana": "GHA",
    "Haiti": "HTI", "Iran": "IRN", "Iraq": "IRQ", "Ivory Coast": "CIV",
    "Japan": "JPN", "Jordan": "JOR", "Mexico": "MEX", "Morocco": "MAR",
    "Netherlands": "NLD", "New Zealand": "NZL", "Norway": "NOR", "Panama": "PAN",
    "Paraguay": "PRY", "Portugal": "PRT", "Qatar": "QAT", "Saudi Arabia": "SAU",
    "Scotland": "GBR", "Senegal": "SEN", "South Africa": "ZAF", "South Korea": "KOR",
    "Spain": "ESP", "Sweden": "SWE", "Switzerland": "CHE", "Tunisia": "TUN",
    "Turkey": "TUR", "United States": "USA", "Uruguay": "URY", "Uzbekistan": "UZB",
}

# Por equipo: (nombre en espanol, codigo FIFA, ISO-2 para la bandera, grupo oficial FIFA).
# La bandera emoji se genera en runtime con bandera(); el archivo se mantiene en ASCII.
INFO = {
    "Algeria": ("Argelia", "ALG", "DZ", "J"),
    "Argentina": ("Argentina", "ARG", "AR", "J"),
    "Australia": ("Australia", "AUS", "AU", "D"),
    "Austria": ("Austria", "AUT", "AT", "J"),
    "Belgium": ("Bélgica", "BEL", "BE", "G"),
    "Bosnia and Herzegovina": ("Bosnia y Herzegovina", "BIH", "BA", "B"),
    "Brazil": ("Brasil", "BRA", "BR", "C"),
    "Canada": ("Canadá", "CAN", "CA", "B"),
    "Cape Verde": ("Cabo Verde", "CPV", "CV", "H"),
    "Colombia": ("Colombia", "COL", "CO", "K"),
    "Croatia": ("Croacia", "CRO", "HR", "L"),
    "Curaçao": ("Curazao", "CUW", "CW", "E"),
    "Czech Republic": ("Rep. Checa", "CZE", "CZ", "A"),
    "DR Congo": ("RD Congo", "COD", "CD", "K"),
    "Ecuador": ("Ecuador", "ECU", "EC", "E"),
    "Egypt": ("Egipto", "EGY", "EG", "G"),
    "England": ("Inglaterra", "ENG", "GB", "L"),
    "France": ("Francia", "FRA", "FR", "I"),
    "Germany": ("Alemania", "GER", "DE", "E"),
    "Ghana": ("Ghana", "GHA", "GH", "L"),
    "Haiti": ("Haití", "HAI", "HT", "C"),
    "Iran": ("Irán", "IRN", "IR", "G"),
    "Iraq": ("Irak", "IRQ", "IQ", "I"),
    "Ivory Coast": ("Costa de Marfil", "CIV", "CI", "E"),
    "Japan": ("Japón", "JPN", "JP", "F"),
    "Jordan": ("Jordania", "JOR", "JO", "J"),
    "Mexico": ("México", "MEX", "MX", "A"),
    "Morocco": ("Marruecos", "MAR", "MA", "C"),
    "Netherlands": ("Países Bajos", "NED", "NL", "F"),
    "New Zealand": ("Nueva Zelanda", "NZL", "NZ", "G"),
    "Norway": ("Noruega", "NOR", "NO", "I"),
    "Panama": ("Panamá", "PAN", "PA", "L"),
    "Paraguay": ("Paraguay", "PAR", "PY", "D"),
    "Portugal": ("Portugal", "POR", "PT", "K"),
    "Qatar": ("Catar", "QAT", "QA", "B"),
    "Saudi Arabia": ("Arabia Saudita", "KSA", "SA", "H"),
    "Scotland": ("Escocia", "SCO", "GB", "C"),
    "Senegal": ("Senegal", "SEN", "SN", "I"),
    "South Africa": ("Sudáfrica", "RSA", "ZA", "A"),
    "South Korea": ("Corea del Sur", "KOR", "KR", "A"),
    "Spain": ("España", "ESP", "ES", "H"),
    "Sweden": ("Suecia", "SWE", "SE", "F"),
    "Switzerland": ("Suiza", "SUI", "CH", "B"),
    "Tunisia": ("Túnez", "TUN", "TN", "F"),
    "Turkey": ("Turquía", "TUR", "TR", "D"),
    "United States": ("Estados Unidos", "USA", "US", "D"),
    "Uruguay": ("Uruguay", "URU", "UY", "H"),
    "Uzbekistan": ("Uzbekistán", "UZB", "UZ", "K"),
}


def bandera(iso2: str) -> str:
    """Emoji de bandera a partir del codigo ISO-2 (generado en runtime)."""
    return "".join(chr(0x1F1E6 + ord(c) - 65) for c in iso2.upper())


def info(equipo: str) -> dict:
    """Datos de presentacion de un equipo: nombre_es, cod, bandera, grupo."""
    es, cod, iso2, grp = INFO.get(equipo, (equipo, equipo[:3].upper(), "", "?"))
    return {"es": es, "cod": cod, "bandera": bandera(iso2) if iso2 else "", "grupo": grp}
