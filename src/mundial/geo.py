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
