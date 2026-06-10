"""Horarios oficiales de la fase de grupos del Mundial 2026 en hora de Peru (UTC-5).

Fuente: Wikipedia (paginas por grupo del torneo). Cada partido trae su hora local
con el offset UTC de la sede; aqui se convierte a hora de Peru. El emparejamiento
con los fixtures se hace por nombres normalizados (sin acentos), por lo que el
orden local/visita no importa.
"""
import unicodedata
from datetime import datetime, timedelta

PERU_OFFSET = -5  # Peru = UTC-5 todo el anio

# (local, visita, fecha ISO, hora local HH:MM, offset UTC de la sede en horas)
_RAW = [
    # Grupo A
    ("Mexico", "South Africa", "2026-06-11", "13:00", -6),
    ("South Korea", "Czech Republic", "2026-06-11", "20:00", -6),
    ("Czech Republic", "South Africa", "2026-06-18", "12:00", -4),
    ("Mexico", "South Korea", "2026-06-18", "19:00", -6),
    ("Czech Republic", "Mexico", "2026-06-24", "19:00", -6),
    ("South Africa", "South Korea", "2026-06-24", "19:00", -6),
    # Grupo B
    ("Canada", "Bosnia and Herzegovina", "2026-06-12", "15:00", -4),
    ("Qatar", "Switzerland", "2026-06-13", "12:00", -7),
    ("Switzerland", "Bosnia and Herzegovina", "2026-06-18", "12:00", -7),
    ("Canada", "Qatar", "2026-06-18", "15:00", -7),
    ("Switzerland", "Canada", "2026-06-24", "12:00", -7),
    ("Bosnia and Herzegovina", "Qatar", "2026-06-24", "12:00", -7),
    # Grupo C
    ("Brazil", "Morocco", "2026-06-13", "18:00", -4),
    ("Haiti", "Scotland", "2026-06-13", "21:00", -4),
    ("Scotland", "Morocco", "2026-06-19", "18:00", -4),
    ("Brazil", "Haiti", "2026-06-19", "20:30", -4),
    ("Scotland", "Brazil", "2026-06-24", "18:00", -4),
    ("Morocco", "Haiti", "2026-06-24", "18:00", -4),
    # Grupo D
    ("United States", "Paraguay", "2026-06-12", "18:00", -7),
    ("Australia", "Turkey", "2026-06-13", "21:00", -7),
    ("United States", "Australia", "2026-06-19", "12:00", -7),
    ("Turkey", "Paraguay", "2026-06-19", "20:00", -7),
    ("Turkey", "United States", "2026-06-25", "19:00", -7),
    ("Paraguay", "Australia", "2026-06-25", "19:00", -7),
    # Grupo E
    ("Germany", "Curacao", "2026-06-14", "12:00", -5),
    ("Ivory Coast", "Ecuador", "2026-06-14", "19:00", -4),
    ("Germany", "Ivory Coast", "2026-06-20", "16:00", -4),
    ("Ecuador", "Curacao", "2026-06-20", "19:00", -5),
    ("Curacao", "Ivory Coast", "2026-06-25", "16:00", -4),
    ("Ecuador", "Germany", "2026-06-25", "16:00", -4),
    # Grupo F
    ("Netherlands", "Japan", "2026-06-14", "15:00", -5),
    ("Sweden", "Tunisia", "2026-06-14", "20:00", -6),
    ("Netherlands", "Sweden", "2026-06-20", "12:00", -5),
    ("Tunisia", "Japan", "2026-06-20", "22:00", -6),
    ("Japan", "Sweden", "2026-06-25", "18:00", -5),
    ("Tunisia", "Netherlands", "2026-06-25", "18:00", -5),
    # Grupo G
    ("Belgium", "Egypt", "2026-06-15", "12:00", -7),
    ("Iran", "New Zealand", "2026-06-15", "18:00", -7),
    ("Belgium", "Iran", "2026-06-21", "12:00", -7),
    ("New Zealand", "Egypt", "2026-06-21", "18:00", -7),
    ("Egypt", "Iran", "2026-06-26", "20:00", -7),
    ("New Zealand", "Belgium", "2026-06-26", "20:00", -7),
    # Grupo H
    ("Saudi Arabia", "Uruguay", "2026-06-15", "18:00", -4),
    ("Spain", "Cape Verde", "2026-06-15", "12:00", -4),
    ("Spain", "Saudi Arabia", "2026-06-21", "12:00", -4),
    ("Uruguay", "Cape Verde", "2026-06-21", "18:00", -4),
    ("Cape Verde", "Saudi Arabia", "2026-06-26", "19:00", -5),
    ("Uruguay", "Spain", "2026-06-26", "18:00", -6),
    # Grupo I
    ("France", "Senegal", "2026-06-16", "15:00", -4),
    ("Iraq", "Norway", "2026-06-16", "18:00", -4),
    ("France", "Iraq", "2026-06-22", "17:00", -4),
    ("Norway", "Senegal", "2026-06-22", "20:00", -4),
    ("Norway", "France", "2026-06-26", "15:00", -4),
    ("Senegal", "Iraq", "2026-06-26", "15:00", -4),
    # Grupo J
    ("Argentina", "Algeria", "2026-06-16", "20:00", -5),
    ("Austria", "Jordan", "2026-06-16", "21:00", -7),
    ("Argentina", "Austria", "2026-06-22", "12:00", -5),
    ("Jordan", "Algeria", "2026-06-22", "20:00", -7),
    ("Algeria", "Austria", "2026-06-27", "21:00", -5),
    ("Jordan", "Argentina", "2026-06-27", "21:00", -5),
    # Grupo K
    ("Portugal", "DR Congo", "2026-06-17", "12:00", -5),
    ("Uzbekistan", "Colombia", "2026-06-17", "20:00", -6),
    ("Portugal", "Uzbekistan", "2026-06-23", "12:00", -5),
    ("Colombia", "DR Congo", "2026-06-23", "20:00", -6),
    ("Colombia", "Portugal", "2026-06-27", "19:30", -4),
    ("DR Congo", "Uzbekistan", "2026-06-27", "19:30", -4),
    # Grupo L
    ("England", "Croatia", "2026-06-17", "15:00", -5),
    ("Ghana", "Panama", "2026-06-17", "19:00", -4),
    ("England", "Ghana", "2026-06-23", "16:00", -4),
    ("Panama", "Croatia", "2026-06-23", "19:00", -4),
    ("Panama", "England", "2026-06-27", "17:00", -4),
    ("Croatia", "Ghana", "2026-06-27", "17:00", -4),
]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().strip().lower()


def _a_peru(fecha: str, hhmm: str, offset: int) -> datetime:
    """Hora local (en UTC+offset) -> hora de Peru (UTC-5)."""
    dt = datetime.strptime(f"{fecha} {hhmm}", "%Y-%m-%d %H:%M")
    return dt - timedelta(hours=offset) + timedelta(hours=PERU_OFFSET)


# clave: frozenset de nombres normalizados -> datetime en hora de Peru
HORARIOS = {frozenset((_norm(h), _norm(a))): _a_peru(f, t, off)
            for h, a, f, t, off in _RAW}


def hora_peru(home: str, away: str):
    """Devuelve el datetime del kickoff en hora de Peru, o None si no se encuentra."""
    return HORARIOS.get(frozenset((_norm(home), _norm(away))))
