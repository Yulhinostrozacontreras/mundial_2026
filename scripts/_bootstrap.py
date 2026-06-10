"""Agrega src/ al path para importar el paquete mundial desde los scripts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
