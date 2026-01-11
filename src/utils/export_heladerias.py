"""
Script para descargar heladerías desde OSM alrededor de una ciudad o coordenadas y guardarlas como CSV.
Usa el radio en metros.
"""
import argparse
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd

from utils.graph_utils import get_ice_cream_places


def main():
    parser = argparse.ArgumentParser(description="Exporta heladerías (amenity=ice_cream) a CSV desde OSM.")
    parser.add_argument("--city", help="Ciudad o punto de interés (ej: 'Plaza Independencia, Mendoza')", default=None)
    parser.add_argument("--lat", type=float, help="Latitud (si no se usa city)", default=None)
    parser.add_argument("--lon", type=float, help="Longitud (si no se usa city)", default=None)
    parser.add_argument("--radius", type=int, help="Radio en metros", default=1000)
    parser.add_argument("--output", type=Path, help="Ruta de salida CSV", default=Path("data/heladerias.csv"))
    parser.add_argument("--cache-dir", type=Path, help="Directorio de cache OSMnx", default=Path("cache"))
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    df = get_ice_cream_places(args.city, args.lat, args.lon, args.radius, cache_dir=str(args.cache_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Total heladerías: {len(df)}")
    print(f"CSV guardado en: {args.output}")


if __name__ == "__main__":
    main()
