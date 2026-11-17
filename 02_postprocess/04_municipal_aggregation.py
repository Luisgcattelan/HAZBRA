#!/usr/bin/env python
"""Aggregate the gridded products to the 5,570 IBGE municipalities (2024 mesh).

Source: paper_analysis/municipal_extraction.py + 10.Municipios.ipynb; code in hazbra/municipal.py.
Area-weighted zonal statistics with a 25 km buffer / nearest-pixel fallback for small municipalities;
every value records the method used. Run after steps 1-2 (needs the NetCDF product).

    python 02_postprocess/04_municipal_aggregation.py                      # all indicators
    python 02_postprocess/04_municipal_aggregation.py --indicators HI TXX
    python 02_postprocess/04_municipal_aggregation.py --test               # TXX only, written below _test_output/
All options of hazbra.municipal are accepted (see --help).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hazbra.municipal import main

if __name__ == "__main__":
    main()
