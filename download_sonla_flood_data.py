#!/usr/bin/env python3
"""Repository-local entrypoint for acquiring the Son La flood dataset."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_SRC = PROJECT_ROOT / "vmd_ardl_ffnn" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from vmd_ardl_ffnn.flood_acquisition import main


if __name__ == "__main__":
    main()
