"""CLI wrapper around src/qso_download.py.

Usage:
    python scripts/download_qso.py --data-root /path/to/cache
    python scripts/download_qso.py --data-root /path/to/cache \\
        --z-min 2.75 --z-max 3.25 --max-spectra 1000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qso_download import download_qso


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-root", required=True, help="Cache directory root")
    p.add_argument("--z-min", type=float, default=2.75)
    p.add_argument("--z-max", type=float, default=3.25)
    p.add_argument("--max-spectra", type=int, default=1000,
                   help="Cap on number of spectra to download (default: 1000)")
    args = p.parse_args()

    download_qso(
        args.data_root,
        z_min=args.z_min,
        z_max=args.z_max,
        max_spectra=args.max_spectra,
    )


if __name__ == "__main__":
    main()
