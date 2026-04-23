"""Download SDSS DR16 quasar spectra and assemble into a cached numpy array.

Intended to be called from the dataset registry (`_fetch_qso` in
`custom_datasets.py`) and from the CLI wrapper `scripts/download_qso.py`.

Cache layout:
    {data_root}/qso/catalog/DR16Q_v4.fits
    {data_root}/qso/z-{zmin:.2f}-{zmax:.2f}_n-{N}/spectra.npy    # shape (N, 3, 2999)

The three channels along axis 1 are (loglam, flux, ivar).

Requires: astropy, numpy, tqdm.
"""

from pathlib import Path

import numpy as np


CATALOG_URL = "https://data.sdss.org/sas/dr16/eboss/qso/DR16Q/DR16Q_v4.fits"
SPECTRA_BASE = "https://data.sdss.org/sas/dr16/eboss/spectro/redux/v5_13_0/spectra/full/"

# Clipping window in log10(wavelength); this picks out exactly 2999 points
# for SDSS DR16 spectra (confirmed empirically in notebooks/qsos_for_chirag.ipynb).
LOGLAM_MIN = 3.56
LOGLAM_MAX = 3.86
N_POINTS = 2999


def _load_catalog(cache_dir: Path):
    from astropy.io import fits

    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / "DR16Q_v4.fits"
    if not local.exists():
        print(f"Downloading DR16 quasar catalog → {local}")
        with fits.open(CATALOG_URL) as hdul:
            hdul.writeto(local)
    return fits.open(local)[1].data


def _fetch_spectrum(plate, mjd, fiberid):
    from astropy.io import fits

    url = f"{SPECTRA_BASE}{plate}/spec-{plate}-{mjd}-{fiberid:04d}.fits"
    return fits.open(url)[1].data


def download_qso(data_root, z_min=2.75, z_max=3.25, max_spectra=1000):
    """Download, filter, and assemble QSO spectra.

    Returns the path to the saved (N, 3, N_POINTS) float32 numpy array.
    If the target file already exists, returns it without re-downloading.
    """
    data_root = Path(data_root)
    catalog_dir = data_root / "qso" / "catalog"
    out_dir = data_root / "qso" / f"z-{z_min:.2f}-{z_max:.2f}_n-{max_spectra}"
    out_path = out_dir / "spectra.npy"

    if out_path.exists():
        return out_path

    catalog = _load_catalog(catalog_dir)
    mask = (catalog["Z"] > z_min) & (catalog["Z"] < z_max) & (catalog["ZWARNING"] == 0)
    selected = catalog[mask]
    if max_spectra is not None and len(selected) > max_spectra:
        selected = selected[:max_spectra]
    print(f"Selected {len(selected)} catalog rows in z=({z_min}, {z_max})")

    from tqdm import tqdm

    records = []
    for row in tqdm(selected, desc="spectra"):
        try:
            spec = _fetch_spectrum(int(row["PLATE"]), int(row["MJD"]), int(row["FIBERID"]))
            loglam = np.asarray(spec["LOGLAM"])
            idx = np.where((loglam > LOGLAM_MIN) & (loglam < LOGLAM_MAX))[0]
            if len(idx) < N_POINTS:
                continue
            idx = idx[:N_POINTS]
            rec = np.stack(
                [np.asarray(spec["LOGLAM"])[idx],
                 np.asarray(spec["FLUX"])[idx],
                 np.asarray(spec["IVAR"])[idx]],
                axis=0,
            )
            records.append(rec)
        except Exception as e:
            print(f"  skipped plate={row['PLATE']} mjd={row['MJD']} fiberid={row['FIBERID']}: {e}")

    if not records:
        raise RuntimeError("No spectra successfully downloaded.")

    arr = np.stack(records, axis=0).astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    print(f"Saved {arr.shape} → {out_path}")
    return out_path
