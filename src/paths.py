"""Centralized default paths for SCSI.

Environment variables:
    SCSI_DATA     — dataset cache root (default: ./data)
    SCSI_RESULTS  — training output root (default: ./results)

Drivers expose these as `--data_root` / `--results_root` argparse flags,
using `default_data_root()` / `default_results_root()` as the default value.
"""
import os


def default_data_root() -> str:
    return os.environ.get("SCSI_DATA", "./data")


def default_results_root() -> str:
    return os.environ.get("SCSI_RESULTS", "./results")
