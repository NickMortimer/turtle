"""Utility functions for GNSS base station file management."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def get_base_file_names(
    times: pd.Series,
    station: str,
    destination: Path | str,
) -> pd.DataFrame:
    """
    Generate GNSS base station file names and paths for download.

    Creates 15-minute high-rate and daily file entries for a time range.

    Parameters
    ----------
    times : pd.Series
        Time series of timestamps.
    station : str
        Station code (e.g., 'EXMT').
    destination : Path | str
        Destination directory for downloaded files.

    Returns
    -------
    pd.DataFrame
        DataFrame with path, file, and destination columns.
    """
    destination = Path(destination)
    starttime = times.floor("1H").min() - pd.Timedelta("1H")
    endtime = times.ceil("1H").max() + pd.Timedelta("1H")
    filerange = pd.date_range(starttime, endtime, freq="15MIN")

    files = [
        {
            "TimeStamp": item,
            "path": f"/rinex/highrate/{item.year}/{item.day_of_year:03d}/{item.hour:02d}",
            "destination": str(destination),
            "file": f"{station}_S_{item.year}{item.day_of_year:03d}{item.hour:02d}{item.minute:02d}_15M_01S_MO.crx.gz",
        }
        for item in filerange
    ]

    # Add daily files
    for day in times.floor("1D").unique():
        files.append(
            {
                "path": f"/rinex/daily/{day.year}/{day.day_of_year:03d}",
                "destination": str(destination),
                "file": f"{station}_R_{day.year}{day.day_of_year:03d}0000_01D_MN.rnx.gz",
            }
        )

    return pd.DataFrame(files)
