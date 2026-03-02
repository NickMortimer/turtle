"""Utility functions for EXIF data processing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


EXIF_COLUMNS = {
    "SourceFile",
    "FileModifyDate",
    "ImageDescription",
    "ExposureTime",
    "FNumber",
    "ExposureProgram",
    "ISO",
    "DateTimeOriginal",
    "Make",
    "SpeedX",
    "SpeedY",
    "SpeedZ",
    "Pitch",
    "Yaw",
    "Roll",
    "CameraPitch",
    "CameraYaw",
    "CameraRoll",
    "ExifImageWidth",
    "ExifImageHeight",
    "SerialNumber",
    "GPSLatitudeRef",
    "GPSLongitudeRef",
    "GPSAltitudeRef",
    "AbsoluteAltitude",
    "RelativeAltitude",
    "GimbalRollDegree",
    "GimbalYawDegree",
    "GimbalPitchDegree",
    "FlightRollDegree",
    "FlightYawDegree",
    "FlightPitchDegree",
    "CamReverse",
    "GimbalReverse",
    "CalibratedFocalLength",
    "CalibratedOpticalCenterX",
    "CalibratedOpticalCenterY",
    "ImageWidth",
    "ImageHeight",
    "GPSAltitude",
    "GPSLatitude",
    "GPSLongitude",
    "CircleOfConfusion",
    "FOV",
    "Latitude",
    "Longitude",
    "SubSecDateTimeOriginal",
    "FlightYSpeed",
    "FlightXSpeed",
    "Orientation",
    "ShutterSpeedValue",
    "ApertureValue",
    "WhiteBalance",
    "RtkFlag",
    "DewarpData",
    "DewarpFlag",
    "Model",
}


def parse_gps_coordinate(gps_string: str) -> float:
    """
    Parse GPS coordinate string to decimal degrees.

    Parameters
    ----------
    gps_string : str
        GPS coordinate in format "D M' S""

    Returns
    -------
    float
        Decimal degrees.
    """
    parts = gps_string.split()
    degrees = float(parts[0])
    minutes = float(parts[2].rstrip("'"))
    seconds = float(parts[3].rstrip('"'))
    return degrees + minutes / 60 + seconds / 3600


def process_exif_json(
    json_data: pd.DataFrame,
    catalog_dir: Path,
) -> pd.DataFrame:
    """
    Process raw EXIF JSON data into structured CSV format.

    Parameters
    ----------
    json_data : pd.DataFrame
        Raw EXIF data from exiftool.
    catalog_dir : Path
        Catalog directory for relative path computation.

    Returns
    -------
    pd.DataFrame
        Processed EXIF data.
    """
    if len(json_data) == 0 or "GPSLongitude" not in json_data.columns:
        return json_data

    output = json_data.copy()

    # Parse GPS coordinates
    output["Longitude"] = np.nan
    output["Latitude"] = np.nan

    gps_lon = output[~output["GPSLongitude"].isna()]
    if len(gps_lon) > 0:
        output.loc[gps_lon.index, "Longitude"] = (
            gps_lon["GPSLongitude"]
            .str.split(" ", expand=True)
            .apply(parse_gps_coordinate, axis=1)
        )

    gps_lat = output[~output["GPSLatitude"].isna()]
    if len(gps_lat) > 0:
        output.loc[gps_lat.index, "Latitude"] = (
            gps_lat["GPSLatitude"]
            .str.split(" ", expand=True)
            .apply(parse_gps_coordinate, axis=1)
        )

    # Apply hemisphere corrections
    output.loc[
        output["GPSLatitudeRef"] == "South",
        "Latitude",
    ] = output.loc[
        output["GPSLatitudeRef"] == "South",
        "Latitude",
    ] * (
        -1
    )
    output.loc[
        output["GPSLongitudeRef"] == "West",
        "Longitude",
    ] = output.loc[
        output["GPSLongitudeRef"] == "West",
        "Longitude",
    ] * (
        -1
    )

    # Filter to wanted columns
    output = output[output.columns[output.columns.isin(EXIF_COLUMNS)]]

    # Parse timestamps
    if "SubSecDateTimeOriginal" in output.columns:
        output["TimeStamp"] = pd.to_datetime(
            output.SubSecDateTimeOriginal,
            format="%Y:%m:%d %H:%M:%S.%f",
        )
    else:
        output["TimeStamp"] = pd.to_datetime(
            output.DateTimeOriginal,
            format="%Y:%m:%d %H:%M:%S",
        )

    # Compute relative source path
    output["SourceRel"] = output.SourceFile.apply(
        lambda x: Path(x).relative_to(catalog_dir)
    )

    # Extract sequence number
    output["Sequence"] = (
        output.SourceFile.str.extract("(?P<Sequence>\\d+)\\.(jpg|JPG)", expand=False)["Sequence"]
    )

    # Parse dewarp calibration data
    if "DewarpData" in output.columns and output["DewarpData"].notna().any():
        output[
            [
                "CalibrationDate",
                "CalibratedFocalLengthX",
                "CalibratedFocalLengthY",
                "CalibratedOpticalCenterX",
                "CalibratedOpticalCenterY",
                "K1",
                "K2",
                "P1",
                "P2",
                "K3",
            ]
        ] = output["DewarpData"].str.split(r"[;,]", expand=True)

        calibration_cols = [
            "CalibratedFocalLengthX",
            "CalibratedFocalLengthY",
            "CalibratedOpticalCenterX",
            "CalibratedOpticalCenterY",
            "K1",
            "K2",
            "P1",
            "P2",
            "K3",
        ]
        output[calibration_cols] = output[calibration_cols].astype(float)

        output["CalibratedOpticalCenterX"] = (
            output["ImageWidth"] / 2 + output["CalibratedOpticalCenterX"]
        )
        output["CalibratedOpticalCenterY"] = (
            output["ImageHeight"] / 2 + output["CalibratedOpticalCenterY"]
        )

    elif "DewarpFlag" not in output.columns:
        # Default calibration for P4 Pro
        output[
            ["CalibratedOpticalCenterX", "CalibratedOpticalCenterY", "CalibratedFocalLength"]
        ] = [2432.0, 1824.0, 3666.665]

    return output
