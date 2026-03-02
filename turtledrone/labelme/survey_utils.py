"""Utility functions for survey processing and area calculation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gp
import numpy as np
import pandas as pd
import shapely.wkt
from shapely.geometry import MultiPoint

import turtledrone.config as config


def build_survey_names(
    survey_data: pd.DataFrame,
    drone_type: str,
    camera_type: str,
    country: str,
) -> pd.DataFrame:
    """
    Generate standardized file names for survey images.

    Parameters
    ----------
    survey_data : pd.DataFrame
        Survey data with SourceFile and id columns.
    drone_type : str
        Drone model identifier.
    camera_type : str
        Camera model identifier.
    country : str
        Country code.

    Returns
    -------
    pd.DataFrame
        DataFrame with Extension and NewName columns added.
    """
    output = survey_data.copy()
    output["Extension"] = output["SourceFile"].apply(
        lambda x: Path(x).suffix.upper()
    )
    output["Counter"] = 1
    output["Counter"] = output["Counter"].cumsum()

    output["NewName"] = output.apply(
        lambda row: f"{drone_type}_{camera_type}_{country}_"
        f"{row['id']}_{row.name.strftime('%Y%m%dT%H%M%S')}_"
        f"{row['Counter']:04d}{row['Extension']}",
        axis=1,
    )
    return output


def calculate_survey_area(gdf: gp.GeoDataFrame) -> float:
    """
    Calculate survey area using convex hull of image positions.

    Parameters
    ----------
    gdf : gp.GeoDataFrame
        GeoDataFrame with ImageEasting and ImageNorthing columns.

    Returns
    -------
    float
        Survey area in hectares.
    """
    points = np.dstack(
        (gdf["ImageEasting"].values, gdf["ImageNorthing"].values)
    )[0]
    hull = MultiPoint(points).convex_hull
    return hull.area / 10000  # Convert m² to hectares


def process_survey_data(
    survey_csv: Path,
) -> pd.DataFrame:
    """
    Load and prepare survey data with area calculation.

    Parameters
    ----------
    survey_csv : Path
        Path to survey CSV file.

    Returns
    -------
    pd.DataFrame
        Survey data with area added.
    """
    data = pd.read_csv(survey_csv, index_col="TimeStamp", parse_dates=["TimeStamp"])

    if len(data) == 0:
        return data

    crs = f"epsg:{int(data['UtmCode'].iloc[0])}"
    gdf = gp.GeoDataFrame(
        data,
        geometry=data.ImagePolygon.apply(shapely.wkt.loads),
        crs=crs,
    )

    gdf["SurveyAreaHec"] = calculate_survey_area(gdf) / 10000
    return gdf


def prepare_image_destinations(
    survey_data: pd.DataFrame,
    destination_dir: str | Path,
) -> pd.DataFrame:
    """
    Add destination file paths to survey data.

    Parameters
    ----------
    survey_data : pd.DataFrame
        Survey data with NewName column.
    destination_dir : str | Path
        Base destination directory.

    Returns
    -------
    pd.DataFrame
        Survey data with FileDest column added.
    """
    output = survey_data.copy()
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    output["FileDest"] = output["NewName"].apply(
        lambda x: destination_dir / x
    )
    return output


def file_destination_from_name(filename: str) -> Path:
    """
    Compute destination path from filename.

    Expected format: COUNTRY_SITE_SITECODE_*.

    Parameters
    ----------
    filename : str
        File name to parse.

    Returns
    -------
    Path
        Destination directory path.
    """
    return config.init().get_destination(filename)
