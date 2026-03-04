"""Reusable processing functions for the turtle counting pipeline."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import cameratransform as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.wkt
from sklearn.cluster import MeanShift
from pyproj import Proj


DEFAULT_FOCAL_LENGTH = 3666.6


def load_and_prepare_points(
    location_file: str | Path,
    json_file: str | Path,
) -> pd.DataFrame:
    """Load and join location metadata with LabelMe shapes data."""
    camera = pd.read_csv(location_file, index_col="Key")
    points = pd.read_csv(json_file).dropna(how="all", axis=1)
    points["Key"] = points.FilePath.apply(lambda x: Path(x).with_suffix("").name)
    points = points.set_index("Key").join(camera)
    if len(points) == 0:
        return points

    points = points[~points.label.isin(["done", "don,e"])]
    points.points = points.points.apply(ast.literal_eval)

    if "CalibratedFocalLength" not in points.columns:
        points["CalibratedFocalLength"] = DEFAULT_FOCAL_LENGTH
        points["CalibratedOpticalCenterX"] = points["ImageWidth"] / 2
        points["CalibratedOpticalCenterY"] = points["ImageHeight"] / 2

    points.loc[
        points.CalibratedFocalLength.isna(),
        "CalibratedFocalLength",
    ] = DEFAULT_FOCAL_LENGTH

    return points[points.points.apply(len) > 0]


def calculate_positions(points: pd.DataFrame) -> pd.DataFrame:
    """Calculate world coordinates for each labeled point."""
    if len(points) == 0:
        return points
    return points.apply(_calculate_realworld_row, axis=1)


def _calculate_realworld_row(item: pd.Series) -> pd.Series:
    """Calculate world position columns for a single annotation row."""
    if "K1" in item.keys():
        cam = ct.Camera(
            ct.RectilinearProjection(
                focallength_x_px=item.CalibratedFocalLengthX,
                focallength_y_px=item.CalibratedFocalLengthY,
                center_x_px=item.CalibratedOpticalCenterX,
                center_y_px=item.CalibratedOpticalCenterY,
            ),
            orientation=ct.SpatialOrientation(
                tilt_deg=item.GimbalPitchDegree,
                elevation_m=item.RelativeAltitude,
                roll_deg=item.GimbalRollDegree,
                heading_deg=item.GimbalYawDegree,
            ),
            lens=ct.BrownLensDistortion(item.K1, item.K2, item.K3),
        )
    else:
        cam = ct.Camera(
            ct.RectilinearProjection(
                focallength_px=item.CalibratedFocalLength,
                center_x_px=item.CalibratedOpticalCenterX,
                center_y_px=item.CalibratedOpticalCenterY,
            ),
            orientation=ct.SpatialOrientation(
                tilt_deg=item.GimbalPitchDegree,
                elevation_m=item.RelativeAltitude,
                roll_deg=item.GimbalRollDegree,
                heading_deg=item.GimbalYawDegree,
            ),
        )

    cam.setGPSpos(item.Latitude, item.Longitude)
    point = np.mean(np.stack(item.points), axis=0)
    delta = cam.spaceFromImage(point)
    pos = cam.gpsFromImage(point)

    item["PointLatitude"] = pos[0]
    item["PointLongitude"] = pos[1]
    item["PointNorthing"] = item["ImageNorthing"] + delta[1]
    item["PointEasting"] = item["ImageEasting"] + delta[0]
    return item


def process_turtle_clusters(
    drone: pd.DataFrame,
    plotpath: str | Path,
    bandwidth: float = 10.0,
    time_gap_seconds: float = 12.0,
) -> pd.DataFrame:
    """Run turtle filtering, temporal grouping, and MeanShift clustering."""
    filtered = drone[
        drone.label.isin(["turtle_jbs", "turtle_surface"])
        & ~drone.ImagePolygon.isna()
        & ~drone.PointEasting.isna()
    ]

    if len(filtered) == 0:
        return filtered

    filtered = filtered.copy()
    filtered["ImagePolygon"] = filtered.ImagePolygon.apply(shapely.wkt.loads)
    filtered.sort_values("TimeStamp", inplace=True)
    filtered["groups"] = 0
    filtered.loc[
        abs(filtered.TimeStamp.diff().dt.total_seconds()) > time_gap_seconds,
        "groups",
    ] = 1
    filtered["groups"] = filtered["groups"].cumsum()

    tcounts = pd.DataFrame(
        filtered.groupby("groups").apply(
            lambda grp: _count_turtles_in_group(
                grp,
                plotpath,
                bandwidth,
                int(grp.name),
            ),
            include_groups=False,
        ),
        columns=["turtle_count"],
    )

    counts = tcounts.turtle_count.apply(lambda x: x["count"]).reset_index()
    centers = tcounts.turtle_count.apply(lambda x: x["centers"]).reset_index()
    centers.name = "centers"
    counts.name = "counts"

    output = pd.merge(filtered, counts, on=["groups"])
    output = pd.merge(output, centers, on=["groups"])
    return output


def _count_turtles_in_group(
    grp: pd.DataFrame,
    plotpath: str | Path,
    bandwidth: float,
    group_id: int,
) -> dict:
    """Cluster a temporal group and return cluster count and centers."""
    clustering = MeanShift(bandwidth=bandwidth).fit(
        np.dstack([grp.PointEasting.values, grp.PointNorthing.values])[0]
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    for _, row in grp.iterrows():
        x, y = row.ImagePolygon.exterior.xy
        ax.plot(
            x,
            y,
            color="#6699cc",
            alpha=0.7,
            linewidth=3,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(row.PointEasting, row.PointNorthing, marker="x", linestyle="")
    ax.set_aspect(1)
    plt.scatter(
        clustering.cluster_centers_[:, 0],
        clustering.cluster_centers_[:, 1],
        c="red",
        s=50,
    )
    plt.savefig(Path(plotpath) / f"group_plot_{group_id}")
    plt.close()
    return {
        "count": len(clustering.cluster_centers_),
        "centers": clustering.cluster_centers_,
    }


def from_np_array(array_string: str) -> np.ndarray:
    """Convert serialized numpy array strings back to numpy arrays."""
    compact = ",".join(array_string.replace("[ ", "[").split())
    return np.array(ast.literal_eval(compact))


def calculate_turtle_totals(drone: pd.DataFrame) -> pd.DataFrame:
    """Aggregate clustered turtle detections into survey-level points."""

    def process_survey_group(grp: pd.DataFrame) -> pd.DataFrame:
        crs = f"epsg:{int(grp['UtmCode'].min())}"
        utmproj = Proj(crs)
        points = np.mean(np.vstack(grp.turtle_count_y), axis=0)
        data = pd.DataFrame([points], columns=["Easting", "Norting"])
        data["Longitude"], data["Latitude"] = utmproj(
            points[0],
            points[1],
            inverse=True,
        )
        data["SurveyId"] = grp["SurveyId"].min()
        return data

    if len(drone) == 0:
        return pd.DataFrame(
            columns=["Easting", "Norting", "Longitude", "Latitude", "SurveyId"]
        )

    filtered = drone[drone.label.isin(["turtle_surface", "turtle_jbs"])]
    if len(filtered) == 0:
        return pd.DataFrame(
            columns=["Easting", "Norting", "Longitude", "Latitude", "SurveyId"]
        )
    return filtered.groupby("groups").apply(
        process_survey_group,
        include_groups=False,
    )


def merge_csv_files(file_paths: list[str] | list[Path]) -> pd.DataFrame:
    """Merge multiple CSV files into a single dataframe."""
    return pd.concat([pd.read_csv(file) for file in file_paths])


def build_turtles_report(
    images: pd.DataFrame,
    turtle_totals: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-survey turtle density report dataframe."""
    turtle_totals = turtle_totals.copy()
    turtle_totals["TotalTurtles"] = 0
    turtle = turtle_totals.groupby("SurveyId").count().reset_index(-1)
    counts = turtle[["SurveyId", "TotalTurtles"]].set_index("SurveyId")
    output = images.join(counts)
    output["TurtlesPerHec"] = output["TotalTurtles"] / output["Area"]
    output["TurtlesKmSq"] = output["TurtlesPerHec"].round(3) * 100
    output["TurtlesKmSq"] = output["TurtlesKmSq"].round(1)
    return output
