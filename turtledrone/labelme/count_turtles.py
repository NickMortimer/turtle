"""doit tasks for LabelMe turtle processing and reporting."""

import glob
import json
import os

import doit
import pandas as pd
from doit import create_after

import turtledrone.config as config
from turtledrone.labelme.turtle_pipeline import (
    build_turtles_report,
    calculate_positions,
    calculate_turtle_totals,
    from_np_array,
    load_and_prepare_points,
    merge_csv_files,
    process_turtle_clusters,
)


def task_set_up():
    """Load repository configuration for subsequent tasks."""
    config.read_config()


def task_process_labelme():
    """Extract LabelMe shapes from JSON files into per-survey CSV files."""

    def loadshapes(file_path):
        print(file_path)
        with open(file_path, "r") as read_file:
            data = json.load(read_file)
        output = pd.DataFrame(data["shapes"])
        output["FilePath"] = file_path
        return output

    def process_labelme(_, targets):
        jsonfiles = glob.glob(os.path.join(os.path.dirname(targets[0]), "*.json"))
        if jsonfiles:
            data = pd.concat([loadshapes(file_path) for file_path in jsonfiles])
        else:
            data = pd.DataFrame()
        data.to_csv(targets[0], index=False)

    for item in glob.glob(
        os.path.join(config.geturl("output"), "AU/**/"),
        recursive=True,
    ):
        file_dep = glob.glob(os.path.join(os.path.dirname(item), "*.json"))
        if file_dep:
            target = os.path.join(
                os.path.dirname(item),
                f"{os.path.basename(os.path.dirname(item))}_json.csv",
            )
            yield {
                "name": item,
                "file_dep": file_dep,
                "actions": [process_labelme],
                "targets": [target],
                "clean": True,
                "uptodate": [True],
            }


def task_calculate_positions():
    """Calculate world coordinates for labeled turtle points."""

    def process_positions(dependencies, targets):
        location_file = list(filter(lambda x: "location" in x, dependencies))[0]
        json_file = list(filter(lambda x: "_json" in x, dependencies))[0]
        points = load_and_prepare_points(location_file, json_file)
        points = calculate_positions(points)
        points.to_csv(targets[0], index=False)

    file_dep = (config.geturl("output") / "AU").rglob("**/*_json.csv")
    for item in file_dep:
        locations = item.parent / "location.csv"
        if locations.exists():
            target = item.parent / item.name.replace("_json", "json_points")
            yield {
                "name": target,
                "actions": [process_positions],
                "file_dep": [item, locations],
                "targets": [target],
                "clean": True,
            }


@create_after(executed="calculate_positions")
def task_process_turtles():
    """Group and cluster turtle detections for each survey output file."""

    def process_turtles(dependencies, targets):
        plotpath = os.path.dirname(targets[0])
        drone = pd.read_csv(dependencies[0], parse_dates=["TimeStamp"])
        output = process_turtle_clusters(drone, plotpath)
        output.to_csv(targets[0], index=True)

    file_dep = (config.geturl("output") / "AU").rglob("**/*json_points.csv")
    for item in file_dep:
        if os.stat(item).st_size > 100:
            target = item.parent / item.name.replace(
                "points.csv",
                "points_turtleMeanSift.csv",
            )
            yield {
                "name": target,
                "actions": [process_turtles],
                "file_dep": [item],
                "targets": [target],
                "clean": True,
            }


@create_after(executed="process_turtles")
def task_process_turtles_totals():
    """Reduce clustered detections to per-group turtle total coordinates."""

    def process_turtles_totals(dependencies, targets):
        drone = pd.read_csv(
            dependencies[0],
            parse_dates=["TimeStamp"],
            converters={"turtle_count_y": from_np_array},
        )
        turtles = calculate_turtle_totals(drone)
        turtles.to_csv(targets[0], index=False)

    file_dep = (config.geturl("output") / "AU").rglob("**/*turtleMeanSift.csv")
    for item in file_dep:
        target = item.parent / item.name.replace(
            "turtleMeanSift.csv",
            "turtleMeanSift_grouped.csv",
        )
        yield {
            "name": target,
            "actions": [process_turtles_totals],
            "file_dep": [item],
            "targets": [target],
            "clean": True,
        }


@create_after(executed="process_turtles_totals")
def task_merge_turtle_totals():
    """Merge all grouped turtle total files into one reports CSV."""

    def process_merge(dependencies, targets):
        totals = merge_csv_files(dependencies)
        totals.to_csv(targets[0], index=False)

    file_dep = glob.glob(
        os.path.join(
            (config.geturl("output") / "AU"),
            "**/*turtleMeanSift_grouped.csv",
        ),
        recursive=True,
    )
    if file_dep:
        target = config.geturl("reports") / "turtles_totals.csv"
        return {
            "actions": [process_merge],
            "file_dep": file_dep,
            "targets": [target],
            "clean": True,
        }


def task_turtles_report():
    """Build final per-survey turtle density report."""

    def process_survey(_, targets):
        images = pd.read_csv(
            config.geturl("reports") / "image_coverage.csv",
            index_col="SurveyId",
        )
        turtle_file = pd.read_csv(config.geturl("reports") / "turtles_totals.csv")
        output = build_turtles_report(images, turtle_file)
        output.to_csv(targets[0], index=True)

    file_dep = [
        config.geturl("reports") / "image_coverage.csv",
        config.geturl("reports") / "turtles_totals.csv",
    ]
    os.makedirs(config.geturl("reports"), exist_ok=True)
    targets = os.path.join(config.geturl("reports"), "turtles_per_survey.csv")
    return {
        "actions": [process_survey],
        "file_dep": file_dep,
        "targets": [targets],
        "clean": True,
    }


if __name__ == "__main__":
    doit.run(globals())
