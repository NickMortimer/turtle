"""Doit task definitions for turtle analysis pipeline."""

import glob
import json
import os
from pathlib import Path

import pandas as pd
from doit import create_after

import turtledrone.config as config


# doit configuration dictionary (will be populated when run as __main__)
DOIT_CONFIG = {}


def _pipeline():
    """Lazy-load turtle pipeline functions to keep CLI import lightweight."""
    from turtledrone.labelme import turtle_pipeline

    return turtle_pipeline


def task_process_labelme():
    """Extract LabelMe shapes from JSON files into per-survey CSV files."""

    def loadshapes(file_path):
        print(file_path)
        try:
            with open(file_path, "r") as read_file:
                data = json.load(read_file)
            output = pd.DataFrame(data["shapes"])
            output["FilePath"] = file_path
            return output
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {file_path}: {e}")
            raise
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise

    def process_labelme(dependencies, targets):
        jsonfiles = glob.glob(os.path.join(os.path.dirname(targets[0]), "*.json"))
        if jsonfiles:
            data = pd.concat([loadshapes(file_path) for file_path in jsonfiles])
        else:
            data = pd.DataFrame()
        data.to_csv(targets[0], index=False)

    # Get flights path - will fail if config not loaded, which is OK for error messaging
    try:
        flights_path = config.geturl("flights")
    except (RuntimeError, KeyError) as e:
        print(f"Error: Could not get flights path from config: {e}")
        raise
    
    for item in glob.glob(
        os.path.join(flights_path, "**/"),
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
        pipeline = _pipeline()
        points = pipeline.load_and_prepare_points(location_file, json_file)
        points = pipeline.calculate_positions(points)
        points.to_csv(targets[0], index=False)

    file_dep = (config.geturl("flights")).rglob("**/*_json.csv")
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
        pipeline = _pipeline()
        output = pipeline.process_turtle_clusters(drone, plotpath)
        output.to_csv(targets[0], index=True)

    file_dep = (config.geturl("flights")).rglob("**/*json_points.csv")
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
        pipeline = _pipeline()
        drone = pd.read_csv(
            dependencies[0],
            parse_dates=["TimeStamp"],
            converters={"turtle_count_y": pipeline.from_np_array},
        )
        turtles = pipeline.calculate_turtle_totals(drone)
        turtles.to_csv(targets[0], index=False)

    file_dep = (config.geturl("flights")).rglob("**/*turtleMeanSift.csv")
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
        pipeline = _pipeline()
        totals = pipeline.merge_csv_files(dependencies)
        totals.to_csv(targets[0], index=False)

    file_dep = glob.glob(
        os.path.join(
            (config.geturl("flights")),
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


def task_check_survey():
    """Reuse reports task_check_survey in turtle pipeline task set."""
    from turtledrone.reports import task_check_survey as reports_task

    return reports_task()


def task_turtles_report():
    """Build final per-survey turtle density report.
    
    Requires image_coverage.csv from reports pipeline (task_concat_check_survey).
    Run reports.py tasks first to generate survey coverage data.
    """

    def process_survey(dependencies, targets):
        pipeline = _pipeline()
        
        # Load required files - will fail if either is missing
        image_coverage_file = list(filter(lambda x: "image_coverage" in str(x), dependencies))[0]
        turtles_file = list(filter(lambda x: "turtles_totals" in str(x), dependencies))[0]
        
        images = pd.read_csv(image_coverage_file, index_col="SurveyId")
        turtle_file = pd.read_csv(turtles_file)
        output = pipeline.build_turtles_report(images, turtle_file)
        output.to_csv(targets[0], index=True)

    file_dep = [
        config.geturl("reports") / "turtles_totals.csv",
        config.geturl("reports")/ "image_coverage.csv",
    ]
    os.makedirs(config.geturl("reports"), exist_ok=True)
    targets = os.path.join(config.geturl("reports"), "turtles_per_survey.csv")
    return {
        "actions": [process_survey],
        "file_dep": file_dep,
        "targets": [targets],
        "clean": True,
    }

if __name__ == '__main__':
    import sys
    import doit
    
    # Parse config file from command line arguments
    config_file = None
    for arg in sys.argv[1:]:
        if arg.startswith('config='):
            config_file = arg.split('=', 1)[1]
        elif arg.startswith('--config='):
            config_file = arg.split('=', 1)[1]
        elif arg == '--config' or arg == '-c':
            # Next argument should be the config file
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                config_file = sys.argv[idx + 1]
    
    if not config_file:
        print("Error: Config file required. Use: config=/path/to/config.yaml or --config /path/to/config.yaml")
        sys.exit(1)
    
    # Load the config before running tasks
    print(f"Loading configuration from: {config_file}")
    config.read_config(Path(config_file), prompt_if_none=False)
    print(f"Configuration loaded successfully")
    
    # Change working directory to config file's parent directory
    config_dir = Path(config_file).parent.resolve()
    os.chdir(config_dir)
    print(f"Working directory: {config_dir}")
    
    # Set up per-config doit database file in the config directory
    db_file = config_dir / ".doit.db"
    print(f"Using task database: {db_file}")
    
    # Configure doit
    DOIT_CONFIG.update({
        "num_processes": 10,
        "verbosity": 2,
        "db_file": str(db_file),
    })
    
    # Clear command line args so doit runs all tasks
    sys.argv = [sys.argv[0]]
    
    # Run doit with the task definitions (will run all tasks when no args provided)
    doit.run(globals())