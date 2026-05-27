"""Doit task definitions for turtle analysis pipeline."""

import glob
import json
import os
import math
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

@create_after(executed="calculate_gsd")
def task_measure_turtles():
    """Calculate turtle length/width from LabelMe line annotations."""
    def _line_length(points):
        if len(points) != 2:
            return None
        (x1, y1), (x2, y2) = points
        return float(math.hypot(x2 - x1, y2 - y1))

    def process_measurements(dependencies, targets):
        rows = []
        for json_file in dependencies:
            with open(json_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)

            length_vals = []
            width_vals = []
            for shape in data.get("shapes", []):
                if shape.get("shape_type") != "line":
                    continue
                value = _line_length(shape.get("points", []))
                if value is None:
                    continue
                if shape.get("label") == "turtle_length":
                    length_vals.append(value)
                elif shape.get("label") == "turtle_width":
                    width_vals.append(value)

            turtle_length_px = max(length_vals) if length_vals else None
            turtle_width_px = max(width_vals) if width_vals else None
            ratio = (
                turtle_length_px / turtle_width_px
                if turtle_length_px is not None and turtle_width_px not in (None, 0)
                else None
            )

            rows.append(
                {
                    "json_file": str(json_file),
                    "image_path": data.get("imagePath"),
                    "image_width": data.get("imageWidth"),
                    "image_height": data.get("imageHeight"),
                    "turtle_length_px": turtle_length_px,
                    "turtle_width_px": turtle_width_px,
                    "length_to_width_ratio": ratio,
                }
            )
        pattern = (
            r"^(?:\d+_)?"
            r"(?P<original_image_stem>[A-Z0-9]+_[A-Z]+_[A-Z]{2}_[A-Z0-9]+_\d{8}T\d{6}_\d{4})"
            r"(?:_det_\d+_.+)?$"
        )

        df = pd.DataFrame(rows)
        df["stem"] = pd.Series(rows).apply(lambda row: Path(row["json_file"]).stem)
        df = pd.concat([df, df["stem"].str.extract(pattern)], axis=1)
        df["original_image_name_jpg"] = df["original_image_stem"]
        df["key"] = (
            df["original_image_stem"]
            .astype(str)
            .str.replace(r"(?i)\.jpe?g$", "", regex=True)
        )
        df.to_csv(targets[0], index=False)

    measurements_dir = config.geturl("measurements")
    recursive = bool(config.get("measurements_recursive", True))

    if recursive:
        json_deps = sorted(measurements_dir.rglob("*.json"))
    else:
        json_deps = sorted(measurements_dir.glob("*.json"))

    if not json_deps:
        return

    target = config.geturl("reports") / "turtle_measurements.csv"
    return {
        "actions": [process_measurements],
        "file_dep": json_deps,
        "task_dep": ["calculate_gsd"],
        "targets": [target],
        "clean": True,
    }

def task_calculate_gsd():
    """Calculate per-image GSD assuming flat ground and nadir camera."""

    def process_gsd(dependencies, targets):
        source = dependencies[0]
        df = pd.read_csv(source, parse_dates=["TimeStamp"])

        nadir_tol_deg = float(config.get("gsd_nadir_tolerance_deg", 6.0))
        strict_nadir = bool(config.get("gsd_strict_nadir", True))

        if "GimbalPitchDegree" in df.columns:
            pitch = pd.to_numeric(df["GimbalPitchDegree"], errors="coerce")
            pitch_source = "GimbalPitchDegree"
        elif "CameraPitch" in df.columns:
            pitch = pd.to_numeric(df["CameraPitch"], errors="coerce")
            pitch_source = "CameraPitch"
        else:
            raise ValueError(
                f"{source} is missing GimbalPitchDegree/CameraPitch for nadir check"
            )

        nadir_error_deg = (pitch + 90.0).abs()
        is_nadir = nadir_error_deg <= nadir_tol_deg

        if "RelativeAltitude" not in df.columns:
            raise ValueError(f"{source} missing RelativeAltitude")
        alt_m = pd.to_numeric(df["RelativeAltitude"], errors="coerce")

        if (
            "CalibratedFocalLengthX" in df.columns
            and "CalibratedFocalLengthY" in df.columns
        ):
            fx = pd.to_numeric(df["CalibratedFocalLengthX"], errors="coerce")
            fy = pd.to_numeric(df["CalibratedFocalLengthY"], errors="coerce")
        elif "CalibratedFocalLength" in df.columns:
            f = pd.to_numeric(df["CalibratedFocalLength"], errors="coerce")
            fx = f
            fy = f
        else:
            raise ValueError(
                f"{source} missing CalibratedFocalLength(X/Y) for GSD calculation"
            )

        gsd_x_m_per_px = alt_m / fx
        gsd_y_m_per_px = alt_m / fy
        gsd_xy_mean_m_per_px = (gsd_x_m_per_px + gsd_y_m_per_px) / 2.0

        if strict_nadir:
            gsd_x_m_per_px = gsd_x_m_per_px.where(is_nadir)
            gsd_y_m_per_px = gsd_y_m_per_px.where(is_nadir)
            gsd_xy_mean_m_per_px = gsd_xy_mean_m_per_px.where(is_nadir)

        out = df.copy()
        out["gsd_pitch_source"] = pitch_source
        out["nadir_error_deg"] = nadir_error_deg
        out["is_nadir"] = is_nadir
        out["gsd_x_m_per_px"] = gsd_x_m_per_px
        out["gsd_y_m_per_px"] = gsd_y_m_per_px
        out["gsd_xy_mean_m_per_px"] = gsd_xy_mean_m_per_px
        out["gsd_xy_mean_cm_per_px"] = gsd_xy_mean_m_per_px * 100.0

        if "NewName" in out.columns:
            out["gsd_image_stem"] = out["NewName"].astype(str).apply(
                lambda value: Path(value).stem
            )
        elif "FileName" in out.columns:
            out["gsd_image_stem"] = out["FileName"].astype(str).apply(
                lambda value: Path(value).stem
            )
        elif "SourceFile" in out.columns:
            out["gsd_image_stem"] = out["SourceFile"].astype(str).apply(
                lambda value: Path(value).stem
            )
        else:
            out["gsd_image_stem"] = pd.NA

        out.to_csv(targets[0], index=False)

    file_dep = list(config.geturl("flights").rglob("*_survey_area_data.csv"))
    for item in file_dep:
        target = item.parent / item.name.replace(
            "_survey_area_data.csv", "_survey_area_gsd.csv"
        )
        yield {
            "name": str(item),
            "actions": [process_gsd],
            "file_dep": [item],
            "targets": [target],
            "clean": True,
        }

@create_after(executed="concat_gsd")
def task_calculate_true_sizes():
    """Convert pixel turtle measurements to true size using per-image GSD."""

    def process_true_sizes(dependencies, targets):
        measurements_file = list(
            filter(lambda value: "turtle_measurements.csv" in str(value), dependencies)
        )[0]
        gsd_file = list(
            filter(lambda value: "gsd_all_images.csv" in str(value), dependencies)
        )[0]

        measurements = pd.read_csv(measurements_file)
        gsd_all = pd.read_csv(gsd_file)

        if "key" not in gsd_all.columns:
            if "NewName" in gsd_all.columns:
                gsd_all["key"] = (
                    gsd_all["NewName"]
                    .astype(str)
                    .str.replace(r"(?i)\.jpe?g$", "", regex=True)
                )
            elif "gsd_image_stem" in gsd_all.columns:
                gsd_all["key"] = gsd_all["gsd_image_stem"].astype(str)
            else:
                raise ValueError(
                    "gsd_all_images.csv is missing key/NewName/gsd_image_stem"
                )

        if "key" not in measurements.columns:
            if "original_image_stem" in measurements.columns:
                measurements["key"] = (
                    measurements["original_image_stem"]
                    .astype(str)
                    .str.replace(r"(?i)\.jpe?g$", "", regex=True)
                )
            else:
                raise ValueError("turtle_measurements.csv is missing key")

        gsd_all["key"] = gsd_all["key"].astype(str)
        measurements["key"] = measurements["key"].astype(str)

        if bool(config.get("true_size_nadir_only", True)) and "is_nadir" in gsd_all.columns:
            gsd_all = gsd_all[gsd_all["is_nadir"] == True]  # noqa: E712

        gsd_column = (
            "gsd_xy_mean_m_per_px"
            if "gsd_xy_mean_m_per_px" in gsd_all.columns
            else "gsd_mean_m_per_px"
        )
        if gsd_column not in gsd_all.columns:
            raise ValueError("No mean GSD column found in gsd_all_images.csv")

        gimbal_columns = [
            column
            for column in [
                "GimbalPitchDegree",
                "CameraPitch"
            ]
            if column in gsd_all.columns
        ]

        gsd_lookup = (
            gsd_all.sort_values("nadir_error_deg")
            .drop_duplicates(subset=["key"], keep="first")[
                ["key", gsd_column, *gimbal_columns]
            ]
            .rename(columns={gsd_column: "gsd_m_per_px"})
        )

        merged = measurements.merge(gsd_lookup, on="key", how="left")

        merged["turtle_length_m"] = merged["turtle_length_px"] * merged["gsd_m_per_px"]
        merged["turtle_width_m"] = merged["turtle_width_px"] * merged["gsd_m_per_px"]
        merged["turtle_length_cm"] = merged["turtle_length_m"] * 100.0
        merged["turtle_width_cm"] = merged["turtle_width_m"] * 100.0

        total_rows = int(len(merged))
        matched_rows = int(merged["gsd_m_per_px"].notna().sum())
        missing_rows = total_rows - matched_rows
        matched_percent = (
            (matched_rows / total_rows) * 100.0 if total_rows > 0 else 0.0
        )
        missing_percent = (
            (missing_rows / total_rows) * 100.0 if total_rows > 0 else 0.0
        )

        summary = pd.DataFrame(
            [
                {
                    "total_measurement_rows": total_rows,
                    "matched_gsd_rows": matched_rows,
                    "missing_gsd_rows": missing_rows,
                    "matched_percent": matched_percent,
                    "missing_percent": missing_percent,
                }
            ]
        )

        merged.sort_values("key").to_csv(targets[0], index=False)
        summary.to_csv(targets[1], index=False)

    file_dep = [
        config.geturl("reports") / "turtle_measurements.csv",
        config.geturl("reports") / "gsd_all_images.csv",
    ]
    target = config.geturl("reports") / "turtle_measurements_true_size.csv"
    summary_target = config.geturl("reports") / "turtle_measurements_true_size_summary.csv"

    return {
        "actions": [process_true_sizes],
        "file_dep": file_dep,
        "task_dep": ["measure_turtles", "concat_gsd"],
        "targets": [target, summary_target],
        "clean": True,
    }


@create_after(executed="calculate_gsd")
def task_concat_gsd():
    """Concatenate all per-survey GSD files into a single report file."""

    def process_concat_gsd(dependencies, targets):
        reports_dir = Path(targets[0]).parent
        os.makedirs(reports_dir, exist_ok=True)

        frames = [pd.read_csv(file) for file in dependencies]
        gsd_all = pd.concat(frames, ignore_index=True)

        if "gsd_image_stem" not in gsd_all.columns:
            if "NewName" in gsd_all.columns:
                gsd_all["gsd_image_stem"] = gsd_all["NewName"].astype(str).apply(
                    lambda value: Path(value).stem
                )
            elif "FileName" in gsd_all.columns:
                gsd_all["gsd_image_stem"] = gsd_all["FileName"].astype(str).apply(
                    lambda value: Path(value).stem
                )
            else:
                gsd_all["gsd_image_stem"] = pd.NA

        if "NewName" in gsd_all.columns:
            gsd_all["key"] = (
                gsd_all["NewName"]
                .astype(str)
                .str.replace(r"(?i)\.jpe?g$", "", regex=True)
            )
        else:
            gsd_all["key"] = gsd_all["gsd_image_stem"].astype(str)

        gsd_all.to_csv(targets[0], index=False)

    file_dep = list(config.geturl("flights").rglob("*_survey_area_gsd.csv"))
    if not file_dep:
        return

    target = config.geturl("reports") / "gsd_all_images.csv"
    return {
        "actions": [process_concat_gsd],
        "file_dep": file_dep,
        "targets": [target],
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