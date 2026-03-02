"""doit tasks for survey processing and area calculation."""

import glob
import os
import shutil
from pathlib import Path

import doit
import numpy as np
import pandas as pd
from doit import create_after
from doit.task import clean_targets

import turtledrone.config as config
from turtledrone.labelme.survey_utils import (
    build_survey_names,
    calculate_survey_area,
    file_destination_from_name,
    prepare_image_destinations,
    process_survey_data,
)


def task_make_surveys():
    """
    Break out the surveys into separate csv files.

    Reads survey data from process directory and creates separate CSV file
    for each survey with standardized naming.
    """

    def process_surveys(dependencies, targets):
        """Process survey file and split by survey ID."""
        drone = pd.read_csv(
            dependencies[0],
            index_col="TimeStamp",
            parse_dates=["TimeStamp"],
        )
        cfg = config.init()
        for name, data in drone.groupby("Survey"):
            data = data.loc[data.index.dropna()]
            data = build_survey_names(
                data,
                cfg.get("drone_type"),
                cfg.get("camera_type"),
                cfg.get("country"),
            )
            filename = (
                cfg.get_url("process")
                / f"{data['SurveyId'].min()}_survey.csv"
            )
            data.to_csv(filename, index=True)

    def clean():
        """Remove generated survey files."""
        cfg = config.init()
        for file in glob.glob(
            os.path.join(cfg.get_url("process"), "*_survey.csv")
        ):
            os.remove(file)

    cfg = config.init()
    file_dep = cfg.get_url("process") / "surveyswitharea.csv"
    if os.path.exists(file_dep):
        surveys = (
            pd.read_csv(file_dep, index_col="TimeStamp", parse_dates=["TimeStamp"])
            .groupby("Survey")
        )
        targets = [
            os.path.join(
                cfg.get_url("process"),
                f'{cfg.get("country")}_{data.id.max()}_'
                f'{data.index.min().strftime("%Y%m%dT%H%M")}_survey.csv',
            )
            for name, data in surveys
        ]
        return {
            "actions": [(process_surveys, [])],
            "file_dep": [file_dep],
            "targets": targets,
            "clean": [clean_targets, clean],
        } 

@create_after(executed="make_surveys")
def task_calculate_survey_areas():
    """Calculate survey area using convex hull of image positions."""

    def calculate_area(dependencies, targets):
        """Calculate area and save to target."""
        output = process_survey_data(dependencies[0])
        output.to_csv(targets[0], index=True)

    def clean():
        """Remove calculated area files."""
        cfg = config.init()
        for file in glob.glob(
            os.path.join(cfg.get_url("process"), "*_survey_area.csv")
        ):
            os.remove(file)

    cfg = config.init()
    file_dep = list(cfg.get_url("process").glob("*_survey.csv"))
    for file in file_dep:
        target = file.parent / file.name.replace("_survey", "_survey_area")
        yield {
            "name": file,
            "actions": [calculate_area],
            "file_dep": [file],
            "targets": [target],
            "uptodate": [True],
            "clean": [clean_targets, clean],
        }


@create_after(executed="calculate_survey_areas")
def task_images_dest():
    """Create new names and destination paths for images."""

    def process_images(dependencies, targets, destination):
        """Add destination paths to survey data."""
        survey = pd.read_csv(dependencies[0])
        output = prepare_image_destinations(survey, destination)
        output.to_csv(targets[0], index=False)

    def clean():
        """Remove generated files."""
        cfg = config.init()
        for pattern in ["*_survey_area_data.csv", "*_survey_area_data_summary.csv"]:
            for file in glob.glob(
                os.path.join(cfg.get_url("process"), pattern)
            ):
                os.remove(file)
        survey_dir = cfg.get_url("output") / cfg.get("country")
        if survey_dir.exists():
            shutil.rmtree(survey_dir)
        report_dir = cfg.get_url("reports")
        if report_dir.exists():
            shutil.rmtree(report_dir)

    cfg = config.init()
    file_dep = list(cfg.get_url("process").glob("*_survey_area.csv"))
    for file in file_dep:
        target = file.parent / file.name.replace("_survey_area", "_survey_area_data")
        yield {
            "name": file,
            "actions": [
                (
                    process_images,
                    [],
                    {"destination": file_destination_from_name(file.name)},
                )
            ],
            "file_dep": [file],
            "targets": [target],
            "uptodate": [True],
            "clean": [clean_targets, clean],
        }


@create_after(executed="images_dest")
def task_file_images():
    """Move images to their final destination directories."""

    def process_images(dependencies, targets):
        """Copy or link images to destination."""
        destination = os.path.dirname(targets[0])
        os.makedirs(destination, exist_ok=True)
        survey = pd.read_csv(dependencies[0])
        cfg = config.init()

        for _, row in survey.iterrows():
            if not os.path.exists(row.FileDest):
                source = Path(row.SourceFile)
                if not os.path.exists(source):
                    root = cfg.get_url("imagehead").name
                    if root in source.parts:
                        index = source.parts.index(root)
                        source = cfg.get_url("imagehead") / Path(
                            *source.parts[index + 1 :]
                        )
                
                if cfg.get("outputhardlink"):
                    os.link(os.path.abspath(source), row.FileDest)
                else:
                    shutil.copyfile(row.SourceFile, row.FileDest)

        shutil.copyfile(dependencies[0], targets[0])

    cfg = config.init()
    file_dep = list(cfg.get_url("process").glob("*_survey_area_data.csv"))
    for file in file_dep:
        target = file_destination_from_name(file.name) / file.name
        yield {
            "name": file,
            "actions": [process_images],
            "file_dep": [file],
            "targets": [target],
            "uptodate": [True],
            "clean": True,
        }


def task_move_summary():
    """Move file summary to output directory."""

    def move_summary(dependencies, targets):
        """Copy summary file to destination."""
        shutil.copyfile(dependencies[0], targets[0])

    cfg = config.init()
    file_dep = glob.glob(
        os.path.join(cfg.get_url("process"), "*_survey_area_data_summary.csv")
    )
    for file in file_dep:
        target = file_destination_from_name(
            os.path.basename(file)
        ) / os.path.basename(file)
        yield {
            "name": file,
            "actions": [move_summary],
            "file_dep": [file],
            "targets": [target],
            "uptodate": [True],
            "clean": True,
        }


if __name__ == "__main__":
    doit.run(globals())   