"""doit tasks for EXIF metadata extraction and processing."""

import os
from pathlib import Path
from shutil import which

import doit
import pandas as pd
from doit import create_after

import turtledrone.config as config
from turtledrone.utils.exif_utils import process_exif_json


def task_create_json():
    """Extract EXIF metadata from images using exiftool."""

    cfg = config.init()
    exifpath = cfg.get_url("exiftool")

    for item in config.init().get("imagesource_dirs", []):
        item = Path(item)
        file_dep = list(item.glob(cfg.get("imagewild", "*.JPG").upper()))
        if len(file_dep) == 0:
            file_dep = list(item.glob(cfg.get("imagewild", "*.JPG").lower()))

        if len(file_dep) > 0:
            target = item / "exif.json"
            if which("exiftool"):
                yield {
                    "name": str(target),
                    "actions": [
                        f'exiftool -ext JPG -ext jpg -json "{item.resolve()}" > '
                        f'"{target.resolve()}"'
                    ],
                    "targets": [target],
                    "uptodate": [True],
                    "clean": True,
                }
            else:
                yield {
                    "name": str(target),
                    "actions": [
                        f'"{exifpath}" -ext JPG -ext jpg -json '
                        f'"{os.path.abspath(item)}" > "{os.path.abspath(target)}"'
                    ],
                    "targets": [target],
                    "uptodate": [True],
                    "clean": True,
                }


@create_after(executed="create_json")
def task_process_json():
    """Process raw EXIF JSON into structured CSV format."""

    def process_json(dependencies, targets):
        """Load JSON, parse GPS/calibration, and save as CSV."""
        source_file = dependencies[0]
        if os.stat(source_file).st_size > 0:
            drone = pd.read_json(source_file)
            cfg = config.init()
            catalog_dir = cfg.catalog_dir or Path(source_file).parent
            drone = process_exif_json(drone, catalog_dir)
            drone.set_index("Sequence", inplace=True)
            drone.to_csv(list(targets)[0], index=True)

    cfg = config.init()
    for item in config.init().get("imagesource_dirs", []):
        item = Path(item)
        file_dep = item / "exif.json"
        if file_dep.exists():
            target = file_dep.with_suffix(".csv")
            yield {
                "name": str(target),
                "actions": [process_json],
                "file_dep": [file_dep],
                "targets": [target],
                "clean": True,
            }


if __name__ == "__main__":
    doit.run(globals())   