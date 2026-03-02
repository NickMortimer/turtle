"""doit tasks for GNSS base station file downloading."""

import os
import sys
from pathlib import Path

import doit
import pandas as pd
import pysftp
from doit import create_after
from doit.tools import run_once

import turtledrone.config as config
from turtledrone.read_rtk import read_mrk_gpst
from turtledrone.utils.gnss_utils import get_base_file_names


def download_base_files(
    files: list[dict],
    username: str = "anonymous",
    password: str = "",
) -> None:
    """
    Download GNSS base station files from Geoscience Australia FTP.

    Parameters
    ----------
    files : list[dict]
        List of file dictionaries with 'path', 'file', and 'destination'.
    username : str
        FTP username.
    password : str
        FTP password.
    """
    with pysftp.Connection(
        "sftp.data.gnss.ga.gov.au",
        username=username,
        password=password,
    ) as sftp:
        for item in files:
            dest = os.path.join(item["destination"], item["file"])
            if not os.path.exists(dest):
                try:
                    sftp.cwd(item["path"])
                    sftp.get(item["file"], dest)
                except FileNotFoundError:
                    print(f"File not found on server: {item['file']}")
                except Exception as e:
                    print(f"Error downloading {item['file']}: {e}")


def task_calc_basefiles():
    """Calculate base station files needed for processing."""

    def calc_basefiles(dependencies, targets):
        """Read mark files and generate base file list."""
        marks = [read_mrk_gpst(mark) for mark in dependencies]
        cfg = config.init()
        files = pd.concat(
            [
                get_base_file_names(
                    df.index,
                    cfg.get("basestation"),
                    cfg.get_url("gnssceche"),
                )
                for df in marks
            ]
        )
        files.drop_duplicates(inplace=True)
        files.to_csv(targets[0], index=False)

    cfg = config.init()
    os.makedirs(cfg.get_url("process"), exist_ok=True)
    os.makedirs(cfg.get_url("gnssceche"), exist_ok=True)

    file_dep = []
    for image_dirs in config.init().get("imagesource_dirs", []):
        file_dep += list(Path(image_dirs).glob("*Timestamp.MRK"))

    file_dep = list(
        filter(lambda x: os.stat(x).st_size > 0, file_dep)
    )
    target = cfg.get_url("process") / "basefiles.csv"

    return {
        "actions": [(calc_basefiles, [])],
        "file_dep": file_dep,
        "targets": [target],
        "uptodate": [True],
        "clean": True,
    }


@create_after(executed="calc_basefiles")
def task_get_basefiles():
    """Download base station files from remote FTP server."""

    def calc_basefiles(dependencies, targets):
        """Download files from base file list."""
        basefiles = pd.read_csv(dependencies[0])
        cfg = config.init()
        download_base_files(
            basefiles.to_dict("records"),
            password=cfg.get("email", ""),
        )

    cfg = config.init()
    file_dep = cfg.get_url("process") / "basefiles.csv"
    bases = pd.read_csv(file_dep)
    gnss = cfg.get_url("gnssceche")
    targets = bases.file.apply(lambda x: gnss / x).to_list()

    return {
        "actions": [calc_basefiles],
        "file_dep": [file_dep],
        "targets": targets,
        "uptodate": [True],
        "clean": True,
    }


@create_after(executed="get_basefiles")
def task_unzip_base():
    """Decompress downloaded gzip base station files."""
    cfg = config.init()
    file_dep = cfg.get_url("gnssceche").glob("*.gz")
    file_dep = list(filter(lambda x: os.stat(x).st_size > 0, file_dep))

    if sys.platform.startswith("linux"):
        for file in file_dep:
            yield {
                "name": file,
                "actions": [f"gunzip -k -d {file}"],
                "file_dep": [file],
                "targets": [file.stem],
                "uptodate": [run_once],
            }


@create_after(executed="unzip_base")
def task_crx2rinx_base():
    """Convert Compact RINEX to standard RINEX format."""
    cfg = config.init()
    file_dep = cfg.get_url("gnssceche").glob("*.crx")

    if sys.platform.startswith("linux"):
        for file in file_dep:
            target = file.with_suffix(".rnx")
            yield {
                "name": file,
                "actions": [f'crx2rnx -f "{file}"'],
                "targets": [target],
                "uptodate": [run_once],
                "clean": True,
            }


if __name__ == "__main__":
    doit.run(globals())




# filename = r"T:/drone/raw/card0/SURVEY/100_0001/100_0001_Timestamp.MRK"
# inputpath =os.path.split(filename)[0]
# jsonfile = inputpath+'/exif.json'
# merge = inputpath+'/merge.csv'

# station = 'EXMT00AUS'
# data = pd.read_csv(merge,parse_dates=['UTCtime'])
# getbase(data.UTCtime,'T:/drone/raw/gnss')
