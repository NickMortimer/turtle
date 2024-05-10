
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from datetime import datetime,  timedelta
import uuid
import psutil
from math import  ceil
import typer
import yaml
import glob
import pandas as pd
import subprocess
import shlex
import logging
import platform
import os
from io import StringIO

def list_sdcards(format_type,maxcardsize=512):
    """
    Scan for SD cards.

    Args:
        format_type : type of format on the sdcard (exfat preffered)
        maxcardsize : select drives with less than the max in Gb
    """
    result =[]
    for i in psutil.disk_partitions():
        p =psutil.disk_usage(i.mountpoint)
        if ceil(p.total/1000000000)<=maxcardsize:
            if i.fstype.lower()==format_type:
                if platform.system() == "Linux":
                    if 'media' in i.mountpoint:
                        result.append(i.mountpoint)
                else:    
                    result.append(i.mountpoint)
    return result

def initialise(config_path,card_path,days,overwrite,dry_run: bool):
    """
    Implementation of the MarImBA initialise command for the BRUVS
    """

    def make_yml(file_path):
        if file_path.exists() and (not overwrite):
            raise typer.Abort(f"Error SDCard already initialise {file_path}")
        else:
            env = Environment(loader = FileSystemLoader(str(config_path.parent)),   trim_blocks=True, lstrip_blocks=True)
            template = env.get_template('import.yml')
            fill = {"instrumentPath" : Path.cwd(), "instrument" : 'gopro_bruv',
                    "importdate" : f"{datetime.now()+timedelta(days=days):%Y-%m-%d}",
                    "importtoken" : str(uuid.uuid4())[0:8]}
            #self.logger.info(f'{dry_run_log_string}Making import file "{file_path}"')
            if not dry_run:
                with open(file_path, "w") as file:
                    file.write(template.render(fill))
    # Set dry run log string to prepend to logging
    dry_run_log_string = "DRY_RUN - " if dry_run else ""    
    if isinstance(card_path,list):
        [make_yml(Path(path)/ "import.yml") for path in card_path]
    else:
        make_yml(Path(card_path) /"import.yml")

def import_command(instrument_path,card_path,copy,move,find,file_extension,dry_run: bool):
    """
    Implementation of the MarImBA initalise command for the BRUVS
    """
    for card in card_path:
        dry_run_log_string = "DRY_RUN - " if dry_run else ""
        importyml =f"{card}/import.yml"
        if os.path.exists(importyml):
            with open(importyml, 'r') as stream:
                try:
                    importdetails=yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    raise typer.Abort(f"Error possible corrupt yaml {importyml}")
        else:
            typer.echo(f"Error {importyml} not found")
        importdetails['instrument_path'] = instrument_path
        destination =importdetails["import_template"].format(**importdetails)
        if destination is None:
            logging.warning(f"Warning no path found")
        else:
            logging.info(f'{dry_run_log_string}  Copy  {card} --> {destination}')
            command =f"rclone copy {os.path.abspath(card)} {os.path.abspath(destination)} --progress --low-level-retries 1 "
            command = command.replace('\\','/')
            if copy:
                logging.info(f'{dry_run_log_string}  {command}')
                if not dry_run:
                    os.makedirs(destination,exist_ok=True)
                    process = subprocess.Popen(shlex.split(command))
                    process.wait()
            if move:
                command =f"rclone move {card} {destination} --progress --delete-empty-src-dirs"
                command = command.replace('\\','/')
                logging.info(f'{dry_run_log_string}  {command}')
                if not dry_run:
                    os.makedirs(destination,exist_ok=True)
                    process = subprocess.Popen(shlex.split(command))
                    process.wait()
