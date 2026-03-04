#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import typer
from typer import Typer
import sys
from pathlib import Path
from typing import Optional
from cookiecutter.main import cookiecutter
import os
import shutil
import yaml
import turtledrone as td


__author__ = "Turtle Drone Team"
__copyright__ = "Copyright 2023, CSIRO"
__credits__ = [
    "Nick Mortimer <nick.mortimer@csiro.au>",
]
__license__ = "MIT"
__version__ = "0.2"
__maintainer__ = "Nick Mortimer"
__email__ = "nick.mortimer@csiro.au"
__status__ = "Development"

tdrone = typer.Typer(
    name="turtle drone",
    help="""" Turtle Drone
        A Python CLI tool to manage drone data""",
    short_help="turtle drone",
    no_args_is_help=True,
)


def read_config(filename):
    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
        cfg['start_date'] = cfg['start_date'].isoformat()
        cfg['end_date'] = cfg['end_date'].isoformat()
    return cfg


# @goprobruv.callback()
# def global_options(
#         level: LogLevel = typer.Option(LogLevel.INFO, help="Logging level."),
# ):
#     """
#     Global options for MarImBA CLI.
#     """
#     get_rich_handler().setLevel(logging.getLevelName(level.value))
#     logger.info(f"Initialised MarImBA CLI v{__version__}")

@tdrone.command('initialise')
def initialise(
        config_path: str = typer.Argument(None, help="Path to import.yml"),
        card_path: list[str] = typer.Argument(None, help="MarImBA instrument ID.",),
        all: bool = typer.Option(False, help="Execute the command and print logging to the terminal, but do not change any files."),
        days: int = typer.Option(0, help="Add an offset to the import date e.g. +1 = to set the date to tomorrow "),
        dry_run: bool = typer.Option(False, help="Execute the command and print logging to the terminal, but do not change any files."),
        overwrite:bool = typer.Option(False, help="Overwrite import.yaml"),
        cardsize:int = typer.Option(512, help="maximum card size"),
        format_type:str = typer.Option('exfat', help="Card format type"),

):
    """
    initialise sd cards
    """
    from turtledrone.utils.cards import list_sdcards,initialise
    if config_path is None:
        config_path = Path().cwd() / 'import.yml'
        if config_path.exists():
            if all and (not card_path ):
                card_path = list_sdcards(format_type,cardsize)
            initialise(config_path,card_path=card_path,dry_run=dry_run,days=days,overwrite=overwrite)

@tdrone.command('import')
def import_command(
        config_path: str = typer.Argument(None, help="Root path to MarImBA collection."),
        card_path: list[str] = typer.Argument(None, help="MarImBA instrument ID."),
        all: bool = typer.Option(False, help="Execute the command and print logging to the terminal, but do not change any files."),
        copy: bool = typer.Option(True, help="Clean source"),
        move: bool = typer.Option(False, help="move source"),
        find: bool = typer.Option(False, help="import to the same hash"),
        cardsize:int = typer.Option(512, help="maximum card size"),
        format_type:str = typer.Option('exfat', help="Card format type"),
        extra: list[str] = typer.Option([], help="Extra key-value pass-through arguments."),
        dry_run: bool = typer.Option(False, help="Execute the command and print logging to the terminal, but do not change any files."),
        file_extension: str = typer.Option("MP4", help="extension to catalog"),
):
    """
    Import SD cards to working directory
    """ 
    from turtledrone.utils.cards import list_sdcards
    from turtledrone.utils.cards import import_command
    if config_path is None:
        config_path = list(Path().cwd().glob('*_config.yml'))[0]
    if config_path.exists():
        cfg = read_config(config_path)
        instrument_path =Path(cfg['imagesource'].format(CATALOG_DIR=config_path.parent))  
    if all and (not card_path ):
        card_path = list_sdcards(format_type,cardsize)
    import_command(instrument_path,card_path=card_path,copy=copy,move=move,find=find,dry_run=dry_run,file_extension=file_extension)


# @tdrone.command('initailise')
# def initailise(
#         config: str = typer.Argument(..., help="path to config file")):
#     """
#     initialise directory structure
#     """
#     from turtledrone.initailise import run
#     run()

@tdrone.command('exifdata')
def exifdata(
        config: str = typer.Argument(..., help="path to config file")):
    """
    initialise directory structure
    """
    sys.argv = [sys.argv[0]]+sys.argv[2:]
    from turtledrone.exifdata import run
    run()

@tdrone.command('new')
def new(
    template_name: str = typer.Argument(..., help="select type of template (fieldtrip, drone, surveyarea)"),
    template_path: str = typer.Argument((Path(__file__).parent.parent) / 'templates', help="path to repo ")):
    """
    
    """
    template_path = Path(template_path) / template_name
    if template_path.exists():
        if template_name=='fieldtrip':
            cookiecutter(str(template_path))    
        else:
            conf = Path().cwd() / 'fieldtrip.yml'
            if conf.exists():
                cfg = read_config(conf)
                if template_name=='drone':
                    cookiecutter(str(template_path),extra_context=cfg,output_dir=(cfg['drone_path'].format(CATALOG_DIR=Path().cwd())))
                if template_name=='surveyarea':
                    cookiecutter(str(template_path),extra_context=cfg,output_dir=(cfg['survey_area_path'].format(CATALOG_DIR=Path().cwd())))

            else:
                raise typer.Abort(f'No field config file found in directory {conf} \n Please move into the field  trip directory')
    else:
        raise typer.Abort(f'Template not found')
    

@tdrone.command('process')
def process(config : str= typer.Argument(None, help="path to config file")):
    """
     process drone data if no path to config file is specified all drones processed in sequence
    """

    if config is None:
        fieldtrip = Path().cwd() / 'fieldtrip.yml'
        if fieldtrip.exists():
            cfg = read_config(fieldtrip)
            drones =Path(cfg['drone_path'].format(CATALOG_DIR=Path().cwd())).rglob('*_config.yml')
            for drone in drones:
                dronecfg = read_config(drone)
                sys.argv[1]=f'config={drone}'
                os.chdir(drone.parent)
                from turtledrone import initailise as init
                init.run()
                from turtledrone import exifdata as exif
                exif.run()
                if dronecfg['basestation']:
                    from turtledrone import getbase as gb
                    gb.run()
                
                import importlib
                try:
                    module = __import__('turtledrone', fromlist=[dronecfg['drone_type']])

                    # Get the method from the module
                    getattr(module,dronecfg['drone_type'] ).run()
                except:
                    raise typer.Abort(f"Processor {dronecfg['processor']} not found")
                from turtledrone import setsurveyarea
                setsurveyarea.run()
                from turtledrone import surveys
                surveys.run()
                from turtledrone import reports
                reports.run()


@tdrone.command('reports')
def reports_command(
    config: Optional[str] = typer.Argument(
        None,
        help="Path to config file (or config=/path/to/config.yaml)",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
):
    """
    Generate survey reports (image coverage, plots, geopackages).
    
    This command runs the reports pipeline tasks including:
    - Survey coverage summaries
    - Image coverage statistics
    - Survey plots and maps
    - Geopackage exports for QGIS
    """
    import doit
    import turtledrone.config as config_module
    
    resolved_config: Optional[Path] = config_path
    if resolved_config is None and config is not None:
        if config.startswith("config="):
            resolved_config = Path(config.split("=", 1)[1])
        else:
            resolved_config = Path(config)

    if resolved_config is None:
        typer.echo("Error: --config option is required", err=True)
        raise typer.Exit(code=1)

    if not resolved_config.exists():
        typer.echo(f"Error: Config file not found: {resolved_config}", err=True)
        raise typer.Exit(code=1)
    
    typer.echo(f"Loading configuration from: {resolved_config}")
    config_module.read_config(resolved_config, prompt_if_none=False)
    typer.echo("Configuration loaded successfully")
    
    # Change working directory to config file's parent directory
    config_dir = resolved_config.parent.resolve()
    os.chdir(config_dir)
    typer.echo(f"Working directory: {config_dir}")
    
    # Set up per-config doit database file
    db_file = config_dir / ".doit_reports.db"
    typer.echo(f"Using task database: {db_file}")
    
    # Import reports module and run
    try:
        from turtledrone import reports
    except ImportError as e:
        missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
        typer.echo(f"Error: Missing required module '{missing_module}' for reports generation.", err=True)
        typer.echo(f"Install reports dependencies with: pip install -e .[reports]", err=True)
        typer.echo(f"Or with poetry: poetry install --extras reports", err=True)
        raise typer.Exit(code=1)
    
    # Configure doit
    reports.DOIT_CONFIG.update({
        'check_file_uptodate': 'timestamp',
        'num_processes': 10,
        'verbosity': 2,
        'db_file': str(db_file),
    })
    
    # Clear sys.argv so doit runs all tasks
    sys.argv = [sys.argv[0]]
    
    typer.echo("Starting reports generation...")
    doit.run(reports.__dict__)
    typer.echo("✓ Reports generation completed")


@tdrone.command('clean')
def clean(
    config: str = typer.Argument(None, help="Path to config file"),
):
    """
    Clean generated outputs from drone processing.
    
    This removes cached files, temporary outputs, and task databases
    from previous processing runs.
    """
    import shutil
    import glob
    
    if config is None:
        config_file = Path().cwd() / 'fieldtrip.yml'
        if config_file.exists():
            cfg = read_config(config_file)
            drone_configs = Path(cfg['drone_path'].format(CATALOG_DIR=Path().cwd())).rglob('*_config.yml')
            for drone_config in drone_configs:
                typer.echo(f"Cleaning {drone_config.parent}...")
                _clean_drone_dir(drone_config.parent)
    else:
        config_path = Path(config)
        if config_path.exists():
            _clean_drone_dir(config_path.parent)
        else:
            typer.echo(f"Config file not found: {config}", err=True)
            raise typer.Exit(code=1)
    
    typer.echo("✓ Cleanup completed")


def _clean_drone_dir(drone_dir: Path) -> None:
    """Remove generated files from a drone directory."""
    # Remove __pycache__ directories
    for pycache in drone_dir.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
    
    # Remove .pyc files
    for pyc_file in drone_dir.rglob("*.pyc"):
        pyc_file.unlink(missing_ok=True)
    
    # Remove doit database if present
    doit_db = drone_dir / ".doit.db"
    if doit_db.exists():
        doit_db.unlink()
    
    # Remove temporary processing files (based on naming patterns)
    for pattern in ["*.tmp", "*.bak", ".cache"]:
        for file in drone_dir.glob(pattern):
            if file.is_file():
                file.unlink(missing_ok=True)
            elif file.is_dir():
                shutil.rmtree(file, ignore_errors=True)



# def import_command(
#         collection_path: str = typer.Argument(..., help="Root path to MarImBA collection."),
#         instrument_id: str = typer.Argument(None, help="MarImBA instrument ID."),
#         card_path: list[str] = typer.Argument(None, help="MarImBA instrument ID."),
#         all: bool = typer.Option(False, help="Execute the command and print logging to the terminal, but do not change any files."),
#         exiftool_path: str = typer.Option("exiftool", help="Path to exiftool"),
#         copy: bool = typer.Option(True, help="Clean source"),
#         move: bool = typer.Option(False, help="move source"),
#         cardsize:int = typer.Option(512, help="maximum card size"),
#         format_type:str = typer.Option('exfat', help="Card format type"),
#         extra: list[str] = typer.Option([], help="Extra key-value pass-through arguments."),
#         dry_run: bool = typer.Option(False, help="Execute the command and print logging to the terminal, but do not change any files."),
#         file_extension: str = typer.Option("MP4", help="extension to catalog"),
# ):
#     """
#     Import SD cards to working directory
#     """ 
#     if all and (not card_path ):
#         card_path = list_sdcards(format_type,cardsize)
#     #run_command('import_command', collection_path, instrument_id,None,extra,card_path=card_path,copy=copy,move=move,dry_run=dry_run, exiftool_path=exiftool_path,file_extension=file_extension)


# @goprobruv.command('doit')
# def doit(
#         collection_path: str = typer.Argument(None, help="Root path to MarImBA collection."),
# ):
#     makebruvs(collection_path)
    
 


if __name__ == "__main__":
    tdrone()
