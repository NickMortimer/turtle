"""Turtle analysis CLI commands."""

import os
import sys
from pathlib import Path
from typing import Optional

import doit
import typer

import turtledrone.config as config
from turtledrone.cli import turtle_tasks
from turtledrone.labelme.compare_survey import (
    compare_annotation_directories,
    format_summary,
)

# Note: reports.py tasks are not yet compatible with this CLI.
# They require additional configuration and intermediate files that aren't
# part of the core turtle counting pipeline.

# Module-level variable to store CLI output_dir override
_output_dir_override: Optional[Path] = None

# doit configuration dictionary (will be populated by count command)
DOIT_CONFIG = {}

# Create Typer app for multi-command support
app_instance = typer.Typer(
    name="turtle",
    help="Turtle survey analysis and processing pipeline",
    no_args_is_help=True,
)


# Import task definitions from turtle_tasks module
task_process_labelme = turtle_tasks.task_process_labelme
task_calculate_positions = turtle_tasks.task_calculate_positions
task_process_turtles = turtle_tasks.task_process_turtles
task_process_turtles_totals = turtle_tasks.task_process_turtles_totals
task_merge_turtle_totals = turtle_tasks.task_merge_turtle_totals
task_check_survey = turtle_tasks.task_check_survey
task_turtles_report = turtle_tasks.task_turtles_report
task_calculate_gsd = turtle_tasks.task_calculate_gsd
task_measure_turtles = turtle_tasks.task_measure_turtles
task_concat_gsd = turtle_tasks.task_concat_gsd
task_calculate_true_sizes = turtle_tasks.task_calculate_true_sizes


def process(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to YAML configuration file"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Override output directory"),
) -> None:
    """Run turtle counting pipeline.

    This command processes LabelMe annotations, calculates turtle positions,
    clusters detections, and generates reports.
    """
    global _output_dir_override

    if config_path is None:
        typer.echo("Error: --config option is required", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading configuration from: {config_path}")
    
    if output_dir:
        _output_dir_override = output_dir.resolve()
        typer.echo(f"Using output directory: {_output_dir_override}")

    # Load config from specified file - this initializes the global config instance
    cfg = config.read_config(config_path, prompt_if_none=False)
    typer.echo(f"Configuration loaded with keys: {list(cfg.cfg.keys())}")

    # Apply override if provided
    if _output_dir_override:
        config.init().set("output", str(_output_dir_override))

    # Change working directory to config file's parent directory
    config_dir = Path(config_path).parent.resolve()
    os.chdir(config_dir)
    typer.echo(f"Working directory: {config_dir}")
    
    # Set up per-country doit database file in the config directory
    db_file = config_dir / ".doit.db"
    typer.echo(f"Using task database: {db_file}")

    # Store config in module-level variable so DOIT_CONFIG can access it
    globals()["_doit_config"] = {
        "num_processes": 10,
        "verbosity": 2,
        "db_file": str(db_file),
    }

    # Update the DOIT_CONFIG dictionary
    globals()["DOIT_CONFIG"] = globals()["_doit_config"]

    # Clear sys.argv so doit runs all tasks
    sys.argv = [sys.argv[0]]

    typer.echo("Starting task execution...")
    # Run doit tasks
    doit.run(globals())


def clean(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to YAML configuration file"),
) -> None:
    """Clean generated task outputs.

    This command removes all generated files (CSV files, reports, etc.) from previous runs.
    The doit task database (.doit.db) is also removed to reset task tracking.
    """
    if config_path is None:
        typer.echo("Error: --config option is required", err=True)
        raise typer.Exit(code=1)

    config.read_config(config_path, prompt_if_none=False)
    
    # Change working directory to config file's parent directory
    config_dir = Path(config_path).parent.resolve()
    os.chdir(config_dir)
    typer.echo(f"Working directory: {config_dir}")
    
    db_file = config_dir / ".doit.db"
    
    # Set up doit config for cleaning
    globals()["_doit_config"] = {
        "num_processes": 10,
        "verbosity": 2,
        "db_file": str(db_file),
    }
    globals()["DOIT_CONFIG"] = globals()["_doit_config"]
    
    # Clear sys.argv and add clean flag
    sys.argv = [sys.argv[0], "clean"]
    
    typer.echo(f"Cleaning generated outputs from {config_path}...")
    doit.run(globals())
    typer.echo("✓ Cleanup completed")


def getsizes(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Override output directory"
    ),
) -> None:
    """Run only turtle size extraction from LabelMe JSON files."""
    global _output_dir_override

    if config_path is None:
        typer.echo("Error: --config option is required", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading configuration from: {config_path}")

    if output_dir:
        _output_dir_override = output_dir.resolve()
        typer.echo(f"Using output directory: {_output_dir_override}")

    cfg = config.read_config(config_path, prompt_if_none=False)
    typer.echo(f"Configuration loaded with keys: {list(cfg.cfg.keys())}")

    if _output_dir_override:
        config.init().set("output", str(_output_dir_override))

    config_dir = Path(config_path).parent.resolve()
    os.chdir(config_dir)
    typer.echo(f"Working directory: {config_dir}")

    db_file = config_dir / ".doit.db"
    typer.echo(f"Using task database: {db_file}")

    globals()["_doit_config"] = {
        "num_processes": 10,
        "verbosity": 2,
        "db_file": str(db_file),
    }
    globals()["DOIT_CONFIG"] = globals()["_doit_config"]

    sys.argv = [sys.argv[0], "calculate_gsd", "measure_turtles", "concat_gsd", "calculate_true_sizes"]

    typer.echo("Starting turtle size extraction...")
    doit.run(globals())


def compare_surveys(
    annotations_a: Path = typer.Argument(
        ..., help="Directory of LabelMe JSON files from annotator A"
    ),
    annotations_b: Path = typer.Argument(
        ..., help="Directory of LabelMe JSON files from annotator B"
    ),
    output_csv: Path = typer.Option(
        Path("compare-survey-report.csv"),
        "--output",
        "-o",
        help="Path to the CSV report to write",
    ),
    iou_threshold: float = typer.Option(
        0.5,
        "--iou-threshold",
        min=0.0,
        max=1.0,
        help="Minimum IoU required to match two ROIs",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose debug logging",
    ),
    skip_errors: bool = typer.Option(
        False,
        "--skip-errors",
        help="Skip files with parse errors instead of failing",
    ),
    workers: int = typer.Option(
        8,
        "--workers",
        "-w",
        min=1,
        max=64,
        help="Number of parallel worker threads to load JSON files",
    ),
) -> None:
    """Compare two LabelMe annotation directories by ROI overlap."""
    import logging
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(name)s - %(levelname)s - %(message)s"
        )

    try:
        summary = compare_annotation_directories(
            annotations_a.resolve(),
            annotations_b.resolve(),
            output_csv.resolve(),
            iou_threshold=iou_threshold,
            skip_errors=skip_errors,
            max_workers=workers,
        )
        typer.echo(format_summary(summary, output_csv.resolve()))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


def main():
    """CLI entry point with pre-processing for key=value argument syntax."""
    # Pre-process arguments to convert key=value format to --key value format
    # This allows both 'turtle --config file.yaml' and 'turtle config=file.yaml' syntax
    processed_args = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if "=" in arg and not arg.startswith("-"):
            # This is a key=value format argument
            key, value = arg.split("=", 1)
            processed_args.append(f"--{key}")
            processed_args.append(value)
        else:
            processed_args.append(arg)
        i += 1
    
    sys.argv = processed_args
    
    # Add commands to the app
    app_instance.command("process")(process)
    app_instance.command("clean")(clean)
    app_instance.command("getsizes")(getsizes)
    app_instance.command("compare-surveys")(compare_surveys)
    
    # Run the app
    app_instance()


def app():
    """CLI entry point."""
    main()


if __name__ == "__main__":
    app()
