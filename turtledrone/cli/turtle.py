"""Turtle analysis CLI commands."""

import os
import sys
from pathlib import Path
from typing import Optional

import doit
import typer

import turtledrone.config as config
from turtledrone.cli import turtle_tasks

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
    
    # Run the app
    app_instance()


def app():
    """CLI entry point."""
    main()


if __name__ == "__main__":
    app()
