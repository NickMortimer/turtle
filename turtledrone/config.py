"""Configuration management for turtle survey processing.

Loads and manages YAML configuration files with path expansion and default values.
Provides both class-based and module-level API for backward compatibility.
"""

from __future__ import annotations

import logging
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Manages survey processing configuration."""

    def __init__(self, path: Optional[str | Path] = None):
        """
        Initialize configuration.

        Parameters
        ----------
        path : str | Path, optional
            Path to YAML configuration file. If None, user will be prompted.
        """
        self.cfg: dict = {}
        self.catalog_dir: Optional[Path] = None

        if path:
            self.load(path)
        else:
            path = self._prompt_config_file()
            if path:
                self.load(path)

    @staticmethod
    def _prompt_config_file() -> Optional[Path]:
        """Prompt user to select configuration file."""
        try:
            root = tk.Tk()
            root.withdraw()
            config_path = filedialog.askopenfilename(
                title="Select configuration YAML file",
                filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            )
            root.destroy()
            return Path(config_path) if config_path else None
        except Exception as e:
            logger.warning(f"Could not prompt for config file: {e}")
            return None

    def load(self, path: str | Path) -> None:
        """
        Load configuration from YAML file.

        Parameters
        ----------
        path : str | Path
            Path to YAML configuration file.

        Raises
        ------
        FileNotFoundError
            If config file does not exist.
        yaml.YAMLError
            If YAML parsing fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        self.catalog_dir = path.parent
        logger.debug(f"Loading configuration from {path}")

        try:
            with open(path, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    self.cfg.update(loaded)
                logger.info(f"Loaded config with keys: {list(self.cfg.keys())}")
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Parameters
        ----------
        key : str
            Configuration key.
        default : Any, optional
            Default value if key not found.

        Returns
        -------
        Any
            Configuration value or default.
        """
        return self.cfg.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Parameters
        ----------
        key : str
            Configuration key.
        value : Any
            Value to set.
        """
        self.cfg[key] = value

    def get_url(self, key: str) -> Path:
        """
        Get configuration path, expanding catalog directory reference.

        Parameters
        ----------
        key : str
            Configuration key pointing to a path string.

        Returns
        -------
        Path
            Resolved Path object.

        Raises
        ------
        KeyError
            If key not found in configuration.
        ValueError
            If catalog_dir not set and {CATALOG_DIR} placeholder found.
        """
        if key not in self.cfg:
            raise KeyError(f"Configuration key not found: {key}")

        value = str(self.cfg[key])

        if "{CATALOG_DIR}" in value:
            if self.catalog_dir is None:
                raise ValueError(
                    "Configuration references {CATALOG_DIR} but catalog_dir not set"
                )
            value = value.format(CATALOG_DIR=self.catalog_dir)

        return Path(value)

    def get_destination(self, file_path: str | Path) -> Path:
        """
        Compute destination directory for a file based on naming convention.

        Expected filename: COUNTRY_SITE_SITECODE_*.

        Parameters
        ----------
        file_path : str | Path
            File path to parse.

        Returns
        -------
        Path
            Destination directory path.
        """
        parts = Path(file_path).name.split("_")
        if len(parts) < 3:
            raise ValueError(f"File name does not match expected pattern: {file_path}")

        country = parts[0]
        site = parts[1]
        sitecode = "_".join(parts[1:3])

        return self.get_url("output") / country / site / sitecode

    def save(self, path: Optional[str | Path] = None) -> None:
        """
        Save current configuration to YAML file.

        Parameters
        ----------
        path : str | Path, optional
            Output path. If None, uses original loaded path.

        Raises
        ------
        ValueError
            If no path provided and original path not remembered.
        """
        if not path and not self.catalog_dir:
            raise ValueError("No path provided and original config path not remembered")

        output_path = Path(path or self.catalog_dir)
        logger.debug(f"Saving configuration to {output_path}")

        with open(output_path, "w") as f:
            yaml.safe_dump(self.cfg, f, default_flow_style=False)

    def __repr__(self) -> str:
        """Return string representation."""
        keys = ", ".join(self.cfg.keys())
        return f"Config({keys})"


# Module-level singleton for backward compatibility
_instance: Optional[Config] = None


def init(path: Optional[str | Path] = None) -> Config:
    """
    Initialize or get the global config instance.

    Parameters
    ----------
    path : str | Path, optional
        Path to config file. If not provided, will prompt user or use cached instance.

    Returns
    -------
    Config
        Configuration instance.
    """
    global _instance
    if _instance is None:
        _instance = Config(path)
    return _instance


def read_config(path: Optional[str | Path] = None) -> Config:
    """
    Read and initialize global configuration.

    Parameters
    ----------
    path : str | Path, optional
        Path to config file.

    Returns
    -------
    Config
        Configuration instance.
    """
    return init(path)


def geturl(key: str) -> Path:
    """
    Get configuration URL/path by key (module-level convenience function).

    Parameters
    ----------
    key : str
        Configuration key.

    Returns
    -------
    Path
        Resolved path.
    """
    return init().get_url(key)


def get(key: str, default: Any = None) -> Any:
    """
    Get configuration value by key (module-level convenience function).

    Parameters
    ----------
    key : str
        Configuration key.
    default : Any, optional
        Default value if key not found.

    Returns
    -------
    Any
        Configuration value.
    """
    return init().get(key, default)


# Module-level attribute access for backward compatibility with config.cfg
class _ConfigProxy:
    """Proxy object to allow config.cfg.['key'] syntax."""

    def __getitem__(self, key: str) -> Any:
        """Get item from config."""
        return init().cfg[key]

    def __getattr__(self, key: str) -> Any:
        """Get attribute from config."""
        return init().cfg.get(key)

    def __repr__(self) -> str:
        """Return string representation."""
        return repr(init().cfg)


cfg = _ConfigProxy()
