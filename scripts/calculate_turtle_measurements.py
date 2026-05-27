#!/usr/bin/env python3
"""Calculate turtle length and width from LabelMe JSON line annotations.

This script scans a directory for JSON files, extracts line measurements with
labels "turtle_length" and "turtle_width", and writes one row per JSON file
to a CSV output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


@dataclass
class MeasurementRecord:
    """Store extracted turtle measurements for one JSON annotation file."""

    json_file: str
    image_path: Optional[str]
    image_width: Optional[int]
    image_height: Optional[int]
    turtle_length_px: Optional[float]
    turtle_width_px: Optional[float]
    length_to_width_ratio: Optional[float]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for measurement extraction."""

    parser = argparse.ArgumentParser(
        description=(
            "Calculate turtle length and width from LabelMe JSON files in a "
            "directory."
        )
    )
    parser.add_argument(
        "input_directory",
        type=Path,
        help="Directory containing JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("turtle_measurements.csv"),
        help="Output CSV file path (default: turtle_measurements.csv).",
    )
    parser.add_argument(
        "--length-label",
        "--length",
        dest="length_label",
        default="turtle_length",
        help="Label name used for turtle length lines.",
    )
    parser.add_argument(
        "--width-label",
        "--width",
        dest="width_label",
        default="turtle_width",
        help="Label name used for turtle width lines.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only search the top-level directory for JSON files.",
    )
    return parser.parse_args()


def find_json_files(directory: Path, recursive: bool) -> List[Path]:
    """Return sorted JSON file paths in a directory."""

    if recursive:
        return sorted(directory.rglob("*.json"))
    return sorted(directory.glob("*.json"))


def line_length(points: Sequence[Sequence[float]]) -> Optional[float]:
    """Calculate Euclidean length for a two-point line annotation."""

    if len(points) != 2:
        return None

    first, second = points
    if len(first) < 2 or len(second) < 2:
        return None

    x1, y1 = first[0], first[1]
    x2, y2 = second[0], second[1]
    return float(math.hypot(x2 - x1, y2 - y1))


def build_record(
    json_file: Path,
    length_label: str,
    width_label: str,
) -> MeasurementRecord:
    """Build one measurement record from a LabelMe JSON file."""

    with json_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    length_values: List[float] = []
    width_values: List[float] = []

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "line":
            continue

        points = shape.get("points", [])
        value = line_length(points)
        if value is None:
            continue

        label = shape.get("label")
        if label == length_label:
            length_values.append(value)
        elif label == width_label:
            width_values.append(value)

    turtle_length_px = max(length_values) if length_values else None
    turtle_width_px = max(width_values) if width_values else None

    if turtle_length_px is not None and turtle_width_px not in (None, 0.0):
        ratio = turtle_length_px / turtle_width_px
    else:
        ratio = None

    image_width = data.get("imageWidth")
    image_height = data.get("imageHeight")

    return MeasurementRecord(
        json_file=str(json_file),
        image_path=data.get("imagePath"),
        image_width=int(image_width) if image_width is not None else None,
        image_height=int(image_height) if image_height is not None else None,
        turtle_length_px=turtle_length_px,
        turtle_width_px=turtle_width_px,
        length_to_width_ratio=ratio,
    )


def iter_records(
    json_files: Iterable[Path],
    length_label: str,
    width_label: str,
) -> Iterable[MeasurementRecord]:
    """Yield measurement records while skipping malformed JSON files."""

    for json_file in json_files:
        try:
            yield build_record(json_file, length_label, width_label)
        except json.JSONDecodeError as error:
            print(f"Skipping invalid JSON file {json_file}: {error}")
        except UnicodeDecodeError as error:
            print(f"Skipping non-UTF8 JSON file {json_file}: {error}")
        except OSError as error:
            print(f"Skipping unreadable file {json_file}: {error}")


def write_csv(output_file: Path, records: Sequence[MeasurementRecord]) -> None:
    """Write measurement records to a CSV file."""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "json_file",
        "image_path",
        "image_width",
        "image_height",
        "turtle_length_px",
        "turtle_width_px",
        "length_to_width_ratio",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def main() -> int:
    """Run the CLI workflow and return process exit code."""

    args = parse_args()
    input_directory = args.input_directory

    if not input_directory.exists() or not input_directory.is_dir():
        print(f"Input directory does not exist or is not a directory: "
              f"{input_directory}")
        return 1

    json_files = find_json_files(
        input_directory,
        recursive=not args.no_recursive,
    )

    if not json_files:
        print(f"No JSON files found in: {input_directory}")
        return 1

    records = list(
        iter_records(
            json_files,
            length_label=args.length_label,
            width_label=args.width_label,
        )
    )
    write_csv(args.output, records)

    print(f"Processed {len(json_files)} JSON files")
    print(f"Wrote {len(records)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
