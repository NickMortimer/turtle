"""Tests for LabelMe survey comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from turtledrone.labelme.compare_survey import (
    compare_annotation_directories,
    load_annotations,
    match_annotations,
    shape_to_bbox,
)


def _write_labelme_json(
    json_path: Path,
    image_name: str,
    shapes: list[dict[str, object]],
    image_width: int = 100,
    image_height: int = 100,
) -> None:
    """Write a minimal LabelMe JSON fixture."""

    payload = {
        "version": "5.0.0",
        "imagePath": image_name,
        "imageWidth": image_width,
        "imageHeight": image_height,
        "shapes": shapes,
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.unit
def test_shape_to_bbox_circle_and_polygon_are_normalized() -> None:
    """Test circle and polygon annotations convert to valid boxes."""

    circle_shape = {
        "label": "turtle",
        "shape_type": "circle",
        "points": [[10, 10], [13, 14]],
    }
    polygon_shape = {
        "label": "turtle",
        "shape_type": "polygon",
        "points": [[5, 2], [9, 1], [8, 7], [3, 6]],
    }

    circle_bbox = shape_to_bbox(circle_shape, image_width=20, image_height=20)
    polygon_bbox = shape_to_bbox(polygon_shape, image_width=20, image_height=20)

    assert circle_bbox == (5.0, 5.0, 15.0, 15.0)
    assert polygon_bbox == (3.0, 1.0, 9.0, 7.0)


@pytest.mark.unit
def test_match_annotations_is_one_to_one_and_greedy(tmp_path: Path) -> None:
    """Test the highest-IoU candidate wins when one ROI overlaps many."""

    survey_a = tmp_path / "annotator_a"
    survey_b = tmp_path / "annotator_b"
    survey_a.mkdir()
    survey_b.mkdir()

    _write_labelme_json(
        survey_a / "image_a.json",
        "shared.JPG",
        [
            {
                "label": "turtle",
                "shape_type": "rectangle",
                "points": [[10, 10], [30, 30]],
            }
        ],
    )
    _write_labelme_json(
        survey_b / "image_b.json",
        "shared.JPG",
        [
            {
                "label": "candidate_low_turtle",
                "shape_type": "rectangle",
                "points": [[8, 8], [26, 26]],
            },
            {
                "label": "candidate_high_turtle",
                "shape_type": "rectangle",
                "points": [[10, 10], [30, 30]],
            },
        ],
    )

    annotations_a, _ = load_annotations(survey_a)
    annotations_b, _ = load_annotations(survey_b)
    matches, unmatched_a, unmatched_b = match_annotations(
        annotations_a["shared.JPG"],
        annotations_b["shared.JPG"],
        iou_threshold=0.5,
    )

    assert len(matches) == 1
    assert matches[0].annotation_b.label == "candidate_high_turtle"
    assert len(unmatched_a) == 0
    assert len(unmatched_b) == 1
    assert unmatched_b[0].label == "candidate_low_turtle"


@pytest.mark.unit
def test_compare_annotation_directories_writes_csv_report(tmp_path: Path) -> None:
    """Test directory comparison writes matched and unmatched CSV rows."""

    survey_a = tmp_path / "annotator_a"
    survey_b = tmp_path / "annotator_b"
    survey_a.mkdir()
    survey_b.mkdir()

    _write_labelme_json(
        survey_a / "shared.json",
        "shared.JPG",
        [
            {
                "label": "green_turtle",
                "shape_type": "rectangle",
                "points": [[10, 10], [30, 30]],
            },
            {
                "label": "loggerhead",
                "shape_type": "polygon",
                "points": [[60, 60], [80, 60], [80, 80], [60, 80]],
            },
        ],
    )
    _write_labelme_json(
        survey_b / "shared_other_name.json",
        "shared.JPG",
        [
            {
                "label": "observer_two_turtle",
                "shape_type": "rectangle",
                "points": [[12, 12], [32, 32]],
            }
        ],
    )
    _write_labelme_json(
        survey_b / "only_b.json",
        "only_b.JPG",
        [
            {
                "label": "hawksbill_turtle",
                "shape_type": "rectangle",
                "points": [[5, 5], [15, 15]],
            }
        ],
    )

    output_csv = tmp_path / "report.csv"
    summary = compare_annotation_directories(survey_a, survey_b, output_csv)

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary.total_annotations_a == 1
    assert summary.total_annotations_b == 2
    assert summary.matched_annotations == 1
    assert summary.unmatched_annotations_a == 0
    assert summary.unmatched_annotations_b == 1
    assert summary.label_agreement_matches == 0
    assert summary.label_disagreement_matches == 1
    assert summary.images_only_in_a == 0
    assert summary.images_only_in_b == 1

    assert len(rows) == 2
    statuses = {row["status"] for row in rows}
    assert statuses == {"label_disagreement", "only_in_b"}

    matched_row = next(row for row in rows if row["status"] == "label_disagreement")
    assert matched_row["image_name"] == "shared.JPG"
    assert matched_row["label_a"] == "green_turtle"
    assert matched_row["label_b"] == "observer_two_turtle"
    assert float(matched_row["iou"]) > 0.5


@pytest.mark.unit
def test_load_annotations_ignores_non_turtle_labels(tmp_path: Path) -> None:
    """Test only labels containing 'turtle' are loaded as ROIs."""

    survey = tmp_path / "annotator"
    survey.mkdir()

    _write_labelme_json(
        survey / "mixed.json",
        "mixed.JPG",
        [
            {
                "label": "green_turtle",
                "shape_type": "rectangle",
                "points": [[10, 10], [20, 20]],
            },
            {
                "label": "ray",
                "shape_type": "rectangle",
                "points": [[30, 30], [40, 40]],
            },
        ],
    )

    annotations, _ = load_annotations(survey)

    assert "mixed.JPG" in annotations
    assert len(annotations["mixed.JPG"]) == 1
    assert annotations["mixed.JPG"][0].label == "green_turtle"


@pytest.mark.unit
def test_compare_annotation_directories_rejects_invalid_threshold(
    tmp_path: Path,
) -> None:
    """Test invalid IoU thresholds fail fast."""

    survey_a = tmp_path / "annotator_a"
    survey_b = tmp_path / "annotator_b"
    survey_a.mkdir()
    survey_b.mkdir()

    with pytest.raises(ValueError, match="IoU threshold"):
        compare_annotation_directories(
            survey_a,
            survey_b,
            tmp_path / "report.csv",
            iou_threshold=1.5,
        )