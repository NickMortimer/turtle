"""Compare LabelMe annotations from two survey directories."""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoiAnnotation:
    """Normalized representation of a LabelMe ROI."""

    image_name: str
    json_path: Path
    label: str
    shape_type: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float

    def as_bbox(self) -> tuple[float, float, float, float]:
        """Return the ROI as an ``(x1, y1, x2, y2)`` tuple."""

        return (
            self.bbox_x1,
            self.bbox_y1,
            self.bbox_x2,
            self.bbox_y2,
        )


@dataclass(frozen=True)
class MatchedPair:
    """Pairing of two annotations that overlap by IoU."""

    annotation_a: RoiAnnotation
    annotation_b: RoiAnnotation
    iou: float

    @property
    def labels_match(self) -> bool:
        """Return ``True`` when both annotations use the same label."""

        return self.annotation_a.label == self.annotation_b.label


@dataclass(frozen=True)
class ComparisonSummary:
    """Aggregate counts from a directory comparison."""

    total_annotations_a: int
    total_annotations_b: int
    matched_annotations: int
    unmatched_annotations_a: int
    unmatched_annotations_b: int
    label_agreement_matches: int
    label_disagreement_matches: int
    images_only_in_a: int
    images_only_in_b: int


def _normalize_image_key(json_path: Path, payload: dict[str, Any]) -> str:
    """Return a stable per-image key used to pair files across folders."""

    image_path = payload.get("imagePath")
    if image_path:
        return Path(str(image_path)).name
    return f"{json_path.stem}.JPG"


def _coerce_point(point: Any) -> tuple[float, float]:
    """Validate and normalize a two-value point."""

    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"Invalid point value: {point!r}")
    return float(point[0]), float(point[1])


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a coordinate to the image extent."""

    return max(lower, min(value, upper))


def _is_turtle_label(label: str) -> bool:
    """Return ``True`` when a label contains the word 'turtle'."""

    return "turtle" in label.casefold()


def shape_to_bbox(
    shape: dict[str, Any],
    image_width: Optional[float],
    image_height: Optional[float],
) -> tuple[float, float, float, float]:
    """Convert a LabelMe shape into a normalized bounding box.
    
    Robustly handles circles, rectangles, and polygons by converting all to
    image-space bounding boxes. Tolerates malformed data by defaulting to
    sensible behavior rather than failing.
    """

    points = shape.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("Shape is missing point data")

    shape_type = str(shape.get("shape_type") or "polygon")
    x_values: list[float] = []
    y_values: list[float] = []

    if shape_type == "circle" and len(points) >= 2:
        try:
            center_x, center_y = _coerce_point(points[0])
            edge_x, edge_y = _coerce_point(points[1])
            radius = ((center_x - edge_x) ** 2 + (center_y - edge_y) ** 2) ** 0.5
            x_values = [center_x - radius, center_x + radius]
            y_values = [center_y - radius, center_y + radius]
        except (ValueError, TypeError):
            points = points[:1]  # Fall through to treat as single point
    
    if not x_values:
        coerced_points = []
        for point in points:
            try:
                coerced_points.append(_coerce_point(point))
            except (ValueError, TypeError):
                continue
        
        if not coerced_points:
            raise ValueError("Shape has no valid point data")
        
        x_values = [point[0] for point in coerced_points]
        y_values = [point[1] for point in coerced_points]

    x1 = min(x_values)
    y1 = min(y_values)
    x2 = max(x_values)
    y2 = max(y_values)
    
    if x2 <= x1:
        x2 = x1 + 1.0
    if y2 <= y1:
        y2 = y1 + 1.0

    if image_width is not None:
        x1 = _clamp(x1, 0.0, float(image_width))
        x2 = _clamp(x2, 0.0, float(image_width))
    if image_height is not None:
        y1 = _clamp(y1, 0.0, float(image_height))
        y2 = _clamp(y2, 0.0, float(image_height))

    return x1, y1, x2, y2


def load_annotations(directory: Path, skip_errors: bool = False, max_workers: int = 8) -> tuple[dict[str, list[RoiAnnotation]], dict[str, Path]]:
    """Load all LabelMe JSON annotations from a directory using parallel workers.
    
    Parameters:
        directory: Path to directory containing LabelMe JSON files
        skip_errors: If True, skip files with parse errors and continue. If False, raise on first error.
        max_workers: Number of parallel worker threads to use for loading JSON files.

    Notes:
        Only shapes whose ``label`` contains 'turtle' (case-insensitive)
        are included in the returned annotations.
    """

    if not directory.exists():
        raise FileNotFoundError(f"Annotation directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Annotation path is not a directory: {directory}")

    annotations_by_image: dict[str, list[RoiAnnotation]] = {}
    json_by_image: dict[str, Path] = {}
    skipped_files: list[tuple[Path, str]] = []

    json_files = sorted(directory.glob("*.json"))
    logger.debug(f"Found {len(json_files)} JSON files in {directory}")
    
    def _load_single_json(json_path: Path) -> tuple[dict[str, Any], Path, list[str]]:
        """Load a single JSON file and return payload, path, and any errors."""
        errors = []
        try:
            logger.debug(f"Loading JSON file: {json_path}")
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload, json_path, errors
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in {json_path}: {e}"
            errors.append(error_msg)
            if not skip_errors:
                raise ValueError(error_msg) from e
            logger.warning(error_msg)
            return {}, json_path, errors
        except Exception as e:
            error_msg = f"Failed to read {json_path}: {e}"
            errors.append(error_msg)
            if not skip_errors:
                raise ValueError(error_msg) from e
            logger.warning(error_msg)
            return {}, json_path, errors
    
    # Load all JSON files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_load_single_json, json_path): json_path for json_path in json_files}
        
        for future in as_completed(futures):
            try:
                payload, json_path, errors = future.result()
            except Exception as e:
                if not skip_errors:
                    raise
                logger.warning(f"Worker error loading {json_path}: {e}")
                continue
            
            if errors:
                skipped_files.extend((json_path, err) for err in errors)
                continue
            
            if not payload:
                continue
            
            if not isinstance(payload, dict):
                error_msg = f"JSON file does not contain an object: {json_path}"
                if skip_errors:
                    logger.warning(error_msg)
                    skipped_files.append((json_path, error_msg))
                    continue
                raise ValueError(error_msg)

            image_name = _normalize_image_key(json_path, payload)
            logger.debug(f"Normalized image name: {image_name}")
            json_by_image.setdefault(image_name, json_path)

            image_width = payload.get("imageWidth")
            image_height = payload.get("imageHeight")
            logger.debug(f"Image dimensions: {image_width}x{image_height}")
            shapes = payload.get("shapes", [])
            if not isinstance(shapes, list):
                error_msg = f"'shapes' must be a list in {json_path}"
                if skip_errors:
                    logger.warning(error_msg)
                    skipped_files.append((json_path, error_msg))
                    continue
                raise ValueError(error_msg)

            logger.debug(f"Found {len(shapes)} shapes in {json_path}")
            for shape_idx, shape in enumerate(shapes):
                if not isinstance(shape, dict):
                    if skip_errors:
                        continue
                    raise ValueError(f"Invalid shape entry in {json_path}: {shape!r}")

                label = str(shape.get("label") or "")
                if not _is_turtle_label(label):
                    continue
                
                try:
                    bbox = shape_to_bbox(shape, image_width, image_height)
                    annotations_by_image.setdefault(image_name, []).append(
                        RoiAnnotation(
                            image_name=image_name,
                            json_path=json_path,
                            label=label,
                            shape_type=str(shape.get("shape_type") or "polygon"),
                            bbox_x1=bbox[0],
                            bbox_y1=bbox[1],
                            bbox_x2=bbox[2],
                            bbox_y2=bbox[3],
                        )
                    )
                except (ValueError, TypeError) as e:
                    if skip_errors:
                        continue
                    raise ValueError(f"Failed to convert shape {shape_idx} in {json_path}: {e}")

    if skipped_files:
        logger.info(f"Skipped {len(skipped_files)} files/shapes due to errors")
        for skip_path, skip_reason in skipped_files[:10]:  # Log first 10
            logger.debug(f"  Skipped: {skip_path} ({skip_reason})")
        if len(skipped_files) > 10:
            logger.debug(f"  ... and {len(skipped_files) - 10} more")
    
    return annotations_by_image, json_by_image


def compute_iou(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    """Calculate intersection-over-union for two boxes."""

    left = max(bbox_a[0], bbox_b[0])
    top = max(bbox_a[1], bbox_b[1])
    right = min(bbox_a[2], bbox_b[2])
    bottom = min(bbox_a[3], bbox_b[3])

    if right <= left or bottom <= top:
        return 0.0

    intersection_area = (right - left) * (bottom - top)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union_area = area_a + area_b - intersection_area

    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area


def match_annotations(
    annotations_a: Iterable[RoiAnnotation],
    annotations_b: Iterable[RoiAnnotation],
    iou_threshold: float,
) -> tuple[list[MatchedPair], list[RoiAnnotation], list[RoiAnnotation]]:
    """Match annotations one-to-one using descending IoU."""

    list_a = list(annotations_a)
    list_b = list(annotations_b)
    candidates: list[tuple[float, int, int]] = []

    for index_a, annotation_a in enumerate(list_a):
        for index_b, annotation_b in enumerate(list_b):
            iou = compute_iou(annotation_a.as_bbox(), annotation_b.as_bbox())
            if iou >= iou_threshold:
                candidates.append((iou, index_a, index_b))

    candidates.sort(reverse=True)

    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[MatchedPair] = []

    for iou, index_a, index_b in candidates:
        if index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        matches.append(
            MatchedPair(
                annotation_a=list_a[index_a],
                annotation_b=list_b[index_b],
                iou=iou,
            )
        )

    unmatched_a = [annotation for index, annotation in enumerate(list_a) if index not in used_a]
    unmatched_b = [annotation for index, annotation in enumerate(list_b) if index not in used_b]
    return matches, unmatched_a, unmatched_b


def _matched_row(match: MatchedPair) -> dict[str, Any]:
    """Build a CSV row for a matched pair."""

    status = "matched" if match.labels_match else "label_disagreement"
    return {
        "image_name": match.annotation_a.image_name,
        "status": status,
        "iou": f"{match.iou:.6f}",
        "label_agreement": match.labels_match,
        "json_a": str(match.annotation_a.json_path),
        "label_a": match.annotation_a.label,
        "shape_type_a": match.annotation_a.shape_type,
        "bbox_a_x1": match.annotation_a.bbox_x1,
        "bbox_a_y1": match.annotation_a.bbox_y1,
        "bbox_a_x2": match.annotation_a.bbox_x2,
        "bbox_a_y2": match.annotation_a.bbox_y2,
        "json_b": str(match.annotation_b.json_path),
        "label_b": match.annotation_b.label,
        "shape_type_b": match.annotation_b.shape_type,
        "bbox_b_x1": match.annotation_b.bbox_x1,
        "bbox_b_y1": match.annotation_b.bbox_y1,
        "bbox_b_x2": match.annotation_b.bbox_x2,
        "bbox_b_y2": match.annotation_b.bbox_y2,
    }


def _unmatched_row(
    annotation: RoiAnnotation,
    side: str,
) -> dict[str, Any]:
    """Build a CSV row for an unmatched annotation."""

    if side == "a":
        return {
            "image_name": annotation.image_name,
            "status": "only_in_a",
            "iou": "",
            "label_agreement": "",
            "json_a": str(annotation.json_path),
            "label_a": annotation.label,
            "shape_type_a": annotation.shape_type,
            "bbox_a_x1": annotation.bbox_x1,
            "bbox_a_y1": annotation.bbox_y1,
            "bbox_a_x2": annotation.bbox_x2,
            "bbox_a_y2": annotation.bbox_y2,
            "json_b": "",
            "label_b": "",
            "shape_type_b": "",
            "bbox_b_x1": "",
            "bbox_b_y1": "",
            "bbox_b_x2": "",
            "bbox_b_y2": "",
        }

    return {
        "image_name": annotation.image_name,
        "status": "only_in_b",
        "iou": "",
        "label_agreement": "",
        "json_a": "",
        "label_a": "",
        "shape_type_a": "",
        "bbox_a_x1": "",
        "bbox_a_y1": "",
        "bbox_a_x2": "",
        "bbox_a_y2": "",
        "json_b": str(annotation.json_path),
        "label_b": annotation.label,
        "shape_type_b": annotation.shape_type,
        "bbox_b_x1": annotation.bbox_x1,
        "bbox_b_y1": annotation.bbox_y1,
        "bbox_b_x2": annotation.bbox_x2,
        "bbox_b_y2": annotation.bbox_y2,
    }


def compare_annotation_directories(
    directory_a: Path,
    directory_b: Path,
    output_csv: Path,
    iou_threshold: float = 0.5,
    skip_errors: bool = False,
    max_workers: int = 8,
) -> ComparisonSummary:
    """Compare two LabelMe annotation directories and write a CSV report.
    
    Parameters:
        directory_a: First annotation directory
        directory_b: Second annotation directory
        output_csv: Output CSV file path
        iou_threshold: Minimum IoU required to match two ROIs
        skip_errors: If True, skip files with parse errors. If False, raise on first error.
        max_workers: Number of threads for parallel JSON loading (default 8)

    Notes:
        The comparison only includes ROIs whose labels contain 'turtle'
        (case-insensitive).
    """

    if iou_threshold <= 0.0 or iou_threshold > 1.0:
        raise ValueError("IoU threshold must be greater than 0 and at most 1")

    annotations_a, jsons_a = load_annotations(directory_a, skip_errors=skip_errors, max_workers=max_workers)
    annotations_b, jsons_b = load_annotations(directory_b, skip_errors=skip_errors, max_workers=max_workers)

    all_images = sorted(set(annotations_a) | set(annotations_b) |
                        set(jsons_a) | set(jsons_b))

    fieldnames = [
        "image_name",
        "status",
        "iou",
        "label_agreement",
        "json_a",
        "label_a",
        "shape_type_a",
        "bbox_a_x1",
        "bbox_a_y1",
        "bbox_a_x2",
        "bbox_a_y2",
        "json_b",
        "label_b",
        "shape_type_b",
        "bbox_b_x1",
        "bbox_b_y1",
        "bbox_b_x2",
        "bbox_b_y2",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    matched_count = 0
    unmatched_a_count = 0
    unmatched_b_count = 0
    label_agreement_count = 0
    label_disagreement_count = 0

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for image_name in all_images:
            image_annotations_a = annotations_a.get(image_name, [])
            image_annotations_b = annotations_b.get(image_name, [])
            matches, unmatched_a, unmatched_b = match_annotations(
                image_annotations_a,
                image_annotations_b,
                iou_threshold=iou_threshold,
            )

            for match in matches:
                matched_count += 1
                if match.labels_match:
                    label_agreement_count += 1
                else:
                    label_disagreement_count += 1
                writer.writerow(_matched_row(match))

            for annotation in unmatched_a:
                unmatched_a_count += 1
                writer.writerow(_unmatched_row(annotation, side="a"))

            for annotation in unmatched_b:
                unmatched_b_count += 1
                writer.writerow(_unmatched_row(annotation, side="b"))

    total_annotations_a = sum(len(items) for items in annotations_a.values())
    total_annotations_b = sum(len(items) for items in annotations_b.values())
    images_only_in_a = len(set(jsons_a) - set(jsons_b))
    images_only_in_b = len(set(jsons_b) - set(jsons_a))

    logger.debug(
        f"Comparison complete: \n"
        f"  Total A: {total_annotations_a} annotations \n"
        f"  Total B: {total_annotations_b} annotations \n"
        f"  Matched: {matched_count} \n"
        f"  Unmatched A: {unmatched_a_count} \n"
        f"  Unmatched B: {unmatched_b_count}"
    )

    return ComparisonSummary(
        total_annotations_a=total_annotations_a,
        total_annotations_b=total_annotations_b,
        matched_annotations=matched_count,
        unmatched_annotations_a=unmatched_a_count,
        unmatched_annotations_b=unmatched_b_count,
        label_agreement_matches=label_agreement_count,
        label_disagreement_matches=label_disagreement_count,
        images_only_in_a=images_only_in_a,
        images_only_in_b=images_only_in_b,
    )


def format_summary(summary: ComparisonSummary, output_csv: Path) -> str:
    """Return a human-readable comparison summary."""

    return "\n".join(
        [
            f"Wrote comparison report to: {output_csv}",
            f"Total annotations in A: {summary.total_annotations_a}",
            f"Total annotations in B: {summary.total_annotations_b}",
            f"Matched annotations: {summary.matched_annotations}",
            f"Unmatched annotations in A: {summary.unmatched_annotations_a}",
            f"Unmatched annotations in B: {summary.unmatched_annotations_b}",
            (
                "Matched pairs with same label: "
                f"{summary.label_agreement_matches}"
            ),
            (
                "Matched pairs with different labels: "
                f"{summary.label_disagreement_matches}"
            ),
            f"Images only in A: {summary.images_only_in_a}",
            f"Images only in B: {summary.images_only_in_b}",
        ]
    )
