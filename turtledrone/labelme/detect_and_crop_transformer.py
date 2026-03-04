"""Detect and crop objects using a transformer detector checkpoint (.pth).

This script mirrors the existing detect-and-crop flow but uses a torchvision
DETR model loaded from a local .pth checkpoint.
"""

from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
import torch
import typer
from PIL import Image
from tqdm import tqdm

app = typer.Typer()


def _resolve_output_dir(
    output_root: Path,
    model_path: Path,
    run_number: Optional[int],
) -> Path:
    """Resolve run-specific output directory.

    Parameters
    ----------
    output_root : Path
        Base output directory.
    model_path : Path
        Model checkpoint path.
    run_number : Optional[int]
        Optional run number used to continue/restart runs.

    Returns
    -------
    Path
        Output directory for this run.
    """
    model_stem = model_path.stem
    if run_number is not None:
        output_dir = output_root / f"{model_stem}_run{run_number:02d}"
    else:
        output_dir = output_root / model_stem
        run_idx = 1
        while output_dir.exists():
            output_dir = output_root / f"{model_stem}_run{run_idx:02d}"
            run_idx += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _collect_input_dirs(
    image_dirs: Optional[list[Path]],
    image_dir_file: Optional[Path],
) -> list[Path]:
    """Collect directories from CLI args and optional text file."""
    dirs_to_process: list[Path] = []
    if image_dirs:
        dirs_to_process.extend(image_dirs)
    if image_dir_file is not None:
        with image_dir_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                candidate = line.strip()
                if candidate:
                    dirs_to_process.append(Path(candidate))
    return dirs_to_process


def _load_class_names(class_names_file: Optional[Path]) -> dict[int, str]:
    """Load class names from file where each line is one class name."""
    if class_names_file is None:
        return {}
    mapping: dict[int, str] = {}
    with class_names_file.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            label = line.strip()
            if label:
                mapping[index] = label
    return mapping


def _build_detr_model(arch: str, num_classes: int) -> torch.nn.Module:
    """Build a torchvision DETR model architecture for inference."""
    try:
        from torchvision.models.detection import (
            detr_resnet50,
            detr_resnet50_dc5,
        )
    except Exception as exc:  # pragma: no cover - import-time dependency issue
        raise RuntimeError(
            "torchvision with detection models is required for DETR inference"
        ) from exc

    if arch == "detr_resnet50":
        return detr_resnet50(weights=None, weights_backbone=None,
                             num_classes=num_classes)
    if arch == "detr_resnet50_dc5":
        return detr_resnet50_dc5(weights=None, weights_backbone=None,
                                 num_classes=num_classes)

    raise ValueError(f"Unsupported architecture: {arch}")


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Extract a model state dict from common checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Could not find state dict in checkpoint. Expected one of: "
        "state_dict, model_state_dict, model."
    )


def _normalize_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove common wrappers like DistributedDataParallel 'module.' prefix."""
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        normalized[new_key] = value
    return normalized


def _predict_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    conf_threshold: float,
) -> list[tuple[int, float, list[int]]]:
    """Run inference for one image and return filtered predictions."""
    try:
        import torchvision.transforms.functional as tvf
    except Exception as exc:  # pragma: no cover - import-time dependency issue
        raise RuntimeError("torchvision is required for image transforms") from exc

    image_rgb = Image.open(image_path).convert("RGB")
    tensor = tvf.to_tensor(image_rgb).to(device)

    with torch.no_grad():
        outputs = model([tensor])[0]

    scores = outputs["scores"].detach().cpu().numpy()
    labels = outputs["labels"].detach().cpu().numpy()
    boxes = outputs["boxes"].detach().cpu().numpy()

    results: list[tuple[int, float, list[int]]] = []
    for score, label, box in zip(scores, labels, boxes):
        if float(score) < conf_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        results.append((
            int(label),
            float(score),
            [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
        ))
    return results


@app.command("detect")
def detect(
    model_path: Path = typer.Option(..., help="Path to transformer .pth file"),
    num_classes: int = typer.Option(
        ..., help="Number of classes used when training the checkpoint"
    ),
    arch: str = typer.Option(
        "detr_resnet50",
        help="Model architecture: detr_resnet50 or detr_resnet50_dc5",
    ),
    image_dir: Optional[list[Path]] = typer.Option(
        None, help="One or more directories containing images to process"
    ),
    image_dir_file: Optional[Path] = typer.Option(
        None, help="Text file with directories to process, one per line"
    ),
    output_dir: Path = typer.Option(..., help="Directory to save crops"),
    class_names_file: Optional[Path] = typer.Option(
        None, help="Optional class names file, one class per line"
    ),
    image_glob: str = typer.Option("DJIP4*.JPG", help="Image glob pattern"),
    conf: float = typer.Option(0.5, help="Confidence threshold"),
    run_number: Optional[int] = typer.Option(
        None, help="Run number to continue from (overrides auto-increment)"
    ),
    require_cuda: bool = typer.Option(
        False, help="Fail if CUDA is not available"
    ),
) -> None:
    """Run transformer detection and save crops + metadata CSV."""
    if not model_path.exists():
        raise typer.BadParameter(f"Model path does not exist: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        print("ERROR: --require-cuda was set but CUDA is not available.")
        raise typer.Exit(1)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device: CUDA ({gpu_name})")
    else:
        print("Using device: CPU")

    model = _build_detr_model(arch=arch, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _normalize_state_dict(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )
    if missing_keys:
        print(f"Warning: Missing keys while loading checkpoint: {len(missing_keys)}")
    if unexpected_keys:
        print(
            "Warning: Unexpected keys while loading checkpoint: "
            f"{len(unexpected_keys)}"
        )

    model.to(device)
    model.eval()

    output_root = _resolve_output_dir(output_dir, model_path, run_number)
    print(f"Saving results to {output_root}")

    class_names = _load_class_names(class_names_file)
    dirs_to_process = _collect_input_dirs(image_dir, image_dir_file)
    if not dirs_to_process:
        print("No image directories provided.")
        raise typer.Exit(1)

    last_csv_path: Optional[Path] = None
    for img_dir in dirs_to_process:
        img_dir = Path(img_dir)
        image_files = sorted(img_dir.glob(image_glob))

        dir_output_dir = output_root / img_dir.name
        dir_output_dir.mkdir(exist_ok=True, parents=True)
        csv_path = dir_output_dir / f"{img_dir.name}_crops_metadata.csv"
        last_csv_path = csv_path

        dir_crop_records: list[dict[str, object]] = []
        processed_images: set[str] = set()

        if run_number is not None and csv_path.exists():
            try:
                existing = pd.read_csv(csv_path)
                processed_images = {
                    Path(path).stem for path in existing["original_image"]
                }
                dir_crop_records = existing.to_dict("records")
                print(
                    f"Skipping {len(processed_images)} images already "
                    f"in {csv_path}"
                )
            except Exception as exc:
                print(f"Warning: Could not read existing CSV {csv_path}: {exc}")

        for image_path in tqdm(image_files, desc=f"Processing {img_dir.name}"):
            if run_number is not None and image_path.stem in processed_images:
                continue

            try:
                detections = _predict_image(
                    model=model,
                    image_path=image_path,
                    device=device,
                    conf_threshold=conf,
                )

                if detections:
                    image_bgr = cv2.imread(str(image_path))
                    if image_bgr is None:
                        raise ValueError(f"Could not read image: {image_path}")

                    height, width = image_bgr.shape[:2]
                    for idx, (label_id, score, box) in enumerate(detections):
                        x1, y1, x2, y2 = box
                        x1 = max(0, min(width - 1, x1))
                        x2 = max(0, min(width, x2))
                        y1 = max(0, min(height - 1, y1))
                        y2 = max(0, min(height, y2))
                        if x2 <= x1 or y2 <= y1:
                            continue

                        crop = image_bgr[y1:y2, x1:x2]
                        class_name = class_names.get(label_id, f"class_{label_id}")
                        class_dir = dir_output_dir / class_name
                        class_dir.mkdir(exist_ok=True, parents=True)
                        conf_int = int(round(score * 100))
                        crop_file = (
                            class_dir
                            / (
                                f"{conf_int:03d}_{image_path.stem}_"
                                f"det_{idx}_{class_name}.jpg"
                            )
                        )
                        cv2.imwrite(str(crop_file), crop)

                        dir_crop_records.append(
                            {
                                "crop_path": str(crop_file.resolve()),
                                "original_image": str(image_path.resolve()),
                                "class_name": class_name,
                                "class_id": label_id,
                                "confidence": score,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "no_detections": False,
                            }
                        )
                else:
                    dir_crop_records.append(
                        {
                            "crop_path": "",
                            "original_image": str(image_path.resolve()),
                            "class_name": "",
                            "class_id": "",
                            "confidence": "",
                            "x1": "",
                            "y1": "",
                            "x2": "",
                            "y2": "",
                            "no_detections": True,
                        }
                    )
            except Exception as exc:
                print(f"Image problems {image_path}: {exc}")

            if dir_crop_records:
                frame = pd.DataFrame(dir_crop_records)
                frame.to_csv(csv_path, index=False)

    if last_csv_path is not None:
        print(f"Wrote crop metadata to {last_csv_path}")


if __name__ == "__main__":
    app()
