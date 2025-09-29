import cv2
import pandas as pd
import typer
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

app = typer.Typer()

def detect_turtle(frame, detection_model, category="turtle"):
    result = get_sliced_prediction(
        image=frame,
        detection_model=detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )   
    predictions = [
        obj for obj in result.object_prediction_list
        if category in obj.category.name.lower()
    ]
    return predictions

@app.command()
def find_turtles(
    video_path: Path = typer.Argument(..., help="Path to input drone video"),
    output_dir: Path = typer.Option("turtle_frames", help="Directory to save turtle frames"),
    model_path: str = typer.Option("yolov11", help="YOLO model name or path"),
    category: str = typer.Option("turtle", help="Target class label to search for"),
    sample_rate: int = typer.Option(15, help="Sample every N frames initially"),
    detection_window: int = typer.Option(60, help="Frames before/after to check if turtle detected"),
    device: str = typer.Option("cuda:0", help="Device to run YOLO on"),
    draw_box: bool = typer.Option(False, help="Draw bounding boxes on output frames"),
    csv_path: Path = typer.Option("detections.csv", help="Path to save detection CSV"),
    min_size: int = typer.Option(0, help="Minimum bounding box area in pixels to keep detection")
):
    """
    Scan a drone video and save all frames containing turtles using Ultralytics YOLOv11.
    Saves CSV with detections and elapsed time.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        typer.echo("❌ Error: Unable to open video.")
        raise typer.Exit(1)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    detection_model = AutoDetectionModel.from_pretrained(
        model_path=model_path,
        confidence_threshold=0.3,
        device=device,
        model_type="yolov11"
    )

    saved_frames = set()
    detections = []
    frame_idx = 0

    pbar = tqdm(total=frame_count, desc="🔍 Scanning video")

    while frame_idx < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        preds = detect_turtle(frame, detection_model, category)
        preds = list(filter(lambda p: p.bbox.area > min_size, preds))
        if preds:
            for offset in range(-detection_window, detection_window + 1, 5):
                nearby_idx = frame_idx + offset
                if nearby_idx < 0 or nearby_idx >= frame_count or nearby_idx in saved_frames:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, nearby_idx)
                ret2, sub_frame = cap.read()
                if not ret2:
                    continue

                nearby_preds = detect_turtle(sub_frame, detection_model, category)
                filtered_preds =  list(filter(lambda p: p.bbox.area > min_size, nearby_preds))
                if filtered_preds:
                    frame_copy = sub_frame.copy()
                    if draw_box:
                        for pred in filtered_preds:
                            x1, y1, x2, y2 = map(int, pred.bbox.to_voc_bbox())
                            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                frame_copy, f'{pred.category.name} {round(pred.score.value, 3)}', (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                            )

                    out_path = output_dir / f"frame_{nearby_idx:06d}.jpg"
                    cv2.imwrite(str(out_path), frame_copy)
                    saved_frames.add(nearby_idx)

                    for pred in filtered_preds:
                        x1, y1, x2, y2 = map(int, pred.bbox.to_voc_bbox())
                        time_sec = nearby_idx / fps
                        time_str = str(timedelta(seconds=int(time_sec)))
                        detections.append({
                            "frame": nearby_idx,
                            "time_sec": round(time_sec, 2),
                            "time_str": time_str,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": round(pred.score.value, 3),
                            "label": pred.category.name
                        })

        frame_idx += sample_rate
        pbar.update(sample_rate)

    cap.release()
    pbar.close()

    if detections:
        df = pd.DataFrame(detections)
        df = df.sort_values("frame")
        df.to_csv(csv_path, index=False)
        typer.echo(f"📄 Saved {len(df)} detections to: {csv_path}")

    typer.echo(f"✅ Done. Saved {len(saved_frames)} turtle frames to {output_dir}")

if __name__ == "__main__":
    app()
