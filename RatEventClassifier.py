#!/usr/bin/env python3
"""Classify RatMotionDetection candidate clips and count rat events once.

Expected input from RatMotionDetection.py:
    <motion-output-dir>/manifest.csv
    <motion-output-dir>/candidates/<clip filename>

The manifest's source/start_sec/end_sec fields group clips from the same source
video into events. Consecutive clips join one event when the gap from the prior
clip's end to the next clip's start is at most --gap-seconds.

The loaded weights must be the raw state_dict saved by RatDetectionModel_v01_01.
Class 1 is treated as containing a rat; class 0 is no_rat.
"""

import argparse
import csv
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def read_manifest(manifest_path: Path, clips_dir: Path):
    required = {"clip", "source", "start_sec", "end_sec"}
    rows = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        for line_number, raw in enumerate(reader, start=2):
            row = dict(raw)
            try:
                row["start_sec"] = float(row["start_sec"])
                row["end_sec"] = float(row["end_sec"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid start/end timestamp on manifest line {line_number}") from exc
            if row["end_sec"] < row["start_sec"]:
                raise ValueError(f"end_sec precedes start_sec on manifest line {line_number}")

            clip_ref = Path(row["clip"])
            row["clip_path"] = clip_ref if clip_ref.is_absolute() else clips_dir / clip_ref
            row["rat_probability"] = ""
            row["clip_prediction"] = "unclassified"
            row["prediction_error"] = ""
            rows.append(row)
    return rows


def group_events(clip_rows, gap_seconds=20.0):
    """Group clips from one source when the gap after the prior clip is <= limit."""
    ordered = sorted(
        clip_rows,
        key=lambda r: (str(r["source"]), float(r["start_sec"]), float(r["end_sec"])),
    )
    events = []
    current = None

    for row in ordered:
        source = str(row["source"])
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        joins_current = (
            current is not None
            and source == current["source"]
            and start - current["end_sec"] <= gap_seconds
        )

        if not joins_current:
            current = {
                "source": source,
                "start_sec": start,
                "end_sec": end,
                "clips": [row],
            }
            events.append(current)
        else:
            current["end_sec"] = max(current["end_sec"], end)
            current["clips"].append(row)

    return events


def score_event(event, threshold=0.5):
    """Duration-weight clip probabilities so each event contributes one decision."""
    scored = []
    for row in event["clips"]:
        value = row.get("rat_probability", "")
        if value in (None, ""):
            continue
        duration = max(0.0, float(row["end_sec"]) - float(row["start_sec"]))
        scored.append((float(value), max(duration, 1e-6)))

    if not scored:
        probability = None
        prediction = "unclassified"
    else:
        total_weight = sum(weight for _, weight in scored)
        probability = sum(prob * weight for prob, weight in scored) / total_weight
        prediction = "containing_rat" if probability >= threshold else "no_rat"

    return {
        "source": event["source"],
        "start_sec": round(event["start_sec"], 2),
        "end_sec": round(event["end_sec"], 2),
        "clip_count": len(event["clips"]),
        "scored_clip_count": len(scored),
        "rat_clip_votes": sum(
            row.get("clip_prediction") == "containing_rat" for row in event["clips"]
        ),
        "event_rat_probability": "" if probability is None else round(probability, 6),
        "prediction": prediction,
    }


def load_v01_model(weights_path: Path, device):
    """Rebuild the v01 architecture and load its state_dict."""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    class BinaryVideoClassifier(nn.Module):
        def __init__(self, hidden_dim=256, dropout=0.3):
            super().__init__()
            backbone = resnet18(weights=None)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.frame_encoder = backbone
            self.temporal_model = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, videos):
            batch_size, num_frames, channels, height, width = videos.shape
            frames = videos.reshape(batch_size * num_frames, channels, height, width)
            features = self.frame_encoder(frames)
            features = features.reshape(batch_size, num_frames, -1)
            sequence, _ = self.temporal_model(features)
            return self.classifier(sequence.mean(dim=1)).squeeze(1)

    model = BinaryVideoClassifier().to(device)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        state = torch.load(weights_path, map_location=device)

    # Accept a raw state_dict, and common wrapped state_dict checkpoints.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def video_to_tensor(video_path: Path, num_frames=16, image_size=224):
    """Match v01 VideoDataset sampling, RGB conversion, resize, and normalization."""
    import cv2
    import numpy as np
    import torch

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open clip: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"Clip has no readable frames: {video_path}")

        indices = np.linspace(0, frame_count - 1, num_frames).round().astype(int)
        frames = []
        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
            frames.append(tensor)
    finally:
        cap.release()

    video = torch.stack(frames)  # [T, C, H, W]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (video - mean) / std


def classify_clips(rows, model, device, threshold=0.5, batch_size=8):
    """Fill per-clip probability/label fields; keep decode failures in output."""
    import torch

    pending_rows, pending_videos = [], []

    def flush():
        if not pending_rows:
            return
        batch = torch.stack(pending_videos).to(device)
        with torch.no_grad():
            probabilities = torch.sigmoid(model(batch).reshape(-1)).cpu().tolist()
        for row, probability in zip(pending_rows, probabilities):
            row["rat_probability"] = round(float(probability), 6)
            row["clip_prediction"] = (
                "containing_rat" if probability >= threshold else "no_rat"
            )
        pending_rows.clear()
        pending_videos.clear()

    for row in rows:
        try:
            video = video_to_tensor(Path(row["clip_path"]))
            pending_rows.append(row)
            pending_videos.append(video)
        except Exception as exc:
            row["prediction_error"] = str(exc)
            row["clip_prediction"] = "unclassified"
        if len(pending_rows) >= batch_size:
            flush()
    flush()
    return rows


def classify_events(clip_rows, gap_seconds=20.0, threshold=0.5):
    events = group_events(clip_rows, gap_seconds=gap_seconds)
    event_rows = []
    rat_event_count = 0

    for number, event in enumerate(events, start=1):
        result = score_event(event, threshold=threshold)
        result["event_id"] = f"event_{number:05d}"
        if result["prediction"] == "containing_rat":
            rat_event_count += 1
            result["rat_event_count"] = rat_event_count
        else:
            result["rat_event_count"] = ""
        event_rows.append(result)

    return event_rows, rat_event_count


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Classify motion-detection clips and count rat events once per time group."
    )
    parser.add_argument("--motion-output-dir", required=True, type=Path,
                        help="RatMotionDetection output folder containing manifest.csv and candidates/")
    parser.add_argument("--weights", required=True, type=Path,
                        help="binary_video_classifier.pt saved from RatDetectionModel_v01_01")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to <motion-output-dir>/rat_classification")
    parser.add_argument("--gap-seconds", type=float, default=20.0,
                        help="Maximum gap between clips from the same source to group as one event")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Clip/event rat-probability threshold")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.gap_seconds < 0:
        parser.error("--gap-seconds cannot be negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    manifest_path = args.motion_output_dir / "manifest.csv"
    candidates_dir = args.motion_output_dir / "candidates"
    if not manifest_path.is_file():
        parser.error(f"Motion manifest not found: {manifest_path}")
    if not candidates_dir.is_dir():
        parser.error(f"Candidate clips folder not found: {candidates_dir}")
    if not args.weights.is_file():
        parser.error(f"Weights file not found: {args.weights}")

    rows = read_manifest(manifest_path, candidates_dir)
    if not rows:
        parser.error(f"No clips listed in {manifest_path}")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading v01 model on {device} ...")
    model = load_v01_model(args.weights, device)
    print(f"Classifying {len(rows)} candidate clips ...")
    rows = classify_clips(rows, model, device, threshold=args.threshold,
                          batch_size=args.batch_size)
    event_rows, rat_event_count = classify_events(
        rows, gap_seconds=args.gap_seconds, threshold=args.threshold
    )

    output_dir = args.output_dir or (args.motion_output_dir / "rat_classification")
    clip_fields = ["clip", "source", "start_sec", "end_sec", "bird_hits", "max_conf",
                   "rat_probability", "clip_prediction", "prediction_error"]
    event_fields = ["event_id", "source", "start_sec", "end_sec", "clip_count",
                    "scored_clip_count", "rat_clip_votes", "event_rat_probability",
                    "prediction", "rat_event_count"]
    clip_csv = output_dir / "clip_predictions.csv"
    event_csv = output_dir / "rat_events.csv"
    write_csv(clip_csv, rows, clip_fields)
    write_csv(event_csv, event_rows, event_fields)

    failures = sum(bool(r["prediction_error"]) for r in rows)
    print(f"Grouped {len(rows)} clips into {len(event_rows)} source-time events.")
    print(f"Rat-containing event count: {rat_event_count}")
    print(f"Clip predictions: {clip_csv}")
    print(f"Event summary: {event_csv}")
    if failures:
        print(f"Warning: {failures} clips could not be decoded; see prediction_error in clip_predictions.csv.")


if __name__ == "__main__":
    main()
