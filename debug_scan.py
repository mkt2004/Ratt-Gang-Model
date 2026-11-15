#!/usr/bin/env python3
"""
debug_scan.py -- see what YOLO detects in ONE video, with no class filter.

Use this when the main prefilter misses an owl you can see by eye. It prints every
detection (class + confidence) per sampled frame and saves annotated frames so you
can look at what the model is seeing.

Usage:
    py debug_scan.py "footage/Ber2_Night1.mp4" --start 10 --end 40
    py debug_scan.py "footage/Ber2_Night1.mp4" --sample-every 0.2 --conf 0.05
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--sample-every", type=float, default=0.25, help="seconds between frames")
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--end", type=float, default=None, help="end time in seconds")
    p.add_argument("--save-dir", type=Path, default=Path("debug_frames"))
    p.add_argument("--max-saved", type=int, default=60)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = YOLO(args.weights)
    model.to(device)

    cap = cv2.VideoCapture(str(args.video), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, round(fps * args.sample_every))
    print(f"fps={fps:.1f}, checking every {step} frames ({step / fps:.2f}s)")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    class_counts = Counter()
    frames_checked = frames_with_any = frames_with_bird = saved = 0

    idx = 0
    while cap.grab():
        t = idx / fps
        if args.end is not None and t > args.end:
            break
        if idx % step == 0 and t >= args.start:
            ok, frame = cap.retrieve()
            if ok:
                frames_checked += 1
                r = model.predict(frame, device=device, conf=args.conf,
                                  imgsz=args.imgsz, verbose=False)[0]
                if len(r.boxes) > 0:
                    frames_with_any += 1
                    dets = []
                    for cls, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                        name = r.names[int(cls)]
                        class_counts[name] += 1
                        dets.append(f"{name} {conf:.2f}")
                        if name == "bird":
                            frames_with_bird += 1
                    print(f"t={t:7.2f}s  " + ", ".join(dets))
                    if saved < args.max_saved:
                        cv2.imwrite(str(args.save_dir / f"t{t:08.2f}.jpg"), r.plot())
                        saved += 1
        idx += 1
    cap.release()

    print("\n--- Summary ---")
    print(f"Frames checked: {frames_checked}")
    print(f"Frames with any detection: {frames_with_any}")
    print(f"Frames with a 'bird' detection: {frames_with_bird}")
    print("Classes seen:", dict(class_counts.most_common()) or "none")
    print(f"Annotated frames saved to: {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()