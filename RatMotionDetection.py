""" RatMotionDetection.py -- motion detection for rat detection pipeline

Scans hours of video, uses YOLO (COCO class 14 = bird) on sampled frames to find when something owl-like is in view, merges those moments into events, pads them, and cuts each event into a short clip. The clips go to the stage-2 classifier (the ResNet18 + LSTM notebook) which decides "owl carrying a rodent" vs not.

Outputs
    candidates/*.mp4    short clips to feed the classifier
    manifest.csv        clip, source video, start/end seconds, hit count, max confidence
    processed.txt       source videos already scanned (re-running skips them)

Requires: ultralytics, opencv-python, torch, tqdm, and the ffmpeg binary on PATH

To Run: py RatMotionDetection.py --input_dir "./footage" --output_dir ./owl_candidates """

import argparse
import csv
import hashlib
import multiprocessing as mp
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
BIRD_CLASS = 14, 33

# Per-worker globals, set once in init_worker so the model loads once per process 
_model = None
_device = None
_args = None

def init_worker(n_gpus, args):
    global _model, _device, _args
    _args = args
    worker_idx = mp.current_process()._identity[0] - 1
    _device = f"cuda:{worker_idx % n_gpus}" if n_gpus > 0 else "cpu"
    _model = YOLO(args.weights)
    _model.to(_device)

def scan_video(path):
    """Sequential decode, run YOLO on every Nth frame. Returns (hits, duration).
    hits = [(time_sec, max_bird_conf), ...] for frames containing a bird."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, round(fps * _args.sample_every))
    use_half = _device != "cpu"

    hits, batch, times = [], [], []

    def flush():
        if not batch:
            return
        results = _model.predict(
            batch, device=_device, classes=[BIRD_CLASS], conf=_args.conf,
            imgsz=_args.imgsz, half=use_half, verbose=False,
        )
        for t, r in zip(times, results):
            if len(r.boxes) > 0:
                hits.append((t, float(r.boxes.conf.max())))
        batch.clear()
        times.clear()

    idx = 0
    # grab() skips decoding the pixel conversion; retrieve() onl on sampled frames
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                batch.append(frame)
                times.append(idx / fps)
                if len(batch) >= _args.batch:
                    flush()

        idx += 1
    flush()
    cap.release()
    return hits, idx / fps


def build_windows(hits, duration, a):
    """Merge nearby hits into events, pad them, and split long events into clips."""
    if len(hits) < a.min_hits:
        return []

    groups, cur = [], [hits[0]]
    for h in hits[1:]:
        if h[0] - cur[-1][0] <= a.merge_gap:
            cur.append(h)
        else:
            groups.append(cur)
            cur = [h]
    groups.append(cur)

    windows = []
    for g in groups:
        if len(g) < a.min_hits:
            continue
        start = max(0.0, g[0][0] - a.pre)
        end = min(duration, g[-1][0] + a.post)
        conf = max(c for _, c in g)
        t = start
        while t < end:
            seg_end = min(t + a.max_clip, end)
            if seg_end - t >= 1.0:
                windows.append((t, seg_end, len(g), conf))
            t = seg_end
    return windows


def cut_clip(src, dst, start, end):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{end - start:.2f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(dst),
    ]
    subprocess.run(cmd, check=True)


def process_video(path_str):
    src = Path(path_str)
    try:
        hits, duration = scan_video(src)
        windows = build_windows(hits, duration, _args)
        tag = hashlib.md5(path_str.encode()).hexdigest()[:6]
        rows = []
        for start, end, n_hits, conf in windows:
            dst = _args.output_dir / "candidates" / f"{src.stem}_{tag}_{int(start):06d}s.mp4"

            cut_clip(src, dst, start, end)
            rows.append({
                "clip": dst.name, "source": path_str,
                "start_sec": round(start, 2), "end_sec": round(end, 2),
                "bird_hits": n_hits, "max_conf": round(conf, 3),
            })
        return path_str, duration, rows, None
    except Exception as e:
        return path_str, 0.0, [], str(e)

def main():
    p = argparse.ArgumentParser(description="Find bird activity in long footage and cut candidate clips")
    p.add_argument("--input_dir", "-i", required=True, type=Path)
    p.add_argument("--output_dir", "-o", required=True, type=Path)
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO bird confidence; keep low for high recall")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--sample-every", type=float, default=0.5, help="seconds between YOLO checks")
    p.add_argument("--batch", type=int, default=16, help="frames per YOLO call")
    p.add_argument("--min-hits", type=int, default=2, help="bird frames needed to count as an event")
    p.add_argument("--merge-gap", type=float, default=5.0, help="seconds of no bird before an event ends")
    p.add_argument("--pre", type=float, default=3.0, help="seconds kept before the first detection")
    p.add_argument("--post", type=float, default=4.0, help="seconds kept after the last detection")
    p.add_argument("--max-clip", type=float, default=12.0, help="longest clip; longer events are split")
    p.add_argument("--procs-per-gpu", type=int, default=2)
    args = p.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH (e.g. sudo apt install ffmpeg)")
    if not args.input_dir.is_dir():
        sys.exit(f"Input directory not found: {args.input_dir}")

    (args.output_dir / "candidates").mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    processed_path = args.output_dir / "processed.txt"

    done = set(processed_path.read_text().splitlines()) if processed_path.exists() else set()
    videos = [str(v) for v in sorted(args.input_dir.rglob("*"))
                if v.is_file() and v.suffix.lower() in VIDEO_EXTS and str(v) not in done]
    print(f"{len(videos)} videos to scan ({len(done)} already done)")
    if not videos:
        return

    n_gpus = torch.cuda.device_count()
    n_workers = max(1, n_gpus * args.procs_per_gpu) if n_gpus else 2
    print(f"GPUs: {n_gpus}, workers: {n_workers}")

    fields = ["clip", "source", "start_sec", "end_sec", "bird_hits", "max_conf"]
    new_manifest = not manifest_path.exists()
    total_sec = kept_sec = 0.0
    n_clips = 0
    
    # Only the main process writes the logs, so no file locking is needed (works on Windows too)
    with open(manifest_path, "a", newline="") as mf, open(processed_path, "a") as pf, \
        ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"),
            initializer=init_worker, initargs=(n_gpus, args)) as ex:

        writer = csv.DictWriter(mf, fieldnames=fields)
        if new_manifest:
            writer.writeheader()

        futures = [ex.submit(process_video, v) for v in videos]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="video"):
            path_str, duration, rows, err = fut.result()
            if err:
                tqdm.write(f"FAILED {path_str}: {err}")
                continue # not marked done, so a re-run retries it
            writer.writerows(rows)
            pf.write(path_str + "\n")
            mf.flush()
            pf.flush()
            total_sec += duration
            n_clips += len(rows)
            kept_sec += sum(r["end_sec"] - r["start_sec"] for r in rows)

    print(f"\nScanned {total_sec / 3600:.2f} h of video")
    print(f"Kept {n_clips} clips ({kept_sec / 60:.1f} min) -> {args.output_dir / 'candidates'}")


if __name__ == "__main__":
    main()