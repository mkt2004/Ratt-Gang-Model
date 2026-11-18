"""
MotionPrefilter.py -- lightweight motion prefilter. No machine learning, only OpenCV.

Takes sample frames, shrinks them, diffs against the previous sample, and flags moments where there is enough pixels changed.
Nearby hits get merged into evets, are padded, and are cut into clips for the next stage of processing.

Usage:
    python MotionPrefilter.py -i ./footage -o ./owl_candidates
    python MotionPrefilter.py --probe "train/0/Dev15_Night4_V1-0002.mov"
"""

import argparse, csv, hashlib, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

def scan_video(path, a):
    """Returns (hits, duration). hits = [(time_sec, changed_fraction), ...]"""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, round(fps * a["sample_every"]))
    kernel = np.ones((3, 3), np.uint8)

    hits, scores, prev, idx = [], [], None, 0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (a["width"], int(h * a["width"] / w)),
                                   interpolation=cv2.INTER_AREA)
                gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                if prev is not None:
                    mask = cv2.absdiff(gray, prev) > a["pixel_thresh"]
                    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                    frac = float(mask.mean())
                    scores.append(frac)
                    # too much change would be a lighting flash / IR switch / exposure shift
                    if a["min_frac"] <= frac <= a["max_frac"]:
                        hits.append((idx / fps, frac))
                prev = gray
        idx += 1
    cap.release()
    return hits, idx / fps, scores

def build_windows(hits, duration, a):
    if len(hits) < a["min_hits"]:
        return []
    groups, cur = [], [hits[0]]
    for h in hits[1:]:
        if h[0] - cur[-1][0] <= a["merge_gap"]:
            cur.append(h)
        else:
            groups.append(cur); cur = [h]
    groups.append(cur)

    windows = []
    for g in groups:
        if len(g) < a["min_hits"]:
            continue
        start = max(0.0, g[0][0] - a["pre"])
        end = min(duration, g[-1][0] + a["post"])
        score = max(s for _, s in g)
        t = start
        while t < end:
            seg_end = min(t + a["max_clip"], end)
            if seg_end - t >= 1.0:
                windows.append((t, seg_end, len(g), score))
            t = seg_end
    return windows

def cut_clip(src, dst, start, end):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}",
                    "-i", str(src), "-t" f"{end - start:.2f}", "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(dst)],
                    check=True)

def process_video(path_str, a):
    src = Path(path_str)
    try:
        hits, duration, _ = scan_video(src, a)
        tag = hashlib.md5(path_str.encode()).hexdigest()[:6]
        rows = []
        for start, end, n, score in build_windows(hits, duration, a):
            dst = Path(a["output_dir"]) / "candidates" / f"{src.stem}_{tag}_{int(start):06d}s.mp4"
            cut_clip(src, dst, start, end)
            rows.append({"clip": dst.name, "source": path_str,
                         "start_sec": round(start, 2), "end_sec": round(end, 2),
                         "motion_hits": n, "max_motion": round(score, 4)})
        return path_str, duration, rows, None
    except Exception as e:
        return path_str, 0.0, [], str(e)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", "-i", type=Path)
    p.add_argument("--output_dir", "-o", type=Path)
    p.add_argument("--probe", type=Path, help="scan ONE video, print score stats, cut nothing")
    p.add_argument("--sample_every", type=float, default=0.25, help="seconds between frames compared")
    p.add_argument("--width", type=int, default=320, help="downscale width for analysis")
    p.add_argument("--pixel-thresh", type=int, default=25, help="gray-level change that counts as motion")
    p.add_argument