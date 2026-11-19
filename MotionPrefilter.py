"""
MotionPrefilter.py -- lightweight motion prefilter. No machine learning, only OpenCV.

Takes sample frames, shrinks them, diffs against the previous sample, and flags moments where there is enough pixels changed.
Nearby hits get merged into evets, are padded, and are cut into clips for the next stage of processing.

Usage:
    python MotionPrefilter.py -i ./footage -o ./owl_candidates
    python MotionPrefilter.py --probe "footage/Ber2_Night2.mp4"
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
                    "-i", str(src), "-t", f"{end - start:.2f}", "-an",
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
    p.add_argument("--min-frac", type=float, default=0.002, help="fraction of pixels that must change")
    p.add_argument("--max-frac", type=float, default=0.6, help="above this = lighting change, ignored")
    p.add_argument("--min-hits", type=int, default=2)
    p.add_argument("--merge-gap", type=float, default=5.0)
    p.add_argument("--pre", type=float, default=3.0)
    p.add_argument("--post", type=float, default=4.0)
    p.add_argument("--max-clip", type=float, default=12.0)
    p.add_argument("--workers", type=int, default=0, help="0 = CPU count minus 1")
    args = p.parse_args()
    a = vars(args)

    if args.probe:
        hits, dur, scores = scan_video(args.probe, a)
        s = np.array(scores)
        print(f"duration {dur:.0f}s, {len(s)} comparisons, {len(hits)} hits "
              f"({100 * len(hits) / max(1, len(s)):.1f}% of samples)")
        print("score percentiles 50/90/95/99/max:",
              [round(float(np.percentile(s, q)), 4) for q in (50, 90, 95, 99, 100)])
        print("events:", len(build_windows(hits, dur, a)))
        return

    if not (args.input_dir and args.output_dir):
        sys.exit("need --input_dir and --output_dir (or --probe)")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH")

    (args.output_dir / "candidates").mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    processed_path = args.output_dir / "processed.txt"
    done = set(processed_path.read_text().splitlines()) if processed_path.exists() else set()
    videos = [str(v) for v in sorted(args.input_dir.rglob("*"))
              if v.is_file() and v.suffix.lower() in VIDEO_EXTS and str(v) not in done]
    print(f"{len(videos)} videos to scan ({len(done)} already done)")
    if not videos:
        return

    import os
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    fields = ["clip", "source", "start_sec", "end_sec", "motion_hits", "max_motion"]
    new = not manifest_path.exists()
    total = kept = 0.0
    n_clips = 0

    with open(manifest_path, "a", newline="") as mf, open(processed_path, "a") as pf, \
        ProcessPoolExecutor(max_workers=workers) as ex:
        w = csv.DictWriter(mf, fieldnames=fields)
        if new:
            w.writeheader()
        futs = [ex.submit(process_video, v, a) for v in videos]
        for fut in tqdm(as_completed(futs), total=len(futs), unit="video"):
            path_str, dur, rows, err = fut.result()
            if err:
                tqdm.write(f"FAILED {path_str}: {err}")
                continue
            w.writerows(rows)
            pf.write(path_str + "\n")
            mf.flush(); pf.flush()
            total += dur
            n_clips += len(rows)
            kept += sum(r["end_sec"] - r["start_sec"] for r in rows)

    print(f"\nScanned {total / 3600:.2f} h, kept {n_clips} clips ({kept / 60:.1f} min)"
          f" = {100 * kept / max(1, total):.1f}% of footage")

if __name__ == "__main__":
    main()