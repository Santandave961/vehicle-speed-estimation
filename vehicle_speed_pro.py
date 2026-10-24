"""
Vehicle Speed Estimation PRO: YOLO11 + built-in tracker

Features
- Click-to-calibrate speed lines (no hardcoded pixels)
- Live dashboard: counts per class, direction (up/down), avg/max speed, FPS
- Motion trails behind every tracked vehicle
- Speed-limit alerts: red box + "SPEEDING" + snapshot saved per violator
- CSV log of every vehicle (id, class, direction, speed, time, speeding)
- Traffic heatmap image saved at the end
- Speed distribution report (PNG) saved at the end

pip install ultralytics opencv-python numpy matplotlib

Run:
  python vehicle_speed_pro.py --source traffic.mp4 --distance 10 --limit 60 --calibrate
"""
import argparse
import csv
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = {2: "car", 3: "bike", 5: "bus", 7: "truck"}
COLORS = {"car": (0, 200, 0), "bike": (255, 160, 0), "bus": (0, 180, 255), "truck": (200, 0, 200)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="traffic.mp4", help="video path or 0 for webcam")
    p.add_argument("--out", default="speed_output.mp4")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--distance", type=float, default=10.0, help="metres between the two lines")
    p.add_argument("--limit", type=float, default=60.0, help="speed limit in km/h")
    p.add_argument("--line-a", type=int, default=None, help="y pixel of first line")
    p.add_argument("--line-b", type=int, default=None, help="y pixel of second line")
    p.add_argument("--calibrate", action="store_true", help="click two lines on the first frame")
    p.add_argument("--no-show", action="store_true", help="run headless (no window)")
    p.add_argument("--trails", action="store_true", help="draw motion trails (off by default)")
    p.add_argument("--imgsz", type=int, default=480, help="YOLO input size; smaller = faster")
    p.add_argument("--show-every", type=int, default=2, help="refresh the live window every N frames")
    return p.parse_args()


def calibrate_lines(frame):
    """Click twice on the first frame: top line, then bottom line."""
    ys = []
    win = "Calibrate: click LINE A then LINE B (ESC to skip)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(ys) < 2:
            ys.append(y)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while len(ys) < 2:
        view = frame.copy()
        for y in ys:
            cv2.line(view, (0, y), (view.shape[1], y), (0, 255, 255), 2)
        cv2.putText(view, f"Click line {'A' if not ys else 'B'}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(win, view)
        if cv2.waitKey(30) == 27:
            break
    cv2.destroyWindow(win)
    return sorted(ys) if len(ys) == 2 else None


def draw_panel(frame, stats, fps_now, limit):
    """Semi-transparent dashboard in the top-left corner."""
    lines = [
        f"FPS: {fps_now:.0f}   Limit: {limit:.0f} km/h",
        f"Total: {stats['total']}   Down: {stats['down']}   Up: {stats['up']}",
        "  ".join(f"{k}: {v}" for k, v in stats["by_class"].items()) or "no vehicles yet",
        f"Avg: {stats['avg']:.0f} km/h   Max: {stats['max']:.0f} km/h",
        f"Speeding: {stats['speeding']}",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (470, 20 + 28 * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, t in enumerate(lines):
        color = (0, 0, 255) if (i == 4 and stats["speeding"]) else (255, 255, 255)
        cv2.putText(frame, t, (20, 38 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def save_report(speeds, limit, path="speed_report.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping report")
        return
    if not speeds:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(speeds, bins=15, color="#2a9d8f", edgecolor="white")
    ax.axvline(limit, color="red", linestyle="--", label=f"Limit {limit:.0f} km/h")
    ax.axvline(np.mean(speeds), color="black", linestyle=":", label=f"Mean {np.mean(speeds):.0f} km/h")
    ax.set_title("Vehicle speed distribution")
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Vehicles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def save_heatmap(heat, background, path="heatmap.png"):
    heat = cv2.GaussianBlur(heat, (0, 0), 25)
    if heat.max() > 0:
        heat = heat / heat.max()
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(background, 0.5, colored, 0.5, 0)
    cv2.imwrite(path, blended)
    print(f"Saved {path}")


def main():
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    ok, first = cap.read()
    if not ok:
        raise SystemExit(f"Cannot read video: {args.source}")
    h, w = first.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # ---- lines ----
    line_a, line_b = int(h * 0.55), int(h * 0.75)
    if args.line_a is not None and args.line_b is not None:
        line_a, line_b = sorted((args.line_a, args.line_b))
    if args.calibrate and not args.no_show:
        picked = calibrate_lines(first)
        if picked:
            line_a, line_b = picked
    print(f"Lines: A={line_a}px  B={line_b}px  distance={args.distance} m")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    os.makedirs("violations", exist_ok=True)

    model = YOLO(args.model)

    prev_y = {}
    crossings = {}                       # id -> {"A": frame, "B": frame, "first": "A"/"B"}
    speeds = {}                          # id -> km/h
    classes_of = {}                      # id -> class name
    trails = defaultdict(lambda: deque(maxlen=30))
    snapped = set()
    heat = np.zeros((h, w), np.float32)

    stats = {"total": 0, "up": 0, "down": 0, "by_class": defaultdict(int),
             "avg": 0.0, "max": 0.0, "speeding": 0}

    log_file = open("speed_log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["id", "class", "direction", "speed_kmh", "video_time_s", "speeding"])

    frame_no, t_prev, fps_now = 0, time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1

        result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                             classes=list(CLASSES), conf=args.conf, imgsz=args.imgsz,
                             verbose=False)[0]

        clean = frame.copy()             # unannotated copy for violation snapshots
        cv2.line(frame, (0, line_a), (w, line_a), (0, 255, 255), 2)
        cv2.line(frame, (0, line_b), (w, line_b), (0, 100, 255), 2)

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            cls_ids = result.boxes.cls.int().cpu().tolist()

            for (x1, y1, x2, y2), tid, cid in zip(boxes, ids, cls_ids):
                name = CLASSES[cid]
                classes_of[tid] = name
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                trails[tid].append((cx, cy))
                if 0 <= cy < h and 0 <= cx < w:
                    heat[cy, cx] += 1

                # ---- line crossing ----
                if tid in prev_y and tid not in speeds:
                    py = prev_y[tid]
                    rec = crossings.setdefault(tid, {})
                    for tag, ly in (("A", line_a), ("B", line_b)):
                        if tag not in rec and py != cy and (py - ly) * (cy - ly) <= 0:
                            rec[tag] = frame_no
                            rec.setdefault("first", tag)
                    if "A" in rec and "B" in rec:
                        dt = abs(rec["B"] - rec["A"]) / fps
                        if dt > 0:
                            kmh = args.distance / dt * 3.6
                            speeds[tid] = kmh
                            direction = "down" if rec["first"] == "A" else "up"
                            speeding = kmh > args.limit
                            stats["total"] += 1
                            stats[direction] += 1
                            stats["by_class"][name] += 1
                            stats["speeding"] += int(speeding)
                            log.writerow([tid, name, direction, f"{kmh:.1f}",
                                          f"{frame_no / fps:.2f}", speeding])
                prev_y[tid] = cy

                # ---- drawing ----
                kmh = speeds.get(tid)
                speeding = kmh is not None and kmh > args.limit
                color = (0, 0, 255) if speeding else COLORS[name]

                if args.trails:
                    pts = np.array(trails[tid], np.int32).reshape(-1, 1, 2)
                    if len(pts) > 1:
                        cv2.polylines(frame, [pts], False, color, 2)

                x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 3 if speeding else 2)
                label = f"{name} #{tid}" + (f"  {kmh:.0f} km/h" if kmh else "")
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1i, y1i - th - 10), (x1i + tw + 6, y1i), color, -1)
                cv2.putText(frame, label, (x1i + 3, y1i - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                if speeding:
                    cv2.putText(frame, "SPEEDING", (x1i, y2i + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if tid not in snapped:
                        snapped.add(tid)
                        crop = clean[max(y1i, 0):max(y2i, 0), max(x1i, 0):max(x2i, 0)]
                        if crop.size:
                            cv2.imwrite(f"violations/id{tid}_{name}_{kmh:.0f}kmh.jpg", crop)

        if speeds:
            vals = list(speeds.values())
            stats["avg"], stats["max"] = float(np.mean(vals)), float(np.max(vals))

        now = time.time()
        fps_now = 0.9 * fps_now + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now

        draw_panel(frame, stats, fps_now, args.limit)
        writer.write(frame)
        if not args.no_show and frame_no % args.show_every == 0:
            cv2.imshow("Vehicle Speed PRO (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    log_file.close()
    cv2.destroyAllWindows()

    save_heatmap(heat, first)
    save_report(list(speeds.values()), args.limit)
    print(f"Saved {args.out}, speed_log.csv, violations/")


if __name__ == "__main__":
    main()