# Vehicle Speed Estimation (YOLO11 + OpenCV)

Real-time vehicle speed estimation from traffic video. Detects cars, bikes, buses and trucks, tracks them with YOLO11's built-in tracker, times each vehicle between two virtual lines, and shows its speed in km/h.

## Features
- YOLO11 detection with built-in tracking (`persist=True`)
- Click-to-calibrate speed lines
- Live dashboard: counts per class, direction, average/max speed, FPS
- Speed-limit alerts with snapshots of violators
- CSV log of every vehicle
- Traffic heatmap and speed distribution report

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
python vehicle_speed_pro.py --source traffic.mp4 --distance 10 --limit 60 --calibrate
```

| Option | Description |
|---|---|
| `--source` | Video path, or `0` for webcam |
| `--distance` | Real distance in metres between the two lines |
| `--limit` | Speed limit in km/h |
| `--calibrate` | Click the two lines on the first frame |
| `--line-a`, `--line-b` | Set line positions (y pixels) manually |
| `--imgsz` | YOLO input size (smaller = faster, default 480) |
| `--show-every` | Refresh the live window every N frames |
| `--trails` | Draw motion trails |
| `--no-show` | Run without a window |

## Outputs
- `speed_output.mp4`: annotated video
- `speed_log.csv`: speed of every vehicle
- `heatmap.png`: where vehicles were tracked most
- `speed_report.png`: speed distribution
- `violations/`: snapshots of speeding vehicles

## Accuracy note
Speed = distance / time between the two lines. Results are only as accurate as `--distance`, so measure the real gap on the road. Use a fixed camera. Speeds are estimates.

## Roadmap
- Perspective (homography) correction for angled cameras
- Streamlit/Gradio demo
- Number-plate recognition for violators

## Stack
Python, Ultralytics YOLO11, OpenCV, NumPy, Matplotlib
