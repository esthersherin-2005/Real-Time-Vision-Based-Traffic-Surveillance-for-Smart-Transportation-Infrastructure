# visualization.py
import cv2
import numpy as np


# ---------------------------------------------------------
# Draw bounding box (YOLO format: x1, y1, x2, y2)
# ---------------------------------------------------------
def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return frame


# ---------------------------------------------------------
# Draw text on frame
# ---------------------------------------------------------
def draw_text(frame, text, position, color=(0, 255, 255), scale=0.6, thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return frame


# ---------------------------------------------------------
# YOLO + Tracker Visualization
# ---------------------------------------------------------
def draw_detections(frame, detections, tracked_objects):

    # YOLO detections (GREEN)
    if detections:
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cname = d.get("class_name", "obj")
            conf = d.get("conf", d.get("confidence", 0.0))

            # box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)

            # label
            cv2.putText(frame,
                        f"{cname} {conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

    # Tracker output (ORANGE)
    if tracked_objects:
        for t in tracked_objects:
            x1, y1, x2, y2 = t["bbox"]
            tid = t.get("track_id", -1)
            cname = t.get("class_name", "obj")
            conf = t.get("conf", 0.0)

            # box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (255, 128, 0), 2)

            # label
            cv2.putText(frame,
                        f"ID:{tid} {cname} {conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 128, 0),
                        2)

    return frame


# ---------------------------------------------------------
# Dashboard Overlay
# ---------------------------------------------------------
def overlay_dashboard(frame, tracks, metrics, fps):

    panel = frame.copy()

    # background
    cv2.rectangle(panel, (10, 10), (280, 260), (0, 0, 0), -1)
    cv2.rectangle(panel, (10, 10), (280, 260), (255, 255, 255), 2)

    y = 35

    def put(text):
        nonlocal y
        cv2.putText(panel, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        y += 22

    # Main stats
    put(f"FPS: {fps:.1f}")
    put(f"Active Tracks: {len(tracks)}")
    put(f"Total Vehicles: {metrics.total}")

    # Class-wise counts
    for cls, count in metrics.class_counts.items():
        put(f"{cls}: {count}")

    # speed info
    put(f"Avg Speed: {metrics.avg_speed:.1f} km/h")
    put(f"Max Speed: {metrics.max_speed:.1f} km/h")

    # traffic stats
    put(f"Density: {metrics.density}")
    put(f"Flow Rate: {metrics.flow_rate} veh/min")
    put(f"Congestion: {metrics.congestion}")

    return panel

