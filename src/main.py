# main.py
import cv2
import time
import argparse
import numpy as np
from detector import YOLODetector
from tracker import SimpleTracker
from metrics import MetricsEngine
from visualization import draw_detections, overlay_dashboard
import config as cfg


def filter_vehicle_detections(dets):
    """
    Filter YOLO detections to only vehicle classes.
    """
    vehicle_names = {
        'car', 'truck', 'bus', 'motorbike',
        'bicycle', 'autorickshaw', 'motorcycle', 'van'
    }

    filtered = []
    for d in dets:
        name = d.get('class_name', '').lower()
        if name in vehicle_names or ('car' in name) or ('truck' in name) or ('bus' in name) or ('motor' in name):
            filtered.append(d)
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=cfg.VIDEO_SOURCE)
    parser.add_argument('--model', default=cfg.MODEL_PATH)
    parser.add_argument('--conf', type=float, default=cfg.CONF_THRESHOLD)
    args = parser.parse_args()

    # Load video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("ERROR: Could not open video:", args.source)
        return

    # FPS fallback
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_val = vid_fps if vid_fps and vid_fps > 0 else cfg.FRAME_RATE

    # Initialize core components
    detector = YOLODetector(args.model, conf=args.conf)
    tracker = SimpleTracker(
        iou_threshold=cfg.TRACKER_IOU_THRESHOLD,
        max_age=cfg.TRACKER_MAX_AGE,
        min_hits=cfg.TRACKER_MIN_HITS
    )
    metrics = MetricsEngine(
        roi_poly=cfg.ROI_POLYGON,
        fps=fps_val,
        pixel_to_meter=cfg.PIXEL_TO_METER
    )

    smoothed_fps = 0.0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        t0 = time.time()

        # 1) YOLO Detection
        dets = detector.predict(frame)



        # 2) Filter vehicles only
        dets = filter_vehicle_detections(dets)

        # 3) Tracking
        tracks = tracker.update(dets)
        print("TRACKS SAMPLE:", tracks[:3])


        # 4) Metrics update
        metrics.update(tracks, timestamp=time.time())

        # 5) FPS calculation
        fps_now = 1.0 / (time.time() - t0 + 1e-6)
        smoothed_fps = fps_now if smoothed_fps == 0 else (0.9 * smoothed_fps + 0.1 * fps_now)

        # 6) Draw bounding boxes + IDs
        frame = draw_detections(frame, dets, tracks)

        # 7) Overlay Dashboard
        out = overlay_dashboard(frame, tracks, metrics, smoothed_fps)

        # Display window
        cv2.imshow("Vehicle Detection System", out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
