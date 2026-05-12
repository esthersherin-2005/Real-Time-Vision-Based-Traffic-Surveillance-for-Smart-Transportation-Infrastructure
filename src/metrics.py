# metrics.py (paste whole file)
import time
import math
from collections import deque

def _get_track_id(track):
    for k in ("track_id","id","tid"):
        if k in track:
            return track[k]
    return None

def _get_bbox(track):
    for k in ("bbox","xyxy","box"):
        if k in track:
            return tuple(track[k])
    # try keys of nested values
    return None

def _get_class(track):
    for k in ("class_name","cls","class","label"):
        if k in track:
            return track[k]
    return "unknown"

def _get_conf(track):
    for k in ("conf","confidence","score"):
        if k in track:
            try:
                return float(track[k])
            except:
                return 0.0
    return 0.0

class MetricsEngine:
    def __init__(self, roi_poly=None, fps=30, pixel_to_meter=None):
        self.roi_poly = roi_poly
        self.fps = fps
        self.meters_per_pixel = pixel_to_meter if pixel_to_meter is not None else 0.04

        # aggregated stats
        self.total = 0             # unique stable tracks seen
        self.class_counts = {}     # counts in current frame
        self.avg_speed = 0.0
        self.max_speed = 0.0
        self.flow_rate = 0.0
        self.density = 0
        self.congestion = "LOW"

        # internal
        self.seen_ids = set()
        self.last_positions = {}   # track_id -> (cx,cy)
        self.last_time = {}        # track_id -> timestamp
        self.per_track_speeds = {} # track_id -> deque for smoothing
        self.speeds_global = deque(maxlen=1000)  # cap global speeds memory

        # flow window
        self.flow_window_start = time.time()
        self.vehicles_in_window = 0
        self.flow_window_seconds = 60

    def _center(self, bbox):
        if bbox is None:
            return None
        x1,y1,x2,y2 = bbox
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def update(self, tracks, timestamp=None):
        now = timestamp if timestamp is not None else time.time()
        self.class_counts = {}
        self.density = len(tracks)
        frame_speeds = []

        for track in tracks:
            # accept dicts
            if not isinstance(track, dict):
                try:
                    track = vars(track)
                except:
                    continue

            tid = _get_track_id(track)
            bbox = _get_bbox(track)
            cname = str(_get_class(track))
            conf = _get_conf(track)

            center = self._center(bbox)
            if center is None:
                cx = cy = None
            else:
                cx, cy = center

            # per-class count
            self.class_counts[cname] = self.class_counts.get(cname, 0) + 1

            # count unique stable tracks (main program should only pass stable tracks)
            if tid is not None and tid not in self.seen_ids:
                # mark seen and increment total only once
                self.seen_ids.add(tid)
                self.total += 1
                self.vehicles_in_window += 1

            # speed estimation using last_positions
            if tid is not None and cx is not None and tid in self.last_positions:
                prev_cx, prev_cy = self.last_positions[tid]
                prev_t = self.last_time.get(tid, now)
                dt = now - prev_t
                if dt > 0:
                    dx = cx - prev_cx
                    dy = cy - prev_cy
                    pixel_dist = math.hypot(dx, dy)
                    meters = pixel_dist * (self.meters_per_pixel if self.meters_per_pixel is not None else 0.04)
                    speed_m_s = meters / dt
                    speed_kmh = speed_m_s * 3.6
                    if 0 < speed_kmh < 300:
                        # per track smoothing
                        if tid not in self.per_track_speeds:
                            self.per_track_speeds[tid] = deque(maxlen=5)
                        self.per_track_speeds[tid].append(speed_kmh)
                        # use smoothed per-track speed
                        smoothed = sum(self.per_track_speeds[tid]) / len(self.per_track_speeds[tid])
                        frame_speeds.append(smoothed)
                        self.speeds_global.append(smoothed)

            # update last pos/time
            if tid is not None and cx is not None:
                self.last_positions[tid] = (cx, cy)
                self.last_time[tid] = now

        # update speed aggregates
        if len(self.speeds_global) > 0:
            self.avg_speed = sum(self.speeds_global) / len(self.speeds_global)
            self.max_speed = max(self.speeds_global)

        # update flow rate per minute sliding window
        if now - self.flow_window_start >= self.flow_window_seconds:
            self.flow_rate = self.vehicles_in_window * (60.0 / self.flow_window_seconds)
            self.vehicles_in_window = 0
            self.flow_window_start = now

        # congestion heuristic (based on density)
        if self.density > 25:
            self.congestion = "HIGH"
        elif self.density > 10:
            self.congestion = "MEDIUM"
        else:
            self.congestion = "LOW"
