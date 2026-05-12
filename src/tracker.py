# tracker.py
"""
Simple IoU-based tracker (keeps it self-contained).
Each track has: id, bbox, class_id, class_name, conf, hits, age, last_seen.
Matching: greedy matching by IoU.
This is not ByteTrack but works well for demos and experiments.
"""

import numpy as np
from collections import deque
from utils import iou


class Track:
    def __init__(self, tid, bbox, class_id, conf, frame_id):
        """
        bbox expected as (x1, y1, x2, y2) -- may be any numeric type; we'll convert on output.
        """
        self.id = int(tid)
        self.bbox = tuple(int(v) for v in bbox)
        self.class_id = int(class_id) if class_id is not None else None
        self.class_name = None
        self.conf = float(conf) if conf is not None else 0.0
        self.hits = 1            # number of times this track was matched
        self.age = 0             # frames since last matched (incremented when unmatched)
        self.last_seen = int(frame_id)
        self.history = deque(maxlen=30)  # centroid history
        x1, y1, x2, y2 = self.bbox
        self.history.append(((x1 + x2) // 2, (y1 + y2) // 2))


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=1):
        """
        iou_threshold: minimum IoU to consider a match
        max_age: maximum frames to keep an unmatched track before deletion
        min_hits: minimum consecutive matches before a track is returned as 'confirmed'
        """
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)

        self.tracks = []     # list of Track objects
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections):
        """
        detections: list of dicts like:
           {'bbox': (x1,y1,x2,y2), 'class_id': int, 'conf': float, 'class_name': str}
        returns: list of dicts for confirmed tracks:
           [{'track_id': id, 'bbox': (x1,y1,x2,y2), 'class_id': int,
             'class_name': str, 'conf': float}, ...]
        """

        self.frame_count += 1
        frame_id = self.frame_count

        # Short-circuit: if no detections, age all tracks and purge old ones
        if not detections or len(detections) == 0:
            # increment age for all tracks
            for track in self.tracks:
                track.age += 1
            # purge
            self.tracks = [t for t in self.tracks if (frame_id - t.last_seen) <= self.max_age]
            # return confirmed tracks
            return self._export_confirmed_tracks()

        # Build IoU matrix between existing tracks and new detections
        num_t = len(self.tracks)
        num_d = len(detections)
        iou_mat = np.zeros((num_t, num_d), dtype=float)

        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                try:
                    iou_mat[t_idx, d_idx] = float(iou(track.bbox, det['bbox']))
                except Exception:
                    iou_mat[t_idx, d_idx] = 0.0

        # Greedy matching: pick highest IoU pair repeatedly
        matches = []
        unmatched_tracks = list(range(num_t))
        unmatched_dets = list(range(num_d))

        if iou_mat.size > 0:
            iou_arr = iou_mat.copy()
            while True:
                # find index of largest IoU
                t_idx, d_idx = np.unravel_index(iou_arr.argmax(), iou_arr.shape)
                max_val = iou_arr[t_idx, d_idx]
                if max_val < self.iou_threshold:
                    break
                # accept match
                matches.append((int(t_idx), int(d_idx)))
                # invalidate row and column
                iou_arr[t_idx, :] = -1.0
                iou_arr[:, d_idx] = -1.0
                if t_idx in unmatched_tracks:
                    unmatched_tracks.remove(t_idx)
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)

        # Update matched tracks with detection info
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            det = detections[d_idx]

            # update bounding box and meta
            track.bbox = tuple(int(v) for v in det.get('bbox', track.bbox))
            track.class_id = det.get('class_id', track.class_id)
            track.class_name = det.get('class_name', track.class_name)
            track.conf = float(det.get('conf', det.get('confidence', track.conf)))
            track.hits += 1
            track.age = 0
            track.last_seen = frame_id

            # append centroid
            x1, y1, x2, y2 = track.bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            track.history.append((cx, cy))

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            bbox = det.get('bbox')
            class_id = det.get('class_id', None)
            conf = det.get('conf', det.get('confidence', 0.0))
            new_track = Track(self.next_id, bbox, class_id, conf, frame_id)
            new_track.class_name = det.get('class_name', None)
            self.tracks.append(new_track)
            self.next_id += 1

        # Increment age for unmatched tracks
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            track.age += 1

        # Purge old tracks
        self.tracks = [t for t in self.tracks if (frame_id - t.last_seen) <= self.max_age]

        # Prepare and return confirmed tracks
        return self._export_confirmed_tracks()

    def _export_confirmed_tracks(self):
        """
        Return a list of track dicts for tracks that have hits >= min_hits.
        Ensures bbox values are integer tuples and confidence is float.
        """
        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                out.append({
                    'track_id': int(t.id),
                    'bbox': tuple(int(v) for v in t.bbox),
                    'class_id': int(t.class_id) if t.class_id is not None else None,
                    'class_name': t.class_name,
                    'conf': float(t.conf)
                })
        return out
