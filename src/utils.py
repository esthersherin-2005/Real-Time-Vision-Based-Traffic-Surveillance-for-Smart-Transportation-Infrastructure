# utils.py
import cv2
import numpy as np

def draw_bbox(frame, bbox, label=None, color=(0,255,0), thickness=2):
    x1,y1,x2,y2 = bbox
    cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, thickness)
    if label:
        cv2.putText(frame, str(label), (int(x1), int(y1)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def centroid(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def bbox_area(bbox):
    x1,y1,x2,y2 = bbox
    w = max(0, x2-x1)
    h = max(0, y2-y1)
    return w*h

def polygon_area(poly):
    if poly is None:
        return None
    pts = np.array(poly, dtype=np.int32)
    return cv2.contourArea(pts)

def iou(boxA, boxB):
    # boxes are (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0.0
    return interArea / union

def point_in_poly(point, poly):
    # point: (x,y), poly list of (x,y) or None
    if poly is None:
        return True
    return cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (int(point[0]), int(point[1])), False) >= 0
