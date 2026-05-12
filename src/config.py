# config.py
# Project configuration parameters

VIDEO_SOURCE = 'dataset/sample_video.mp4'
MODEL_PATH = 'models/yolov8n.pt'
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

CLASS_NAMES = [
    'person','bicycle','car','motorbike','aeroplane','bus','truck','train','boat'
]

FRAME_RATE = 30.0

# IMPORTANT: SET PROPER PIXEL TO METER SCALE
# You can tune between 0.03–0.08 depending on your video angle
PIXEL_TO_METER = 0.05       

# ROI for congestion:
ROI_POLYGON = None  

OCCUPANCY_THRESHOLDS = {
    'free': 0.3,
    'moderate': 0.6,
    'heavy': 0.9
}

TRACKER_IOU_THRESHOLD = 0.3
TRACKER_MAX_AGE = 30
TRACKER_MIN_HITS = 3
