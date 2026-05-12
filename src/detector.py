# detector.py
from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path, conf=0.35, iou=0.45, device=None):
        """
        device: None -> auto-select, or 'cpu', 'cuda:0'
        """
        self.model = YOLO(model_path)

        if device:
            self.model.to(device)

        self.conf = conf
        self.iou = iou

    def predict(self, frame):
        """
        returns list of dict:
        {
          'bbox': (x1,y1,x2,y2),
          'conf': float,
          'class_id': int,
          'class_name': str
        }
        """
        # Convert to RGB for YOLO
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO
        results = self.model.predict(
            source=[img],
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )

        if len(results) == 0:
            return []

        r = results[0]
        dets = []

        # Parse detections
        for box in r.boxes:
            xyxy = box.xyxy.cpu().numpy().flatten()
            conf = float(box.conf.cpu().numpy())
            cls_id = int(box.cls.cpu().numpy())
            cls_name = self.model.names[cls_id]

            x1, y1, x2, y2 = map(int, xyxy)

            # filter tiny boxes
            if (x2 - x1) < 15 or (y2 - y1) < 15:
                continue

            dets.append({
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'class_id': cls_id,
                'class_name': cls_name
            })

        return dets
