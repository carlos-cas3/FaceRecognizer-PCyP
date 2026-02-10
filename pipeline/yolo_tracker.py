import cv2
import time
from ultralytics import YOLO

class FaceTracker:
    def __init__(self, model_path, interval=0.5):
        self.model = YOLO(model_path)
        self.interval = interval
        self.last_time = 0
        self.trackers = []
        self.ids = []
        self.next_id = 0

    def process(self, frame):
        now = time.time()
        faces = []

        # Re-detectar cada 'interval' segundos
        if now - self.last_time >= self.interval:
            self.trackers.clear()
            self.ids.clear()

            results = self.model(frame, conf=0.4, device="cpu", verbose=False)
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                    
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1
                    
                    if w <= 0 or h <= 0:
                        continue

                    # Crear tracker KCF para esta cara
                    tracker = cv2.legacy.TrackerKCF_create()  # Usar legacy en OpenCV 4.x
                    tracker.init(frame, (x1, y1, w, h))
                    self.trackers.append(tracker)
                    self.ids.append(self.next_id)
                    self.next_id += 1

            self.last_time = now

        # Actualizar trackers existentes
        valid_trackers = []
        valid_ids = []
        
        for tracker, face_id in zip(self.trackers, self.ids):
            ok, bbox = tracker.update(frame)
            if not ok:
                continue

            x, y, w, h = map(int, bbox)
            
            # Validar límites
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
            
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            faces.append((face_id, face, (x, y, w, h)))
            valid_trackers.append(tracker)
            valid_ids.append(face_id)

        # Actualizar listas con trackers válidos
        self.trackers = valid_trackers
        self.ids = valid_ids

        return faces

    def reset(self):
        """Reinicia todos los trackers"""
        self.trackers.clear()
        self.ids.clear()
        self.next_id = 0