import cv2
import time
import logging
from typing import List, Tuple, Any
from pipeline.detector import FaceDetector

logger = logging.getLogger(__name__)


class FaceTracker:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.last_detection = 0
        self.trackers: List[Any] = []
        self.ids: List[int] = []
        self.next_id = 0
        
        logger.info(f"Tracker inicializado: interval={interval}s")
    
    def _create_tracker(self):
        try:
            # OpenCV 4.5.1+
            return cv2.legacy.TrackerKCF_create()
        except AttributeError:
            pass
        
        raise RuntimeError(
            "No se pudo crear el tracker. "
        )
    
    def process(
        self, 
        frame, 
        detector: FaceDetector
    ) -> List[Tuple[int, Any, Tuple[int, int, int, int]]]:
        now = time.time()
        faces: List[Tuple[int, Any, Tuple[int, int, int, int]]] = []
        
        # Re-detectar si pasó el intervalo
        if now - self.last_detection >= self.interval:
            self._redetect(frame, detector)
            self.last_detection = now
        
        # Actualizar trackers existentes
        valid_trackers: List[Any] = []
        valid_ids: List[int] = []
        
        for tracker, face_id in zip(self.trackers, self.ids):
            try:
                ok, bbox = tracker.update(frame)
            except Exception as e:
                logger.debug(f"Error actualizando tracker {face_id}: {e}")
                continue
            
            if not ok:
                logger.debug(f"Tracker {face_id} perdió el objeto")
                continue
            
            x, y, w, h = map(int, bbox)
            
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                logger.debug(f"Tracker {face_id} tiene bbox inválido")
                continue
            
            try:
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue
            except Exception as e:
                logger.debug(f"Error extrayendo crop: {e}")
                continue
            
            faces.append((face_id, face_crop, (x, y, w, h)))
            valid_trackers.append(tracker)
            valid_ids.append(face_id)
        
        self.trackers = valid_trackers
        self.ids = valid_ids
        
        return faces
    
    def _redetect(self, frame, detector: FaceDetector):
        self.trackers.clear()
        self.ids.clear()
        
        boxes = detector.detect(frame)
        
        for (x1, y1, x2, y2) in boxes:
            w = x2 - x1
            h = y2 - y1
            
            if w <= 0 or h <= 0:
                continue
            
            try:
                tracker = self._create_tracker()
                success = tracker.init(frame, (x1, y1, w, h))
                
                if success:
                    self.trackers.append(tracker)
                    self.ids.append(self.next_id)
                    logger.debug(f"Tracker {self.next_id} creado")
                    self.next_id += 1
                    
            except Exception as e:
                logger.error(f"Error creando tracker: {e}")
                continue
        
        logger.debug(f"Total trackers activos: {len(self.trackers)}")
    
    def reset(self):
        """Resetea todos los trackers"""
        self.trackers.clear()
        self.ids.clear()
        self.next_id = 0
        logger.info("Trackers reseteados")