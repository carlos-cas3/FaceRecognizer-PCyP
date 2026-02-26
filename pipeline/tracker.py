import cv2
import time
import logging
from typing import List, Tuple, Any, Dict
from pipeline.detector import FaceDetector

logger = logging.getLogger(__name__)


class FaceTracker:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.last_detection = 0
        self.trackers: List[Any] = []
        self.ids: List[int] = []
        self.next_id = 0
        
        # NUEVO: Guardamos (bbox, timestamp) para dar un "periodo de gracia"
        self.last_boxes: Dict[int, Tuple[Tuple[int, int, int, int], float]] = {}
        self.memory_timeout = 2.0  # Segundos que recordamos una cara perdida
        
        logger.info(f"Tracker inicializado: interval={interval}s")
    
    def _create_tracker(self):
        try:
            return cv2.legacy.TrackerKCF_create()
        except AttributeError:
            pass
        
        raise RuntimeError(
            "No se pudo crear el tracker. "
            "Instala opencv-contrib-python. "
            f"OpenCV version: {cv2.__version__}"
        )
    
    def _calculate_iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        if interArea == 0:
            return 0.0
            
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def process(
        self, 
        frame, 
        detector: FaceDetector
    ) -> List[Tuple[int, Any, Tuple[int, int, int, int]]]:
        now = time.time()
        faces: List[Tuple[int, Any, Tuple[int, int, int, int]]] = []
        
        if now - self.last_detection >= self.interval:
            self._redetect(frame, detector)
            self.last_detection = now
        
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
                continue
            
            try:
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue
            except Exception as e:
                logger.debug(f"Error extrayendo crop: {e}")
                continue
            
            bbox_format = (x, y, w, h)
            faces.append((face_id, face_crop, bbox_format))
            valid_trackers.append(tracker)
            valid_ids.append(face_id)
        
        self.trackers = valid_trackers
        self.ids = valid_ids
        
        # NUEVO: Actualizamos la memoria de posiciones con el timestamp actual
        for face_id, _, bbox in faces:
            self.last_boxes[face_id] = (bbox, now)
            
        # NUEVO: Limpiamos SOLO las caras que llevan más de 2 segundos perdidas
        expired_ids = [fid for fid, (box, ts) in self.last_boxes.items() if now - ts > self.memory_timeout]
        for fid in expired_ids:
            del self.last_boxes[fid]
            logger.debug(f"Memoria de ID {fid} expirada y borrada")
        
        return faces
    
    def _redetect(self, frame, detector: FaceDetector):
        boxes = detector.detect(frame)
        new_boxes_xywh = []
        
        for (x1, y1, x2, y2) in boxes:
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                new_boxes_xywh.append((x1, y1, w, h))
        
        new_trackers = []
        new_ids = []
        matched_old_ids = set()
        
        for new_box in new_boxes_xywh:
            best_iou = 0.0
            best_id = -1
            
            # Buscamos en la memoria (incluyendo las caras recién perdidas)
            for old_id, (old_box, _) in self.last_boxes.items():
                if old_id in matched_old_ids:
                    continue
                
                iou = self._calculate_iou(new_box, old_box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = old_id
            
            # Bajamos el umbral a 0.15 para ser más permisivos con el movimiento rápido
            if best_iou > 0.15:
                assigned_id = best_id
                matched_old_ids.add(best_id)
                logger.debug(f"ID {assigned_id} mantenido (IoU: {best_iou:.2f})")
            else:
                assigned_id = self.next_id
                self.next_id += 1
                logger.debug(f"Nuevo rostro detectado. Asignando ID {assigned_id}")
            
            try:
                tracker = self._create_tracker()
                success = tracker.init(frame, new_box)
                
                if success:
                    new_trackers.append(tracker)
                    new_ids.append(assigned_id)
            except Exception as e:
                logger.error(f"Error creando tracker: {e}")
                continue
        
        self.trackers = new_trackers
        self.ids = new_ids
    
    def reset(self):
        self.trackers.clear()
        self.ids.clear()
        self.last_boxes.clear()
        self.next_id = 0
        logger.info("Trackers reseteados")
