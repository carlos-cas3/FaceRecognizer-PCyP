import logging
from typing import List, Tuple
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class FaceDetector:
    
    def __init__(self, model_path: str, confidence: float = 0.5):
        self.model_path = model_path
        self.confidence = confidence
        
        logger.info(f"Cargando modelo YOLO desde: {model_path}")
        self.model = YOLO(model_path)
        logger.info("Modelo YOLO cargado correctamente")
    
    def detect(self, frame) -> List[Tuple[int, int, int, int]]:
        results = self.model(
            frame, 
            conf=self.confidence, 
            device="cpu", 
            verbose=False
        )
        
        boxes = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        
        logger.debug(f"Detectados {len(boxes)} rostros")
        return boxes