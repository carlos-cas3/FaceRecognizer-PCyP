import logging
import torch
from typing import List, Tuple
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class FaceDetector:
    
    def __init__(self, model_path: str, confidence: float = 0.5, device: str = "cpu"):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device # Guardamos el dispositivo (ej: "0" o "cpu")
        
        logger.info(f"Cargando modelo YOLO desde: {model_path}")
        logger.info(f"Dispositivo de inferencia seleccionado: {self.device}")
        
        # Inicializamos el modelo
        self.model = YOLO(model_path)
        
        # Forzamos la carga inicial a la GPU si corresponde
        if self.device != "cpu" and torch.cuda.is_available():
            logger.info(f"Moviendo modelo a GPU: {torch.cuda.get_device_name(int(self.device))}")
            self.model.to(f"cuda:{self.device}")
        
        logger.info("Modelo YOLO cargado correctamente")
    
    def detect(self, frame) -> List[Tuple[int, int, int, int]]:
        # Pasamos el argumento 'device' explícitamente en cada inferencia
        results = self.model(
            frame, 
            conf=self.confidence, 
            device=self.device, 
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