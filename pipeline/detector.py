import logging
import torch
from typing import List, Tuple
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class FaceDetector:
    
    def __init__(self, model_path: str, confidence: float = 0.5, device: str = "cpu"):
        self.model_path = model_path
        self.confidence = confidence
        
        # Auto-detectar si GPU solicitada pero no disponible
        if device != "cpu" and not torch.cuda.is_available():
            logger.warning(f"GPU solicitada ({device}) pero CUDA no disponible. Usando CPU.")
            device = "cpu"
        
        self.device = device
        
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
        logger.info(f"[DETECTOR] YOLO detectando con device={self.device}, conf={self.confidence}")
        
        # Pasamos el argumento 'device' explícitamente en cada inferencia
        results = self.model(
            frame, 
            conf=self.confidence, 
            device=self.device, 
            verbose=True
        )
        
        boxes = []
        for r in results:
            if r.boxes is None:
                logger.warning("[DETECTOR] YOLO retornó boxes=None")
                continue
            
            num_boxes = len(r.boxes) if r.boxes is not None else 0
            logger.info(f"[DETECTOR] YOLO encontró {num_boxes} cajas en este frame")
            
            if num_boxes == 0:
                continue
            
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
                    logger.info(f"[DETECTOR] Box válido: ({x1}, {y1}, {x2}, {y2})")
        
        logger.info(f"[DETECTOR] Total boxes retornados: {len(boxes)}")
        return boxes