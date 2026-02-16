import logging
from typing import Optional, List, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

# FrameManager se encarga de manejar el estado de pausa del frame y las caras detectadas 
class FrameManager:
    def __init__(self):
        self.paused_frame: Optional[np.ndarray] = None
        self.paused_faces: List[Tuple[int, Any, Tuple[int, int, int, int]]] = []
        self._is_paused = False
    
    def pause(self, frame: np.ndarray, faces: List[Tuple[int, Any, Tuple[int, int, int, int]]]):
        if not self._is_paused or self.paused_frame is None:
            self.paused_frame = frame.copy()
            self.paused_faces = faces.copy()
            self._is_paused = True
            logger.debug(f"Frame pausado con {len(faces)} caras")
    
    def resume(self):
        self.paused_frame = None
        self.paused_faces = []
        self._is_paused = False
        logger.debug("Frame reanudado")
    
    def get_frame(self, live_frame: np.ndarray) -> np.ndarray:
        return self.paused_frame if self._is_paused else live_frame
    
    def get_faces(
        self,
        live_faces: List[Tuple[int, Any, Tuple[int, int, int, int]]]
    ) -> List[Tuple[int, Any, Tuple[int, int, int, int]]]:
        return self.paused_faces if self._is_paused else live_faces
    
    @property
    def is_paused(self) -> bool:
        return self._is_paused
    
    def clear(self):
        self.resume()