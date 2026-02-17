import time
import logging
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LockedFace:
    bbox: Tuple[int, int, int, int]
    last_seen: float
    selected: bool

class RegisterManager:
    def __init__(self, id_timeout: float = 5.0, match_threshold: int = 50):
        self.locked_faces: Dict[int, LockedFace] = {}
        self.id_timeout = id_timeout
        self.match_threshold = match_threshold
    
    def _calculate_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def _calculate_distance(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        cx1, cy1 = self._calculate_center(bbox1)
        cx2, cy2 = self._calculate_center(bbox2)
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    
    def _find_matching_locked_id(self, bbox: Tuple[int, int, int, int]) -> int:
        for face_id, locked_face in self.locked_faces.items():
            if self._calculate_distance(bbox, locked_face.bbox) < self.match_threshold:
                return face_id
        return -1
    
    def lock_face(self, face_id: int, bbox: Tuple[int, int, int, int]):
        self.locked_faces[face_id] = LockedFace(
            bbox=bbox,
            last_seen=time.time(),
            selected=True
        )
        logger.info(f"Cara {face_id} bloqueada para registro")
    
    def unlock_face(self, face_id: int):
        if face_id in self.locked_faces:
            del self.locked_faces[face_id]
            logger.debug(f"Cara {face_id} desbloqueada")
    
    def is_locked(self, face_id: int) -> bool:
        return face_id in self.locked_faces
    
    def get_locked_ids(self) -> List[int]:
        return list(self.locked_faces.keys())
    
    def update_locked_position(self, face_id: int, bbox: Tuple[int, int, int, int]):
        if face_id in self.locked_faces:
            self.locked_faces[face_id].bbox = bbox
            self.locked_faces[face_id].last_seen = time.time()
    
    def process_faces(
        self,
        raw_faces: List[Tuple[int, Any, Tuple[int, int, int, int]]]
    ) -> List[Tuple[int, Any, Tuple[int, int, int, int]]]:
        processed_faces: List[Tuple[int, Any, Tuple[int, int, int, int]]] = []
        used_locked_ids: Set[int] = set()
        
        for face_id, face_crop, bbox in raw_faces:
            matching_locked_id = self._find_matching_locked_id(bbox)
            
            if matching_locked_id != -1 and matching_locked_id not in used_locked_ids:
                processed_faces.append((matching_locked_id, face_crop, bbox))
                self.update_locked_position(matching_locked_id, bbox)
                used_locked_ids.add(matching_locked_id)
            elif matching_locked_id == -1:
                if face_id in self.locked_faces:
                    new_id = max(self.locked_faces.keys(), default=0) + 1
                    processed_faces.append((new_id, face_crop, bbox))
                else:
                    processed_faces.append((face_id, face_crop, bbox))
        
        self._cleanup_expired()
        return processed_faces
    
    def _cleanup_expired(self):
        current_time = time.time()
        expired_ids = [
            face_id for face_id, locked_face in self.locked_faces.items()
            if current_time - locked_face.last_seen > self.id_timeout
        ]
        for face_id in expired_ids:
            del self.locked_faces[face_id]
            logger.info(f"ID {face_id} liberado por timeout")
    
    def clear_all(self):
        self.locked_faces.clear()
        logger.debug("Caras bloqueadas limpiadas")
