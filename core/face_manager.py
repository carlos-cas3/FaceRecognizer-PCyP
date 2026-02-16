import time
import logging
from typing import Dict, List, Tuple, Any, Set

logger = logging.getLogger(__name__)

# FaceManager se encarga de gestionar IDs persistentes de caras y su seguimiento.
# Mantiene caras "bloqueadas" durante el modo registro para evitar que cambien de ID.

class FaceManager:
    def __init__(self, id_timeout: float = 5.0, match_threshold: int = 50):
        self.locked_faces: Dict[int, Dict] = {}  # face_id -> {'bbox', 'last_seen', 'selected'}
        self.id_timeout = id_timeout
        self.match_threshold = match_threshold
        self.registered_faces: Dict[int, str] = {}  # face_id -> person_name
    
    def match_face_to_locked(self, bbox: Tuple[int, int, int, int]) -> int:
        x, y, w, h = bbox
        cx, cy = x + w//2, y + h//2  # Centro de la cara
        
        for face_id, data in self.locked_faces.items():
            lx, ly, lw, lh = data['bbox']
            lcx, lcy = lx + lw//2, ly + lh//2  # Centro de la cara bloqueada
            
            # Calcular distancia euclidiana entre centros
            distance = ((cx - lcx)**2 + (cy - lcy)**2)**0.5
            
            if distance < self.match_threshold:
                return face_id
        
        return -1  # No encontró coincidencia
    
    def release_expired_ids(self):
        current_time = time.time()
        expired_ids = []
        
        for face_id, data in self.locked_faces.items():
            if current_time - data['last_seen'] > self.id_timeout:
                expired_ids.append(face_id)
        
        for face_id in expired_ids:
            del self.locked_faces[face_id]
            logger.info(f"ID {face_id} liberado por timeout")
    
    def update_locked_faces(self, faces: List[Tuple[int, Any, Tuple[int, int, int, int]]]):
        current_time = time.time()
        
        for face_id, face_crop, bbox in faces:
            locked_id = self.match_face_to_locked(bbox)
            
            if locked_id != -1:
                # Actualizar posición y timestamp de la cara bloqueada
                self.locked_faces[locked_id]['bbox'] = bbox
                self.locked_faces[locked_id]['last_seen'] = current_time
        
        # Liberar IDs expirados
        self.release_expired_ids()
    
    def lock_face(self, face_id: int, bbox: Tuple[int, int, int, int]):
        if face_id not in self.locked_faces:
            self.locked_faces[face_id] = {
                'bbox': bbox,
                'last_seen': time.time(),
                'selected': True
            }
            logger.info(f"Cara {face_id} bloqueada")
    
    def clear_locked_faces(self):
        self.locked_faces.clear()
        logger.debug("Caras bloqueadas limpiadas")
    
    def process_faces_for_register_mode(
        self,
        raw_faces: List[Tuple[int, Any, Tuple[int, int, int, int]]]
    ) -> List[Tuple[int, Any, Tuple[int, int, int, int]]]:
        faces = []
        used_locked_ids: Set[int] = set()
        
        for face_id, face_crop, bbox in raw_faces:
            # Buscar si esta cara coincide con alguna bloqueada
            locked_id = self.match_face_to_locked(bbox)
            
            if locked_id != -1 and locked_id not in used_locked_ids:
                # Usar ID bloqueado
                faces.append((locked_id, face_crop, bbox))
                self.locked_faces[locked_id]['last_seen'] = time.time()
                used_locked_ids.add(locked_id)
                
            elif locked_id == -1:
                # Es una cara nueva, verificar si su ID está bloqueado
                if face_id in self.locked_faces:
                    # El ID está bloqueado pero la cara no coincide (conflicto)
                    # Asignar nuevo ID
                    new_id = max(self.locked_faces.keys()) + 1 if self.locked_faces else face_id
                    faces.append((new_id, face_crop, bbox))
                else:
                    # ID libre, usar tal cual
                    faces.append((face_id, face_crop, bbox))
        
        # Actualizar caras bloqueadas
        self.update_locked_faces(faces)
        
        return faces
    
    def register_face(self, face_id: int, person_name: str):
        self.registered_faces[face_id] = person_name
        logger.info(f"Cara {face_id} registrada como: {person_name}")
    
    def is_face_registered(self, face_id: int) -> bool:
        return face_id in self.registered_faces
    
    def get_face_name(self, face_id: int) -> str:
        return self.registered_faces.get(face_id, "Desconocido")