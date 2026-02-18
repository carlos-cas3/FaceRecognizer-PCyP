import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RecognizedIdentity:
    person_id: str
    person_name: str
    confidence: float
    timestamp: float

class RecognitionManager:
    def __init__(
        self,
        recognition_timeout: float = 10.0,
        send_interval: float = 1.0,
        confidence_threshold: float = 0.7,
        position_match_threshold: int = 50,
        position_cache_timeout: float = 10.0
    ):
        self.identities: Dict[int, RecognizedIdentity] = {}
        self.last_send_time: Dict[int, float] = {}
        self.position_cache: Dict[Tuple[int, int], RecognizedIdentity] = {}
        
        self.recognition_timeout = recognition_timeout
        self.send_interval = send_interval
        self.confidence_threshold = confidence_threshold
        self.position_match_threshold = position_match_threshold
        self.position_cache_timeout = position_cache_timeout
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
    
    def find_match_by_position(self, bbox: Tuple[int, int, int, int]) -> Optional[RecognizedIdentity]:
        center = self._get_center(bbox)
        now = time.time()
        
        best_match = None
        best_distance = float('inf')
        
        for pos, identity in list(self.position_cache.items()):
            if now - identity.timestamp > self.position_cache_timeout:
                del self.position_cache[pos]
                continue
            
            distance = self._calculate_distance(center, pos)
            if distance < self.position_match_threshold and distance < best_distance:
                best_match = identity
                best_distance = distance
        
        return best_match
    
    def cache_position(self, bbox: Tuple[int, int, int, int], identity: RecognizedIdentity):
        center = self._get_center(bbox)
        new_identity = RecognizedIdentity(
            person_id=identity.person_id,
            person_name=identity.person_name,
            confidence=identity.confidence,
            timestamp=time.time()
        )
        self.position_cache[center] = new_identity
        logger.debug(f"Posición cacheada: {center} → {identity.person_name}")
    
    def assign_identity_from_cache(self, face_id: int, bbox: Tuple[int, int, int, int]) -> bool:
        matched = self.find_match_by_position(bbox)
        if matched:
            self.identities[face_id] = RecognizedIdentity(
                person_id=matched.person_id,
                person_name=matched.person_name,
                confidence=matched.confidence,
                timestamp=time.time()
            )
            logger.info(f"Cara {face_id} identificada por posición: {matched.person_name}")
            return True
        return False
    
    def cleanup_position_cache(self):
        now = time.time()
        expired = [
            pos for pos, identity in self.position_cache.items()
            if now - identity.timestamp > self.position_cache_timeout
        ]
        for pos in expired:
            del self.position_cache[pos]
    
    def should_send(self, face_id: int) -> bool:
        now = time.time()
        last_send = self.last_send_time.get(face_id, 0)
        if now - last_send >= self.send_interval:
            return True
        return False
    
    def mark_sent(self, face_id: int):
        self.last_send_time[face_id] = time.time()
    
    def update_identity(
        self,
        face_id: int,
        person_id: str,
        person_name: str,
        confidence: float,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ):
        if confidence >= self.confidence_threshold:
            identity = RecognizedIdentity(
                person_id=person_id,
                person_name=person_name,
                confidence=confidence,
                timestamp=time.time()
            )
            self.identities[face_id] = identity
            logger.info(f"Reconocimiento: Face {face_id} = {person_name} ({confidence:.2%})")
            
            if bbox:
                self.cache_position(bbox, identity)
        else:
            logger.debug(f"Confianza baja ({confidence:.2%}) para face {face_id}, ignorando")
    
    def refresh_identity(self, face_id: int):
        if face_id in self.identities:
            self.identities[face_id].timestamp = time.time()
    
    def refresh_active_faces(self, active_face_ids: List[int]):
        now = time.time()
        for face_id in active_face_ids:
            if face_id in self.identities:
                self.identities[face_id].timestamp = now
    
    def cleanup_not_visible(self, active_face_ids: List[int]):
        to_remove = [
            face_id for face_id in self.identities.keys()
            if face_id not in active_face_ids
        ]
        for face_id in to_remove:
            del self.identities[face_id]
            logger.debug(f"Identidad eliminada (cara no visible): {face_id}")
        
        old_sends = [
            face_id for face_id in list(self.last_send_time.keys())
            if face_id not in active_face_ids
        ]
        for face_id in old_sends:
            del self.last_send_time[face_id]
        
        self.cleanup_position_cache()
    
    def get_identity(self, face_id: int) -> Optional[RecognizedIdentity]:
        return self.identities.get(face_id)
    
    def is_recognized(self, face_id: int) -> bool:
        return face_id in self.identities
    
    def get_all_identities(self) -> Dict[int, RecognizedIdentity]:
        return self.identities.copy()
    
    def cleanup_expired(self):
        pass
    
    def clear_all(self):
        self.identities.clear()
        self.last_send_time.clear()
        self.position_cache.clear()
        logger.debug("Reconocimientos y cache de posiciones limpiados")
