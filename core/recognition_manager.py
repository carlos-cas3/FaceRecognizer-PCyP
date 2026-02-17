import time
import logging
from typing import Dict, List, Optional, Any
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
        recognition_timeout: float = 5.0,
        send_interval: float = 1.0,
        confidence_threshold: float = 0.7
    ):
        self.identities: Dict[int, RecognizedIdentity] = {}
        self.last_send_time: Dict[int, float] = {}
        self.recognition_timeout = recognition_timeout
        self.send_interval = send_interval
        self.confidence_threshold = confidence_threshold
    
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
        confidence: float
    ):
        if confidence >= self.confidence_threshold:
            self.identities[face_id] = RecognizedIdentity(
                person_id=person_id,
                person_name=person_name,
                confidence=confidence,
                timestamp=time.time()
            )
            logger.info(f"Reconocimiento: Face {face_id} = {person_name} ({confidence:.2%})")
        else:
            logger.debug(f"Confianza baja ({confidence:.2%}) para face {face_id}, ignorando")
    
    def get_identity(self, face_id: int) -> Optional[RecognizedIdentity]:
        identity = self.identities.get(face_id)
        if identity:
            if time.time() - identity.timestamp > self.recognition_timeout:
                del self.identities[face_id]
                return None
        return identity
    
    def is_recognized(self, face_id: int) -> bool:
        return self.get_identity(face_id) is not None
    
    def get_all_identities(self) -> Dict[int, RecognizedIdentity]:
        return self.identities.copy()
    
    def cleanup_expired(self):
        now = time.time()
        expired_ids = [
            face_id for face_id, identity in self.identities.items()
            if now - identity.timestamp > self.recognition_timeout
        ]
        for face_id in expired_ids:
            del self.identities[face_id]
            logger.debug(f"Reconocimiento expirado para face {face_id}")
        
        old_sends = [
            face_id for face_id, last_send in self.last_send_time.items()
            if now - last_send > self.recognition_timeout * 2
        ]
        for face_id in old_sends:
            del self.last_send_time[face_id]
    
    def clear_all(self):
        self.identities.clear()
        self.last_send_time.clear()
        logger.debug("Reconocimientos limpiados")
