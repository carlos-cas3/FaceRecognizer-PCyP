import cv2
import time
import json
import struct
import logging
import zmq
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class RegisterClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.enabled = False
    
    def connect(self) -> bool:
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUSH)
            self.socket.setsockopt(zmq.SNDHWM, 100)
            self.socket.connect(self.endpoint)
            self.enabled = True
            logger.info(f"RegisterClient conectado: {self.endpoint}")
            return True
        except Exception as e:
            logger.error(f"Error conectando RegisterClient: {e}")
            self.enabled = False
            return False
    
    def send_register_request(
        self,
        frame,
        face_id: int,
        bbox: Tuple[int, int, int, int],
        person_name: str,
        camera_id: str = "cam_1"
    ) -> bool:
        if not self.enabled or not self.socket:
            return False
        
        try:
            x, y, w, h = bbox
            
            if frame is None or frame.size == 0:
                logger.error("Frame inválido para registro")
                return False
            
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                logger.error(f"Crop vacío para registro: {bbox}")
                return False
            
            face_resized = cv2.resize(face_crop, (112, 112))
            
            success, jpeg_bytes = cv2.imencode(
                '.jpg',
                face_resized,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            
            if not success:
                logger.error("Error codificando JPEG para registro")
                return False
            
            jpeg_bytes = jpeg_bytes.tobytes()
            
            header = {
                "camera_id": camera_id,
                "face_id": int(face_id),
                "mode": "register",
                "timestamp": time.time(),
                "bbox": [int(x), int(y), int(w), int(h)],
                "person_name": person_name
            }
            
            header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
            header_len = len(header_json)
            message = struct.pack('!I', header_len) + header_json + jpeg_bytes
            
            self.socket.send(message, zmq.NOBLOCK)
            
            logger.info(f"[REGISTER] Enviado - Face ID: {face_id}, Nombre: {person_name}")
            return True
            
        except zmq.Again:
            logger.warning("[REGISTER] Buffer lleno")
            return False
        except Exception as e:
            logger.error(f"[REGISTER] Error: {e}", exc_info=True)
            return False
    
    def close(self):
        logger.info("Cerrando RegisterClient...")
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.enabled = False
        logger.info("RegisterClient cerrado")
    
    @property
    def is_connected(self) -> bool:
        return self.enabled and self.socket is not None
