import cv2
import time
import json
import struct
import logging
import zmq
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Cliente ZMQ - solo envío, sin recepción

class ZMQClient:
    
    def __init__(self, send_endpoint: str):
        self.send_endpoint = send_endpoint
        
        self.context: Optional[zmq.Context] = None
        self.send_socket: Optional[zmq.Socket] = None
        
        self.enabled = False
    
    def connect(self):
        try:
            self.context = zmq.Context()
            self.send_socket = self.context.socket(zmq.PUSH)
            self.send_socket.setsockopt(zmq.SNDHWM, 100)
            self.send_socket.connect(self.send_endpoint)
            
            self.enabled = True
            logger.info(f"ZMQ conectado: {self.send_endpoint}")
            return True
            
        except Exception as e:
            logger.error(f"Error conectando ZMQ: {e}")
            self.enabled = False
            return False
    
    def send_face(
        self,
        frame,
        face_id: int,
        bbox: Tuple[int, int, int, int],
        mode: str,
        person_name: Optional[str] = None,
        camera_id: str = "cam_1"
    ) -> bool:
        if not self.enabled or not self.send_socket:
            return False
        
        try:
            x, y, w, h = bbox
            
            if frame is None or frame.size == 0:
                logger.error("Frame inválido")
                return False
            
            # Extraer y redimensionar cara
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                logger.error(f"Crop vacío: {bbox}")
                return False
            
            face_resized = cv2.resize(face_crop, (112, 112))
            
            # Codificar a JPEG
            success, jpeg_bytes = cv2.imencode(
                '.jpg',
                face_resized,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            
            if not success:
                logger.error("Error codificando JPEG")
                return False
            
            jpeg_bytes = jpeg_bytes.tobytes()
            
            # Crear header
            header = {
                "camera_id": camera_id,
                "face_id": int(face_id),
                "mode": mode,
                "timestamp": time.time(),
                "bbox": [int(x), int(y), int(w), int(h)]
            }
            
            if mode == "register" and person_name:
                header["person_name"] = person_name
            
            # Serializar y enviar
            header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
            header_len = len(header_json)
            message = struct.pack('!I', header_len) + header_json + jpeg_bytes
            
            self.send_socket.send(message, zmq.NOBLOCK)
            
            logger.debug(f"[ZMQ] Enviado - Face ID: {face_id}, Mode: {mode}")
            return True
            
        except zmq.Again:
            logger.warning("[ZMQ] Buffer lleno")
            return False
        except Exception as e:
            logger.error(f"[ZMQ] Error: {e}", exc_info=True)
            return False
    
    def close(self):
        logger.info("Cerrando ZMQ...")
        
        if self.send_socket:
            self.send_socket.close()
        
        if self.context:
            self.context.term()
        
        self.enabled = False
        logger.info("ZMQ cerrado")
    
    @property
    def is_connected(self) -> bool:
        return self.enabled and self.send_socket is not None