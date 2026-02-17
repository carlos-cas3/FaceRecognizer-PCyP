import cv2
import time
import json
import struct
import logging
import zmq
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RecognitionResult:
    face_id: int
    person_id: str
    person_name: str
    confidence: float

class RecognitionClient:
    def __init__(self, send_endpoint: str, recv_endpoint: str):
        self.send_endpoint = send_endpoint
        self.recv_endpoint = recv_endpoint
        
        self.context: Optional[zmq.Context] = None
        self.send_socket: Optional[zmq.Socket] = None
        self.recv_socket: Optional[zmq.Socket] = None
        
        self.enabled = False
    
    def connect(self) -> bool:
        try:
            self.context = zmq.Context()
            
            self.send_socket = self.context.socket(zmq.PUSH)
            self.send_socket.setsockopt(zmq.SNDHWM, 100)
            self.send_socket.connect(self.send_endpoint)
            
            self.recv_socket = self.context.socket(zmq.PULL)
            self.recv_socket.setsockopt(zmq.RCVHWM, 100)
            self.recv_socket.connect(self.recv_endpoint)
            
            self.enabled = True
            logger.info(f"RecognitionClient conectado - Send: {self.send_endpoint}, Recv: {self.recv_endpoint}")
            return True
            
        except Exception as e:
            logger.error(f"Error conectando RecognitionClient: {e}")
            self.enabled = False
            return False
    
    def send_recognition_request(
        self,
        frame,
        face_id: int,
        bbox: Tuple[int, int, int, int],
        camera_id: str = "cam_1"
    ) -> bool:
        if not self.enabled or not self.send_socket:
            return False
        
        try:
            x, y, w, h = bbox
            
            if frame is None or frame.size == 0:
                logger.error("Frame inválido para reconocimiento")
                return False
            
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                logger.error(f"Crop vacío para reconocimiento: {bbox}")
                return False
            
            face_resized = cv2.resize(face_crop, (112, 112))
            
            success, jpeg_bytes = cv2.imencode(
                '.jpg',
                face_resized,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            
            if not success:
                logger.error("Error codificando JPEG para reconocimiento")
                return False
            
            jpeg_bytes = jpeg_bytes.tobytes()
            
            header = {
                "camera_id": camera_id,
                "face_id": int(face_id),
                "mode": "recognize",
                "timestamp": time.time(),
                "bbox": [int(x), int(y), int(w), int(h)]
            }
            
            header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
            header_len = len(header_json)
            message = struct.pack('!I', header_len) + header_json + jpeg_bytes
            
            self.send_socket.send(message, zmq.NOBLOCK)
            
            logger.debug(f"[RECOGNIZE] Enviado - Face ID: {face_id}")
            return True
            
        except zmq.Again:
            logger.warning("[RECOGNIZE] Buffer lleno")
            return False
        except Exception as e:
            logger.error(f"[RECOGNIZE] Error enviando: {e}", exc_info=True)
            return False
    
    def receive_result(self, timeout_ms: int = 10) -> Optional[RecognitionResult]:
        if not self.enabled or not self.recv_socket:
            return None
        
        try:
            if self.recv_socket.poll(timeout_ms):
                message = self.recv_socket.recv_json(flags=zmq.NOBLOCK)
                
                face_id_raw = message.get("face_id", -1)
                person_id_raw = message.get("person_id", "")
                
                if isinstance(face_id_raw, str) and '-' in str(face_id_raw):
                    person_id = str(face_id_raw)
                    face_id = -1
                    logger.debug(f"[RECOGNIZE] Formato legacy: UUID en face_id, usando -1")
                else:
                    try:
                        face_id = int(face_id_raw)
                    except (ValueError, TypeError):
                        face_id = -1
                    person_id = str(person_id_raw) if person_id_raw else ""
                
                result = RecognitionResult(
                    face_id=face_id,
                    person_id=person_id,
                    person_name=str(message.get("person_name", "Desconocido")),
                    confidence=float(message.get("confidence", 0.0))
                )
                
                logger.debug(f"[RECOGNIZE] Recibido: {result.person_name} ({result.confidence:.2%})")
                return result
                
        except zmq.Again:
            pass
        except Exception as e:
            logger.error(f"[RECOGNIZE] Error recibiendo: {e}")
        
        return None
    
    def close(self):
        logger.info("Cerrando RecognitionClient...")
        if self.send_socket:
            self.send_socket.close()
        if self.recv_socket:
            self.recv_socket.close()
        if self.context:
            self.context.term()
        self.enabled = False
        logger.info("RecognitionClient cerrado")
    
    @property
    def is_connected(self) -> bool:
        return self.enabled and self.send_socket is not None and self.recv_socket is not None
