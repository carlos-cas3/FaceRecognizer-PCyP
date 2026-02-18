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
class RegisterResult:
    face_id: int
    person_id: str
    person_name: str
    success: bool

class RegisterClient:
    def __init__(self, send_endpoint: str, recv_endpoint: str):
        self.send_endpoint = send_endpoint
        self.recv_endpoint = recv_endpoint
        
        self.context: Optional[zmq.Context] = None
        self.send_socket: Optional[zmq.Socket] = None
        self.recv_socket: Optional[zmq.Socket] = None
        
        self._monitor_socket: Optional[zmq.Socket] = None
        self._connected = False
        self._last_connect_attempt = 0
        
        self.enabled = False
    
    def connect(self) -> bool:
        try:
            self.context = zmq.Context()
            
            self.send_socket = self.context.socket(zmq.PUSH)
            self.send_socket.setsockopt(zmq.SNDHWM, 100)
            
            monitor_addr = f"inproc://monitor-register-{id(self)}"
            self.send_socket.monitor(monitor_addr, zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED | zmq.EVENT_CONNECT_DELAYED)
            
            self._monitor_socket = self.context.socket(zmq.PAIR)
            self._monitor_socket.setsockopt(zmq.RCVTIMEO, 0)
            self._monitor_socket.connect(monitor_addr)
            
            self.send_socket.connect(self.send_endpoint)
            
            self.recv_socket = self.context.socket(zmq.PULL)
            self.recv_socket.setsockopt(zmq.RCVHWM, 100)
            self.recv_socket.connect(self.recv_endpoint)
            
            self.enabled = True
            logger.info(f"RegisterClient inicializado - Send: {self.send_endpoint}, Recv: {self.recv_endpoint}")
            return True
            
        except Exception as e:
            logger.error(f"Error conectando RegisterClient: {e}")
            self.enabled = False
            return False
    
    def check_connection(self):
        if not self._monitor_socket:
            return
        
        try:
            while self._monitor_socket.poll(0):
                message = self._monitor_socket.recv_multipart(flags=zmq.NOBLOCK)
                if len(message) >= 1:
                    event_data = message[0]
                    if len(event_data) >= 4:
                        event_id = struct.unpack('=H', event_data[:2])[0]
                        
                        if event_id == zmq.EVENT_CONNECTED:
                            self._connected = True
                            logger.info("[REGISTER] Conectado al servidor C++")
                        elif event_id == zmq.EVENT_DISCONNECTED:
                            self._connected = False
                            logger.warning("[REGISTER] Desconectado del servidor C++")
                        elif event_id == zmq.EVENT_CONNECT_DELAYED:
                            self._connected = False
        except zmq.Again:
            pass
        except Exception as e:
            logger.debug(f"[REGISTER] Error monitoreando conexión: {e}")
    
    def send_register_request(
        self,
        frame,
        face_id: int,
        bbox: Tuple[int, int, int, int],
        person_name: str,
        camera_id: str = "cam_1"
    ) -> bool:
        if not self.enabled or not self.send_socket:
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
            
            self.send_socket.send(message, zmq.NOBLOCK)
            
            logger.info(f"[REGISTER] Enviado - Face ID: {face_id}, Nombre: {person_name}")
            return True
            
        except zmq.Again:
            logger.warning("[REGISTER] Buffer lleno")
            return False
        except Exception as e:
            logger.error(f"[REGISTER] Error: {e}", exc_info=True)
            return False
    
    def receive_confirmation(self, timeout_ms: int = 10) -> Optional[RegisterResult]:
        if not self.enabled or not self.recv_socket:
            return None
        
        try:
            if self.recv_socket.poll(timeout_ms):
                message = self.recv_socket.recv_json(flags=zmq.NOBLOCK)
                
                result = RegisterResult(
                    face_id=int(message.get("face_id", -1)),
                    person_id=str(message.get("person_id", "")),
                    person_name=str(message.get("person_name", "")),
                    success=bool(message.get("success", False))
                )
                
                if result.success:
                    logger.info(f"[REGISTER] Confirmación: {result.person_name} registrado correctamente")
                else:
                    logger.warning(f"[REGISTER] Error en registro: {message.get('error', 'Unknown')}")
                
                return result
                
        except zmq.Again:
            pass
        except Exception as e:
            logger.error(f"[REGISTER] Error recibiendo confirmación: {e}")
        
        return None
    
    def close(self):
        logger.info("Cerrando RegisterClient...")
        self._connected = False
        
        if self._monitor_socket:
            try:
                self._monitor_socket.close()
            except:
                pass
            self._monitor_socket = None
        
        if self.send_socket:
            try:
                self.send_socket.setsockopt(zmq.LINGER, 0)
                self.send_socket.close()
            except:
                pass
            self.send_socket = None
        
        if self.recv_socket:
            try:
                self.recv_socket.setsockopt(zmq.LINGER, 0)
                self.recv_socket.close()
            except:
                pass
            self.recv_socket = None
        
        if self.context:
            try:
                self.context.term()
            except:
                pass
            self.context = None
        
        self.enabled = False
        logger.info("RegisterClient cerrado")
    
    @property
    def is_connected(self) -> bool:
        self.check_connection()
        return self._connected
    
    @property
    def is_enabled(self) -> bool:
        return self.enabled
