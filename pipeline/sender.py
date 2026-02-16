import zmq
import cv2
import struct
import time
import json
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ZMQSender:
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        
        logger.info(f"Conectando a ZMQ endpoint: {endpoint}")
        self.socket.connect(endpoint)
        logger.info("Conexión ZMQ establecida")
    
    def send_face(
        self,
        camera_id: str,
        face_track_id: int,
        face_img,
        bbox: Tuple[int, int, int, int],
        mode: str = "recognize",
        frame_number: int = 0,
        camera_name: str = "",
        person_name: Optional[str] = None,
        person_id: Optional[str] = None,
        detection_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:

        try:
            face_resized = cv2.resize(face_img, (112, 112))

            success, buffer = cv2.imencode(
                '.jpg', 
                face_resized,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            
            if not success:
                logger.error("Error codificando imagen a JPEG")
                return False
            
            img_bytes = buffer.tobytes()
            
            header = {
                "camera_id": camera_id,
                "face_track_id": face_track_id,
                "frame_number": frame_number,
                "timestamp": int(time.time()),
                "mode": mode,
                "camera_name": camera_name,
                "bbox": {
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "w": int(bbox[2]),
                    "h": int(bbox[3])
                },
                "detection_index": detection_index,
                "image_size": len(img_bytes),
            }
            
            if person_name:
                header["person_name"] = person_name
            
            if person_id:
                header["person_id"] = person_id
            
            if metadata:
                header.update(metadata)
            
            header_json = json.dumps(header).encode('utf-8')
            
            header_len = struct.pack('!I', len(header_json))
            message = header_len + header_json + img_bytes
            
            self.socket.send(message, flags=zmq.NOBLOCK)
            
            logger.debug(
                f"Enviado: {camera_id} | mode={mode} | "
                f"face_track_id={face_track_id} | size={len(img_bytes)} bytes"
            )
            return True
            
        except zmq.Again:
            logger.warning("ZMQ socket buffer lleno, mensaje descartado")
            return False
        except Exception as e:
            logger.error(f"Error enviando rostro: {e}")
            return False
    
    def send_control_message(
        self,
        camera_id: str,
        command: str,
        params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Envía un mensaje de control (no incluye imagen)"""
        try:
            header = {
                "camera_id": camera_id,
                "command": command,
                "timestamp": int(time.time()),
                "control_message": True
            }
            
            if params:
                header.update(params)
            
            header_json = json.dumps(header).encode('utf-8')
            header_len = struct.pack('!I', len(header_json))
            
            self.socket.send(header_len + header_json)
            
            logger.debug(f"Control enviado: {command}")
            return True
            
        except Exception as e:
            logger.error(f"Error enviando control: {e}")
            return False
    
    def close(self):
        logger.info("Cerrando conexión ZMQ")
        self.socket.close()
        self.context.term()
