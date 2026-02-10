import zmq
import cv2
import struct

class ZMQSender:
    def __init__(self, endpoint):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(endpoint)
        print(f"[ZMQ] Conectado a {endpoint}")

    def send_face(self, face_id, face_img, mode="recognize"):
        """
        Envía imagen de rostro al servidor C++
        Formato: [mode(1byte)][face_id(4bytes)][img_size(4bytes)][img_data]
        """
        # Redimensionar rostro a 112x112 (estándar ArcFace)
        face_resized = cv2.resize(face_img, (112, 112))
        
        # Codificar a JPEG
        success, buffer = cv2.imencode('.jpg', face_resized, 
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            print("[ZMQ] Error encoding image")
            return
        
        img_bytes = buffer.tobytes()
        
        # Preparar header
        mode_byte = 1 if mode == "register" else 0
        
        # Empaquetar: mode(1) + face_id(4) + img_size(4)
        header = struct.pack('!BII', mode_byte, face_id, len(img_bytes))
        
        # Enviar mensaje completo
        message = header + img_bytes
        self.socket.send(message)
        
        print(f"[ZMQ] Sent: mode={mode}, id={face_id}, size={len(img_bytes)} bytes")

    def close(self):
        self.socket.close()
        self.context.term()