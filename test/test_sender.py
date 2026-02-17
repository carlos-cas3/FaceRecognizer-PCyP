import zmq
import cv2
import struct
import numpy as np


def send_test_face(mode: str = "recognize", face_id: int = 1, name: str = ""):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://192.168.18.4:5555")

    # Cara de prueba con un patrón visible (no negra)
    dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
    cv2.circle(dummy_face, (56, 56), 30, (255, 255, 255), -1)  # círculo blanco
    cv2.circle(dummy_face, (40, 45), 8, (0, 0, 0), -1)  # ojo izq
    cv2.circle(dummy_face, (72, 45), 8, (0, 0, 0), -1)  # ojo der
    cv2.ellipse(dummy_face, (56, 70), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # boca

    # Guardar localmente para verificar que se construyó bien
    cv2.imwrite("test_face_sent.jpg", dummy_face)
    print(f"[TEST] Imagen guardada como test_face_sent.jpg")

    _, encoded = cv2.imencode(".jpg", dummy_face)
    img_bytes = encoded.tobytes()

    mode_byte = b"\x01" if mode == "register" else b"\x00"
    name_bytes = name.encode("utf-8")

    message = (
        mode_byte
        + struct.pack(">I", face_id)
        + struct.pack(">I", len(name_bytes))
        + name_bytes
        + struct.pack(">I", len(img_bytes))
        + img_bytes
    )

    socket.send(message)
    print(
        f"[TEST] Enviado: mode={mode}, faceId={face_id}, "
        f"name='{name}', img_size={len(img_bytes)} bytes"
    )

    socket.close()
    context.term()


if __name__ == "__main__":
    send_test_face(mode="recognize", face_id=1, name="")
    send_test_face(mode="register", face_id=2, name="Carlos")

