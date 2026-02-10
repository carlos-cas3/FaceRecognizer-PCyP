import cv2
from pipeline.yolo_tracker import FaceTracker
from pipeline.sender import ZMQSender
from pipeline.camera import Camera

def run_recognize():
    camera = Camera(index=0)
    tracker = FaceTracker("models/yolov8n-face-lindevs.pt", interval=0.5)
    sender = ZMQSender("tcp://192.168.18.4:5555")

    print("[RECOGNIZE] Modo reconocimiento activo")
    print("[RECOGNIZE] Presiona Q para salir")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Error leyendo frame")
                break

            faces = tracker.process(frame)

            for face_id, face, (x, y, w, h) in faces:
                # Enviar rostro para reconocimiento
                sender.send_face(face_id, face, mode="recognize")
                
                # Dibujar rectángulo amarillo
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, f"ID: {face_id}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow("Recognize Mode", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.release()
        sender.close()
        cv2.destroyAllWindows()