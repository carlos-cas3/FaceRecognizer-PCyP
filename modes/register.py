import cv2
from pipeline.yolo_tracker import FaceTracker
from pipeline.sender import ZMQSender
from pipeline.camera import Camera

def run_register():
    # Solicitar nombre
    name = input("[REGISTER] Ingrese nombre de la persona: ").strip()
    if not name:
        print("Nombre vacío. Abortando.")
        return

    camera = Camera(index=0)
    tracker = FaceTracker("models/yolov8n-face-lindevs.pt", interval=1.0)
    sender = ZMQSender("tcp://192.168.18.4:5555")

    print(f"[REGISTER] Registrando a: {name}")
    print("[REGISTER] Presiona S para capturar y registrar")
    print("[REGISTER] Presiona R para reiniciar")
    print("[REGISTER] Presiona Q para salir")

    registered_count = 0
    max_samples = 5  # Número de muestras a capturar

    try:
        while registered_count < max_samples:
            frame = camera.read()
            if frame is None:
                print("Error leyendo frame")
                break

            faces = tracker.process(frame)

            # Dibujar rectángulos
            for face_id, face, (x, y, w, h) in faces:
                color = (0, 255, 0)  # Verde cuando está listo
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"ID: {face_id}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Mostrar contador
            cv2.putText(frame, f"Muestras: {registered_count}/{max_samples}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

            cv2.imshow("Register Mode", frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and len(faces) > 0:
                # Registrar la primera cara detectada
                face_id, face, bbox = faces[0]
                sender.send_face(face_id, face, mode="register")
                print(f"[REGISTER] Muestra {registered_count+1} enviada")
                registered_count += 1
                
            elif key == ord('r'):
                # Reiniciar
                registered_count = 0
                tracker.reset()
                print("[REGISTER] Contador reiniciado")
                
            elif key == ord('q'):
                print("[REGISTER] Cancelado por usuario")
                break

        if registered_count >= max_samples:
            print(f"[REGISTER] ✓ Registro completo para {name}")

    finally:
        camera.release()
        sender.close()
        cv2.destroyAllWindows()