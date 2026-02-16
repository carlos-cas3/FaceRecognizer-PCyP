import cv2

# Renderiza el header con modo y estado de conexión

class HeaderRenderer:
    
    @staticmethod
    def draw(frame, mode: str, zmq_enabled: bool, zmq_connected: bool):
        # Modo actual
        mode_label = "REGISTRO" if mode == "register" else "RECONOCIMIENTO"
        mode_color = (0, 255, 0) if mode == "register" else (0, 255, 255)
        
        cv2.putText(
            frame,
            f"Modo: {mode_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            mode_color,
            2
        )
        
        # Estado de conexión ZMQ
        if zmq_enabled:
            status = "C++ CONECTADO" if zmq_connected else "C++ DESCONECTADO"
            status_color = (0, 255, 0) if zmq_connected else (0, 0, 255)
            cv2.putText(
                frame,
                status,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2
            )