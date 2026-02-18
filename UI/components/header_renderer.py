import cv2

class HeaderRenderer:
    
    COLOR_REGISTER = (0, 255, 0)
    COLOR_RECOGNIZE = (0, 255, 255)
    COLOR_CONNECTED = (0, 255, 0)
    COLOR_DISCONNECTED = (0, 0, 255)
    COLOR_CONNECTING = (0, 165, 255)
    
    @staticmethod
    def draw(frame, mode: str, zmq_register_enabled: bool, zmq_recognition_enabled: bool):
        mode_label = "REGISTRO" if mode == "register" else "RECONOCIMIENTO"
        mode_color = HeaderRenderer.COLOR_REGISTER if mode == "register" else HeaderRenderer.COLOR_RECOGNIZE
        
        cv2.putText(
            frame,
            f"Modo: {mode_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            mode_color,
            2
        )
        
        if mode == "register":
            if zmq_register_enabled:
                status = "C++ CONECTADO"
                status_color = HeaderRenderer.COLOR_CONNECTED
            else:
                status = "C++ DESCONECTADO"
                status_color = HeaderRenderer.COLOR_DISCONNECTED
        else:
            if zmq_recognition_enabled:
                status = "C++ CONECTADO"
                status_color = HeaderRenderer.COLOR_CONNECTED
            else:
                status = "C++ DESCONECTADO"
                status_color = HeaderRenderer.COLOR_DISCONNECTED
        
        cv2.putText(
            frame,
            status,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2
        )
