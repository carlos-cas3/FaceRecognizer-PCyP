import cv2
import logging

logger = logging.getLogger(__name__)


class Camera:

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        self.width = width
        self.height = height

        self.cap = self._open_camera(index)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        logger.info(f"Cámara {index} inicializada correctamente")

    def _open_camera(self, index):
        backends = [
            cv2.CAP_DSHOW,   # DirectShow
            cv2.CAP_MSMF,    # Media Foundation
            None             # Default OpenCV
        ]

        for backend in backends:
            try:
                if backend is None:
                    cap = cv2.VideoCapture(index)
                    backend_name = "DEFAULT"
                else:
                    cap = cv2.VideoCapture(index, backend)
                    backend_name = str(backend)

                if cap and cap.isOpened():
                    logger.info(f"Cámara {index} abierta con backend {backend_name}")
                    return cap
                else:
                    if cap:
                        cap.release()

            except Exception:
                continue

        return None

    def read(self):
        if not self.cap:
            logger.warning(f"Cámara {self.index} no inicializada")
            return None
        ret, frame = self.cap.read()
        if not ret:
            logger.warning(f"Fallo al leer frame de cámara {self.index}")
            return None
        return frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info(f"Cámara {self.index} liberada")
