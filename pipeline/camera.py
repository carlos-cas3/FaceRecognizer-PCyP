import cv2

class Camera:
    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            raise Exception(f"No se pudo abrir la cámara {index}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()