import logging

logger = logging.getLogger(__name__)

# Procesa frames y caras según el modo de operación, encapsulando la lógica de detección y tracking.

class FrameProcessor:
    def __init__(self, tracker, detector, face_manager, frame_manager):
        self.tracker = tracker
        self.detector = detector
        self.face_manager = face_manager
        self.frame_manager = frame_manager
    
    def process(self, frame, app_state):
        # 1. Detectar caras con tracker
        raw_faces = self.tracker.process(frame, self.detector)
        
        # 2. Procesar según modo
        if app_state.mode == "register":
            return self._process_register_mode(frame, raw_faces, app_state)
        else:
            return self._process_recognize_mode(frame, raw_faces)
    
    def _process_register_mode(self, frame, raw_faces, app_state):
        """Procesa en modo registro con IDs persistentes"""
        # Aplicar lógica de IDs persistentes
        faces = self.face_manager.process_faces_for_register_mode(raw_faces)
        
        # Gestionar pausa según sub-estado
        if app_state.register_state == "selecting":
            self.frame_manager.pause(frame, faces)
            return frame, self.frame_manager.paused_faces
        elif app_state.register_state == "idle":
            self.frame_manager.pause(frame, faces)
            return frame, faces
        
        return frame, faces
    
    def _process_recognize_mode(self, frame, raw_faces):
        """Procesa en modo reconocimiento"""
        # Limpiar estado de registro
        self.face_manager.clear_locked_faces()
        self.frame_manager.clear()
        
        return frame, raw_faces