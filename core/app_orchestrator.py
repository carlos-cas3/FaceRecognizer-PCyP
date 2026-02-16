import time
import logging
from typing import Optional
import cv2

from core.render_context import RenderContext
from core.metrics_manager import MetricsManager
from core.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)

# Orquesta el flujo principal de la aplicación, delegando responsabilidades específicas a componentes especializados.

class ApplicationOrchestrator:

    def __init__(
        self,
        camera,
        detector,
        tracker,
        face_manager,
        frame_manager,
        renderer,
        input_handler,
        zmq_client=None
    ):
        self.camera = camera
        self.renderer = renderer
        self.input_handler = input_handler
        self.zmq_client = zmq_client
        
        # Sub-componentes especializados
        self.frame_processor = FrameProcessor(
            tracker=tracker,
            detector=detector,
            face_manager=face_manager,
            frame_manager=frame_manager
        )
        self.metrics = MetricsManager(log_interval=30)
        
        # Referencias directas para operaciones específicas
        self.face_manager = face_manager
        self.frame_manager = frame_manager
        
        self.state = None
        self.running = False
    
    def start(self, state):
        """Inicia el loop principal de la aplicación"""
        self.state = state
        self.running = True
        self.metrics.start()
        
        logger.info("Loop principal iniciado")
        
        while self.running:
            if not self._process_frame():
                break
            
            self.metrics.increment_frame()
    
    def stop(self):
        """Detiene el loop principal"""
        self.running = False
        logger.info(f"Loop detenido - FPS promedio: {self.metrics.get_fps():.1f}")
    
    def _process_frame(self) -> bool:
        """
        Procesa un frame completo.
        
        Returns:
            False si debe salir, True si continúa
        """
        # 1. Capturar frame
        live_frame = self.camera.read()
        if live_frame is None:
            logger.warning("No se pudo capturar frame")
            time.sleep(0.01)
            return True
        
        # 2. Gestionar pausa de frame
        frame = self._handle_frame_pause(live_frame)
        
        # 3. Procesar detección y tracking
        processed_frame, faces = self.frame_processor.process(frame, self.state)
        
        # 4. Renderizar interfaz
        self._render_ui(processed_frame, faces)
        
        # 5. Procesar input de usuario
        return self._process_user_input(faces)
    
    def _handle_frame_pause(self, live_frame):
        """Maneja la lógica de pausa de frames"""
        if self.state.mode == "register" and self.state.register_state == "selecting":
            self.frame_manager.pause(live_frame, [])
            return self.frame_manager.get_frame(live_frame)
        else:
            self.frame_manager.resume()
            return live_frame
    
    def _render_ui(self, frame, faces):
        """Renderiza la interfaz usando contexto unificado"""
        context = RenderContext.from_state(
            frame=frame,
            faces=faces,
            app_state=self.state,
            face_manager=self.face_manager,
            zmq_client=self.zmq_client
        )
        
        # Delegar al renderer con contexto simple
        self.renderer.draw_preview_from_context(context)
    
    def _process_user_input(self, faces) -> bool:
        """Procesa input del usuario y ejecuta acciones"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == 255:  # No hay tecla presionada
            return True
        
        # Procesar tecla
        self.state = self.input_handler.handle_key(
            key=key,
            state=self.state,
            faces=faces,
            face_manager=self.face_manager
        )
        
        # Verificar salida
        if self.state.should_exit:
            return False
        
        # Enviar a servidor si es necesario
        if self.state.should_send_to_cpp and self.zmq_client:
            self._send_face_to_server()
        
        return True
    
    def _send_face_to_server(self):
        """Envía una cara al servidor para registro/reconocimiento"""
        face_id, bbox, name = self.state.face_to_send
        
        success = self.zmq_client.send_face(
            frame=self.frame_manager.paused_frame,
            face_id=face_id,
            bbox=bbox,
            mode=self.state.mode,
            person_name=name
        )
        
        if success:
            self.face_manager.register_face(face_id, name)
            logger.info(f"Cara {face_id} enviada al servidor como '{name}'")