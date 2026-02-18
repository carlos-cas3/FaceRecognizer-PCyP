import time
import logging
from typing import Optional, List, Tuple, Any, Dict
import cv2

from core.render_context import RenderContext
from core.metrics_manager import MetricsManager
from core.frame_processor import FrameProcessor
from core.register_manager import RegisterManager
from core.recognition_manager import RecognitionManager
from communication.register_client import RegisterClient
from communication.recognition_client import RecognitionClient

logger = logging.getLogger(__name__)

class ApplicationOrchestrator:
    def __init__(
        self,
        camera,
        detector,
        tracker,
        frame_manager,
        renderer,
        input_handler,
        register_client: Optional[RegisterClient] = None,
        recognition_client: Optional[RecognitionClient] = None,
        register_config: dict = None,
        recognition_config: dict = None
    ):
        self.camera = camera
        self.renderer = renderer
        self.input_handler = input_handler
        self.frame_manager = frame_manager
        
        self.register_client = register_client
        self.recognition_client = recognition_client
        
        register_config = register_config or {}
        recognition_config = recognition_config or {}
        
        self.register_manager = RegisterManager(
            id_timeout=register_config.get('id_timeout', 5.0),
            match_threshold=register_config.get('match_threshold', 50)
        )
        self.recognition_manager = RecognitionManager(
            recognition_timeout=recognition_config.get('result_timeout', 10.0),
            send_interval=recognition_config.get('interval', 1.0),
            confidence_threshold=recognition_config.get('confidence_threshold', 0.7),
            position_match_threshold=recognition_config.get('position_match_threshold', 50),
            position_cache_timeout=recognition_config.get('position_cache_timeout', 10.0)
        )
        
        self.pending_bboxes: Dict[int, Tuple[int, int, int, int]] = {}
        
        self.frame_processor = FrameProcessor(tracker, detector)
        self.metrics = MetricsManager(log_interval=30)
        
        self.state = None
        self.running = False
    
    def start(self, state):
        self.state = state
        self.running = True
        self.metrics.start()
        logger.info("Loop principal iniciado")
        
        while self.running:
            if not self._process_frame():
                break
            self.metrics.increment_frame()
    
    def stop(self):
        self.running = False
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        logger.info(f"Loop detenido - FPS promedio: {self.metrics.get_fps():.1f}")
    
    def _process_frame(self) -> bool:
        live_frame = self.camera.read()
        if live_frame is None:
            logger.warning("No se pudo capturar frame")
            time.sleep(0.01)
            return True
        
        faces = self.frame_processor.detect_faces(live_frame)
        
        if self.state.mode == "register":
            return self._handle_register_mode(live_frame, faces)
        else:
            return self._handle_recognition_mode(live_frame, faces)
    
    def _handle_register_mode(self, frame, faces) -> bool:
        faces = self.register_manager.process_faces(faces)
        
        display_frame = frame
        display_faces = faces
        
        if self.state.register_state == "selecting":
            self.frame_manager.pause(frame, faces)
            display_frame = self.frame_manager.paused_frame
            display_faces = self.frame_manager.paused_faces
        else:
            self.frame_manager.pause(frame, faces)
        
        self._render_ui(display_frame, display_faces)
        
        return self._process_input_register(display_frame, faces)
    
    def _handle_recognition_mode(self, frame, faces) -> bool:
        self.frame_manager.resume()
        self.register_manager.clear_all()
        
        active_face_ids = [face_id for face_id, _, _ in faces]
        
        self.recognition_manager.refresh_active_faces(active_face_ids)
        
        for face_id, _, bbox in faces:
            if not self.recognition_manager.is_recognized(face_id):
                if self.recognition_manager.assign_identity_from_cache(face_id, bbox):
                    pass
        
        self.recognition_manager.cleanup_not_visible(active_face_ids)
        
        self._send_for_recognition(frame, faces)
        self._receive_recognition_results()
        
        self._render_ui(frame, faces)
        
        return self._process_input_recognition()
    
    def _send_for_recognition(self, frame, faces):
        if not self.recognition_client or not self.recognition_client.is_connected:
            return
        
        for face_id, _, bbox in faces:
            if not self.recognition_manager.is_recognized(face_id):
                if self.recognition_manager.should_send(face_id):
                    success = self.recognition_client.send_recognition_request(
                        frame=frame,
                        face_id=face_id,
                        bbox=bbox
                    )
                    if success:
                        self.recognition_manager.mark_sent(face_id)
                        self.pending_bboxes[face_id] = bbox
                        logger.debug(f"Cara {face_id} enviada para reconocimiento")
    
    def _receive_recognition_results(self):
        if not self.recognition_client or not self.recognition_client.is_connected:
            return
        
        result = self.recognition_client.receive_result()
        if result:
            bbox = self.pending_bboxes.pop(result.face_id, None)
            self.recognition_manager.update_identity(
                face_id=result.face_id,
                person_id=result.person_id,
                person_name=result.person_name,
                confidence=result.confidence,
                bbox=bbox
            )
    
    def _render_ui(self, frame, faces):
        context = RenderContext.from_state(
            frame=frame,
            faces=faces,
            app_state=self.state,
            register_manager=self.register_manager,
            recognition_manager=self.recognition_manager,
            register_client=self.register_client,
            recognition_client=self.recognition_client
        )
        self.renderer.draw_preview_from_context(context)
    
    def _process_input_register(self, frame, faces) -> bool:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 255:
            return True
        
        self.state = self.input_handler.handle_key(
            key=key,
            state=self.state,
            faces=faces,
            register_manager=self.register_manager
        )
        
        if self.state.should_exit:
            return False
        
        if self.state.should_send_to_cpp:
            self._send_register_request(frame)
        
        return True
    
    def _process_input_recognition(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 255:
            return True
        
        self.state = self.input_handler.handle_key(
            key=key,
            state=self.state,
            faces=[],
            register_manager=self.register_manager
        )
        
        if self.state.should_exit:
            return False
        
        return True
    
    def _send_register_request(self, frame):
        if not self.register_client or not self.register_client.is_connected:
            logger.warning("RegisterClient no disponible")
            return
        
        face_id, bbox, person_name = self.state.face_to_send
        
        paused_frame = self.frame_manager.paused_frame
        if paused_frame is None:
            paused_frame = frame
        
        success = self.register_client.send_register_request(
            frame=paused_frame,
            face_id=face_id,
            bbox=bbox,
            person_name=person_name
        )
        
        if success:
            logger.info(f"Registro enviado: Face {face_id} = '{person_name}'")
