import cv2
from typing import List, Tuple, Dict, Any, Optional

from UI.components import HeaderRenderer, FaceRenderer, OverlayRenderer

class UIRenderer:
    
    def __init__(self):
        self.window_name = 'FaceRecognizer'
        self._setup_window()
    
    def _setup_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
    
    def draw_preview_from_context(self, context):
        self.draw_preview(
            frame=context.frame,
            faces=context.faces,
            mode=context.mode,
            register_state=context.register_state,
            selected_face_ids=context.selected_face_ids,
            locked_faces=context.locked_faces,
            recognized_identities=context.recognized_identities,
            current_face_index=context.current_face_index,
            current_name=context.current_name,
            zmq_register_enabled=context.zmq_register_enabled,
            zmq_recognition_enabled=context.zmq_recognition_enabled
        )
    
    def draw_preview(
        self,
        frame,
        faces: List[Tuple[int, Any, Tuple[int, int, int, int]]],
        mode: str,
        register_state: str,
        selected_face_ids: List[int],
        locked_faces: Dict[int, Any],
        recognized_identities: Dict[int, Any],
        current_face_index: int = 0,
        current_name: str = "",
        zmq_register_enabled: bool = False,
        zmq_recognition_enabled: bool = False
    ):
        frame_display = frame.copy()
        h, w = frame_display.shape[:2]
        
        HeaderRenderer.draw(frame_display, mode, zmq_register_enabled, zmq_recognition_enabled)
        
        OverlayRenderer.draw_face_info(
            frame_display,
            faces,
            mode,
            register_state,
            selected_face_ids,
            locked_faces
        )
        
        FaceRenderer.draw(
            frame_display,
            faces,
            mode,
            selected_face_ids,
            locked_faces,
            recognized_identities
        )
        
        if mode == "register" and register_state == "selecting":
            OverlayRenderer.draw_register_panel(
                frame_display,
                w,
                selected_face_ids,
                current_face_index,
                current_name
            )
        
        OverlayRenderer.draw_instructions(frame_display, h, mode, register_state)
        
        cv2.imshow(self.window_name, frame_display)
    
    def cleanup(self):
        cv2.destroyAllWindows()
