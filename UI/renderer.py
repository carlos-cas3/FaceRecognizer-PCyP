import cv2
from typing import List, Tuple, Dict, Any

from UI.components import HeaderRenderer, FaceRenderer, OverlayRenderer

# Orquestador principal de renderizado. Delegar el dibujado a componentes especializados.

class UIRenderer:
    
    def __init__(self):
        self.window_name = 'FaceRecognizer'
        self._setup_window()
    
    def _setup_window(self):
        """Inicializa la ventana de OpenCV"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
    
    def draw_preview_from_context(self, context):
        """
        Dibuja preview usando un RenderContext unificado.
        Método simplificado que delega al método original.
        """
        self.draw_preview(
            frame=context.frame,
            faces=context.faces,
            mode=context.mode,
            register_state=context.register_state,
            selected_face_ids=context.selected_face_ids,
            registered_faces=context.registered_faces,
            locked_faces=context.locked_faces,
            current_face_index=context.current_face_index,
            current_name=context.current_name,
            zmq_enabled=context.zmq_enabled,
            zmq_connected=context.zmq_connected
        )
    
    def draw_preview(
        self,
        frame,
        faces: List[Tuple[int, Any, Tuple[int, int, int, int]]],
        mode: str,
        register_state: str,
        selected_face_ids: List[int],
        registered_faces: Dict[int, str],
        locked_faces: Dict[int, Dict],
        current_face_index: int = 0,
        current_name: str = "",
        zmq_enabled: bool = False,
        zmq_connected: bool = False
    ):
        """
        Dibuja el frame completo delegando a componentes especializados.
        """
        frame_display = frame.copy()
        h, w = frame_display.shape[:2]
        
        # 1. Header (modo y conexión)
        HeaderRenderer.draw(frame_display, mode, zmq_enabled, zmq_connected)
        
        # 2. Información de caras
        OverlayRenderer.draw_face_info(
            frame_display,
            faces,
            mode,
            register_state,
            selected_face_ids,
            locked_faces
        )
        
        # 3. Dibujar caras detectadas
        FaceRenderer.draw(
            frame_display,
            faces,
            mode,
            registered_faces,
            selected_face_ids,
            locked_faces
        )
        
        # 4. Panel de registro (si aplica)
        if mode == "register" and register_state == "selecting":
            OverlayRenderer.draw_register_panel(
                frame_display,
                w,
                selected_face_ids,
                current_face_index,
                current_name
            )
        
        # 5. Instrucciones en footer
        OverlayRenderer.draw_instructions(frame_display, h, mode, register_state)
        
        # 6. Mostrar frame
        cv2.imshow(self.window_name, frame_display)
    
    def cleanup(self):
        """Limpia recursos de UI"""
        cv2.destroyAllWindows()