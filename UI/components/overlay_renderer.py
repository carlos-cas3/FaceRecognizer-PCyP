from typing import List, Dict, Any, Tuple

from UI.components.overlay.face_info_display import FaceInfoDisplay
from UI.components.overlay.register_panel_display import RegisterPanelDisplay
from UI.components.overlay.instructions_display import InstructionsDisplay

class OverlayRenderer:
    """
    Orquestador de overlays en pantalla.
    Delega el renderizado a componentes especializados.
    """
    
    @staticmethod
    def draw_face_info(
        frame,
        faces: List[Tuple[int, Any, Tuple[int, int, int, int]]],
        mode: str,
        register_state: str,
        selected_face_ids: List[int],
        locked_faces: Dict[int, Dict]
    ):
        """Dibuja información de caras detectadas"""
        FaceInfoDisplay.draw(
            frame, faces, mode, register_state,
            selected_face_ids, locked_faces
        )
    
    @staticmethod
    def draw_register_panel(
        frame,
        width: int,
        selected_face_ids: List[int],
        current_face_index: int,
        current_name: str
    ):
        """Dibuja el panel de registro"""
        RegisterPanelDisplay.draw(
            frame, width, selected_face_ids,
            current_face_index, current_name
        )
    
    @staticmethod
    def draw_instructions(
        frame,
        height: int,
        mode: str,
        register_state: str
    ):
        """Dibuja instrucciones en el footer"""
        InstructionsDisplay.draw(frame, height, mode, register_state)