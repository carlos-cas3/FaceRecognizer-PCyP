import cv2
from typing import List, Dict, Any, Tuple

# Este componente se encarga de mostrar información resumida sobre las caras detectadas

class FaceInfoDisplay:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    POSITION_X = 10
    BASE_Y = 90
    
    COLOR_INFO = (255, 255, 0)      # Amarillo
    COLOR_SELECTED = (0, 255, 0)     # Verde
    COLOR_LOCKED = (0, 255, 255)     # Cian
    
    @staticmethod
    def should_draw(mode: str, register_state: str) -> bool:
        return mode == "register" and register_state == "idle"
    
    @staticmethod
    def draw(
        frame,
        faces: List[Tuple[int, Any, Tuple[int, int, int, int]]],
        mode: str,
        register_state: str,
        selected_face_ids: List[int],
        locked_faces: Dict[int, Dict]
    ):
        if not FaceInfoDisplay.should_draw(mode, register_state):
            return
        
        total_faces = len(faces)
        selected_count = len(selected_face_ids)
        locked_count = len(locked_faces)
        
        # Línea principal: resumen
        FaceInfoDisplay._draw_summary(
            frame, total_faces, selected_count, locked_count
        )
        
        # Línea 2: IDs seleccionados
        if selected_face_ids:
            FaceInfoDisplay._draw_selected_ids(frame, selected_face_ids)
        
        # Línea 3: IDs bloqueados
        if locked_faces:
            FaceInfoDisplay._draw_locked_ids(frame, locked_faces)
    
    @staticmethod
    def _draw_summary(frame, total: int, selected: int, locked: int):
        info_text = f"CARAS: {total} | Sel: {selected} | Bloq: {locked}"
        cv2.putText(
            frame,
            info_text,
            (FaceInfoDisplay.POSITION_X, FaceInfoDisplay.BASE_Y),
            FaceInfoDisplay.FONT,
            0.6,
            FaceInfoDisplay.COLOR_INFO,
            2
        )
    
    @staticmethod
    def _draw_selected_ids(frame, selected_face_ids: List[int]):
        selected_text = f"IDs: {selected_face_ids}"
        cv2.putText(
            frame,
            selected_text,
            (FaceInfoDisplay.POSITION_X, FaceInfoDisplay.BASE_Y + 25),
            FaceInfoDisplay.FONT,
            0.5,
            FaceInfoDisplay.COLOR_SELECTED,
            2
        )
    
    @staticmethod
    def _draw_locked_ids(frame, locked_faces: Dict[int, Dict]):
        locked_text = f"IDs bloqueados: {list(locked_faces.keys())}"
        cv2.putText(
            frame,
            locked_text,
            (FaceInfoDisplay.POSITION_X, FaceInfoDisplay.BASE_Y + 50),
            FaceInfoDisplay.FONT,
            0.4,
            FaceInfoDisplay.COLOR_LOCKED,
            1
        )