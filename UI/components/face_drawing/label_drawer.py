import cv2
from typing import Tuple, Dict

# Componente especializado en dibujar labels con fondo coloreado para cada cara detectada.

class LabelDrawer:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 255, 255)  # Blanco
    PADDING = 10
    
    @staticmethod
    def create_label(
        index: int,
        face_id: int,
        is_registered: bool,
        is_selected: bool,
        registered_faces: Dict[int, str]
    ) -> str:

        if is_registered:
            return f"[{index}] {registered_faces[face_id]}"
        elif is_selected:
            return f"[{index}] ID:{face_id} [OK]"
        else:
            return f"[{index}] ID:{face_id}"
    
    @staticmethod
    def draw(
        frame,
        label: str,
        x: int,
        y: int,
        bg_color: Tuple[int, int, int]
    ):
        # Calcular tamaño del texto
        (label_w, label_h), _ = cv2.getTextSize(
            label,
            LabelDrawer.FONT,
            LabelDrawer.FONT_SCALE,
            LabelDrawer.FONT_THICKNESS
        )
        
        # Dibujar fondo
        cv2.rectangle(
            frame,
            (x, y - label_h - LabelDrawer.PADDING),
            (x + label_w, y),
            bg_color,
            -1  # Relleno sólido
        )
        
        # Dibujar texto
        cv2.putText(
            frame,
            label,
            (x, y - 5),
            LabelDrawer.FONT,
            LabelDrawer.FONT_SCALE,
            LabelDrawer.TEXT_COLOR,
            LabelDrawer.FONT_THICKNESS
        )