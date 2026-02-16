import cv2
from typing import Tuple

# Componente especializado en dibujar IDs grandes arriba de las caras en modo registro.

class FaceIDDrawer:
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2
    BG_COLOR = (0, 0, 0)  # Negro
    COLOR_LOCKED = (0, 255, 255)  # Cian
    COLOR_TEMPORARY = (255, 255, 0)  # Amarillo
    PADDING = 5
    OFFSET_Y = 15
    
    @staticmethod
    def should_draw(mode: str) -> bool:
        return mode == "register"
    
    @staticmethod
    def create_id_text(face_id: int, is_locked: bool, is_selected: bool) -> str:
        id_text = f"ID {face_id}"
        if is_locked or is_selected:
            id_text += " [BLOQUEADO]"
        return id_text
    
    @staticmethod
    def get_color(is_locked: bool, is_selected: bool) -> Tuple[int, int, int]:
        if is_locked or is_selected:
            return FaceIDDrawer.COLOR_LOCKED
        else:
            return FaceIDDrawer.COLOR_TEMPORARY
    
    @staticmethod
    def draw(
        frame,
        face_id: int,
        center_x: int,
        top_y: int,
        is_locked: bool,
        is_selected: bool
    ):
        id_text = FaceIDDrawer.create_id_text(face_id, is_locked, is_selected)
        id_color = FaceIDDrawer.get_color(is_locked, is_selected)
        
        # Calcular tamaño del texto
        (id_w, id_h), _ = cv2.getTextSize(
            id_text,
            FaceIDDrawer.FONT,
            FaceIDDrawer.FONT_SCALE,
            FaceIDDrawer.FONT_THICKNESS
        )
        
        # Calcular posición centrada
        id_x = center_x - id_w // 2
        id_y = top_y - FaceIDDrawer.OFFSET_Y
        
        # Dibujar fondo negro
        cv2.rectangle(
            frame,
            (id_x - FaceIDDrawer.PADDING, id_y - id_h - FaceIDDrawer.PADDING),
            (id_x + id_w + FaceIDDrawer.PADDING, id_y + FaceIDDrawer.PADDING),
            FaceIDDrawer.BG_COLOR,
            -1
        )
        
        # Dibujar texto del ID
        cv2.putText(
            frame,
            id_text,
            (id_x, id_y),
            FaceIDDrawer.FONT,
            FaceIDDrawer.FONT_SCALE,
            id_color,
            FaceIDDrawer.FONT_THICKNESS
        )