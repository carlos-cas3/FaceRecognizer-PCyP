import cv2
from typing import Tuple

# Componente para dibujar bounding boxes de caras detectadas, con colores y estilos según el estado

class BoundingBoxDrawer:
    # Colores predefinidos
    COLOR_REGISTERED = (0, 255, 0)      # Verde
    COLOR_SELECTED = (0, 255, 0)        # Verde
    COLOR_REGISTER_MODE = (0, 165, 255) # Naranja
    COLOR_RECOGNIZE_MODE = (255, 0, 0)  # Rojo
    COLOR_HIGHLIGHT = (0, 255, 255)     # Cian
    
    @staticmethod
    def get_color(
        mode: str,
        is_registered: bool,
        is_selected: bool
    ) -> Tuple[int, int, int]:
        if is_registered:
            return BoundingBoxDrawer.COLOR_REGISTERED
        elif is_selected:
            return BoundingBoxDrawer.COLOR_SELECTED
        elif mode == "register":
            return BoundingBoxDrawer.COLOR_REGISTER_MODE
        else:
            return BoundingBoxDrawer.COLOR_RECOGNIZE_MODE
    
    @staticmethod
    def draw_primary_box(
        frame,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        thickness: int = 2
    ):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    @staticmethod
    def draw_highlight_box(
        frame,
        bbox: Tuple[int, int, int, int],
        padding: int = 2
    ):
        x, y, w, h = bbox
        cv2.rectangle(
            frame,
            (x - padding, y - padding),
            (x + w + padding, y + h + padding),
            BoundingBoxDrawer.COLOR_HIGHLIGHT,
            2
        )
    
    @staticmethod
    def draw(
        frame,
        bbox: Tuple[int, int, int, int],
        mode: str,
        is_registered: bool,
        is_selected: bool,
        is_locked: bool
    ):
        color = BoundingBoxDrawer.get_color(mode, is_registered, is_selected)
        # Bounding box principal
        BoundingBoxDrawer.draw_primary_box(frame, bbox, color)
        # Borde de resaltado para caras seleccionadas/bloqueadas
        if is_locked or is_selected:
            BoundingBoxDrawer.draw_highlight_box(frame, bbox)