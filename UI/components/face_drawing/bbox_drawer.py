import cv2
from typing import Tuple, Optional, Any

class BoundingBoxDrawer:
    COLOR_SELECTED = (0, 255, 0)
    COLOR_REGISTER_MODE = (0, 165, 255)
    COLOR_RECOGNIZE_MODE = (255, 0, 0)
    COLOR_HIGHLIGHT = (0, 255, 255)
    COLOR_RECOGNIZED = (0, 200, 0)
    
    @staticmethod
    def get_color(
        mode: str,
        is_selected: bool,
        recognized_identity: Optional[Any] = None
    ) -> Tuple[int, int, int]:
        if is_selected:
            return BoundingBoxDrawer.COLOR_SELECTED
        elif mode == "register":
            return BoundingBoxDrawer.COLOR_REGISTER_MODE
        elif recognized_identity:
            return BoundingBoxDrawer.COLOR_RECOGNIZED
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
        is_selected: bool,
        is_locked: bool,
        recognized_identity: Optional[Any] = None
    ):
        color = BoundingBoxDrawer.get_color(mode, is_selected, recognized_identity)
        BoundingBoxDrawer.draw_primary_box(frame, bbox, color)
        if is_locked or is_selected:
            BoundingBoxDrawer.draw_highlight_box(frame, bbox)
