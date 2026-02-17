import cv2
from typing import Tuple, Optional, Any

class LabelDrawer:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 255, 255)
    PADDING = 10
    
    @staticmethod
    def create_label(
        index: int,
        face_id: int,
        is_selected: bool,
        mode: str = "register",
        recognized_identity: Optional[Any] = None
    ) -> str:
        if mode == "recognize" and recognized_identity:
            confidence_pct = int(recognized_identity.confidence * 100)
            return f"{recognized_identity.person_name} ({confidence_pct}%)"
        elif mode == "recognize":
            return f"ID:{face_id} Identificando..."
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
        (label_w, label_h), _ = cv2.getTextSize(
            label,
            LabelDrawer.FONT,
            LabelDrawer.FONT_SCALE,
            LabelDrawer.FONT_THICKNESS
        )
        
        cv2.rectangle(
            frame,
            (x, y - label_h - LabelDrawer.PADDING),
            (x + label_w, y),
            bg_color,
            -1
        )
        
        cv2.putText(
            frame,
            label,
            (x, y - 5),
            LabelDrawer.FONT,
            LabelDrawer.FONT_SCALE,
            LabelDrawer.TEXT_COLOR,
            LabelDrawer.FONT_THICKNESS
        )
