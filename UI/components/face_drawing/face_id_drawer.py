import cv2
from typing import Tuple

class FaceIDDrawer:
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2
    BG_COLOR = (0, 0, 0)
    COLOR_LOCKED = (0, 255, 255)
    COLOR_TEMPORARY = (255, 255, 0)
    COLOR_RECOGNIZED = (0, 255, 0)
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
        
        (id_w, id_h), _ = cv2.getTextSize(
            id_text,
            FaceIDDrawer.FONT,
            FaceIDDrawer.FONT_SCALE,
            FaceIDDrawer.FONT_THICKNESS
        )
        
        id_x = center_x - id_w // 2
        id_y = top_y - FaceIDDrawer.OFFSET_Y
        
        cv2.rectangle(
            frame,
            (id_x - FaceIDDrawer.PADDING, id_y - id_h - FaceIDDrawer.PADDING),
            (id_x + id_w + FaceIDDrawer.PADDING, id_y + FaceIDDrawer.PADDING),
            FaceIDDrawer.BG_COLOR,
            -1
        )
        
        cv2.putText(
            frame,
            id_text,
            (id_x, id_y),
            FaceIDDrawer.FONT,
            FaceIDDrawer.FONT_SCALE,
            id_color,
            FaceIDDrawer.FONT_THICKNESS
        )
    
    @staticmethod
    def draw_recognition_info(
        frame,
        center_x: int,
        top_y: int,
        person_name: str,
        confidence: float
    ):
        confidence_pct = int(confidence * 100)
        name_text = f"{person_name}"
        conf_text = f"{confidence_pct}%"
        
        font_scale = 0.6
        thickness = 2
        
        (name_w, name_h), _ = cv2.getTextSize(
            name_text, FaceIDDrawer.FONT, font_scale, thickness
        )
        (conf_w, conf_h), _ = cv2.getTextSize(
            conf_text, FaceIDDrawer.FONT, font_scale * 0.8, thickness - 1
        )
        
        name_x = center_x - name_w // 2
        name_y = top_y - FaceIDDrawer.OFFSET_Y - 25
        
        cv2.rectangle(
            frame,
            (name_x - FaceIDDrawer.PADDING, name_y - name_h - FaceIDDrawer.PADDING),
            (name_x + name_w + FaceIDDrawer.PADDING, name_y + FaceIDDrawer.PADDING),
            FaceIDDrawer.BG_COLOR,
            -1
        )
        
        cv2.putText(
            frame,
            name_text,
            (name_x, name_y),
            FaceIDDrawer.FONT,
            font_scale,
            FaceIDDrawer.COLOR_RECOGNIZED,
            thickness
        )
        
        conf_x = center_x - conf_w // 2
        conf_y = name_y + name_h + 15
        
        cv2.rectangle(
            frame,
            (conf_x - 3, conf_y - conf_h - 3),
            (conf_x + conf_w + 3, conf_y + 3),
            FaceIDDrawer.BG_COLOR,
            -1
        )
        
        cv2.putText(
            frame,
            conf_text,
            (conf_x, conf_y),
            FaceIDDrawer.FONT,
            font_scale * 0.7,
            (255, 255, 255),
            thickness - 1
        )