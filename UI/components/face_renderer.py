from typing import List, Tuple, Dict, Any, Optional

from UI.components.face_drawing import BoundingBoxDrawer, LabelDrawer, FaceIDDrawer

class FaceRenderer:
    @staticmethod
    def draw(
        frame,
        faces: List[Tuple[int, Any, Tuple[int, int, int, int]]],
        mode: str,
        selected_face_ids: List[int],
        locked_faces: Dict[int, Any],
        recognized_identities: Optional[Dict[int, Any]] = None
    ):
        for i, (face_id, face_crop, bbox) in enumerate(faces):
            FaceRenderer._draw_single_face(
                frame, i, face_id, bbox,
                mode, selected_face_ids, locked_faces,
                recognized_identities
            )
    
    @staticmethod
    def _draw_single_face(
        frame,
        index: int,
        face_id: int,
        bbox: Tuple[int, int, int, int],
        mode: str,
        selected_face_ids: List[int],
        locked_faces: Dict[int, Any],
        recognized_identities: Optional[Dict[int, Any]] = None
    ):
        x, y, w, h = bbox
        center_x = x + w // 2
        
        is_selected = mode == "register" and face_id in selected_face_ids
        is_locked = face_id in locked_faces
        
        recognized_identity = None
        if recognized_identities and face_id in recognized_identities:
            recognized_identity = recognized_identities[face_id]
        
        BoundingBoxDrawer.draw(
            frame, bbox, mode,
            is_selected, is_locked,
            recognized_identity
        )
        
        label = LabelDrawer.create_label(
            index, face_id,
            is_selected,
            mode, recognized_identity
        )
        
        color = BoundingBoxDrawer.get_color(mode, is_selected, recognized_identity)
        LabelDrawer.draw(frame, label, x, y, color)
        
        if FaceIDDrawer.should_draw(mode):
            FaceIDDrawer.draw(
                frame, face_id, center_x, y,
                is_locked, is_selected
            )
        
        if mode == "recognize" and recognized_identity:
            FaceIDDrawer.draw_recognition_info(
                frame, center_x, y,
                recognized_identity.person_name,
                recognized_identity.confidence
            )
