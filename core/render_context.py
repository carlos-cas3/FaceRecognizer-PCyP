from dataclasses import dataclass
from typing import Dict, List, Optional

# Contexto unificado para renderizado, que encapsula toda la información necesaria para dibujar la UI en cada frame.

@dataclass
class RenderContext:
    frame: any  # numpy array
    faces: List[tuple]
    mode: str
    register_state: str
    selected_face_ids: List[int]
    registered_faces: Dict[int, str]
    locked_faces: Dict[int, Dict]
    current_face_index: int
    current_name: str
    zmq_enabled: bool
    zmq_connected: bool
    
    @classmethod
    def from_state(
        cls,
        frame,
        faces,
        app_state,
        face_manager,
        zmq_client
    ):
        return cls(
            frame=frame,
            faces=faces,
            mode=app_state.mode,
            register_state=app_state.register_state,
            selected_face_ids=app_state.selected_face_ids,
            registered_faces=face_manager.registered_faces,
            locked_faces=face_manager.locked_faces,
            current_face_index=app_state.current_face_index,
            current_name=app_state.current_name,
            zmq_enabled=zmq_client is not None,
            zmq_connected=zmq_client.is_connected if zmq_client else False
        )