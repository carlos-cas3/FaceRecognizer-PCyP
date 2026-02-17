from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.register_manager import RegisterManager
    from core.recognition_manager import RecognitionManager
    from communication.register_client import RegisterClient
    from communication.recognition_client import RecognitionClient

@dataclass
class RenderContext:
    frame: Any
    faces: List[tuple]
    mode: str
    register_state: str
    selected_face_ids: List[int]
    locked_faces: Dict[int, Any]
    recognized_identities: Dict[int, Any]
    current_face_index: int
    current_name: str
    zmq_register_enabled: bool
    zmq_recognition_enabled: bool
    
    @classmethod
    def from_state(
        cls,
        frame,
        faces,
        app_state,
        register_manager: "RegisterManager",
        recognition_manager: "RecognitionManager",
        register_client: Optional["RegisterClient"],
        recognition_client: Optional["RecognitionClient"]
    ):
        return cls(
            frame=frame,
            faces=faces,
            mode=app_state.mode,
            register_state=app_state.register_state,
            selected_face_ids=app_state.selected_face_ids,
            locked_faces=register_manager.locked_faces,
            recognized_identities=recognition_manager.identities,
            current_face_index=app_state.current_face_index,
            current_name=app_state.current_name,
            zmq_register_enabled=register_client is not None and register_client.is_connected,
            zmq_recognition_enabled=recognition_client is not None and recognition_client.is_connected
        )
