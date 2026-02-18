import logging
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.register_manager import RegisterManager

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self):
        self.mode = "recognize"
        self.register_state = "idle"
        self.selected_face_ids: List[int] = []
        self.current_face_index = 0
        self.current_name = ""
        self.should_exit = False
        self.should_send_to_cpp = False
        self.face_to_send: Optional[Tuple[int, Tuple[int, int, int, int], str]] = None
        self.registration_complete = False


class InputHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_key(
        self,
        key: int,
        state: AppState,
        faces: List[Tuple[int, Any, Tuple[int, int, int, int]]],
        register_manager: "RegisterManager"
    ) -> AppState:
        state.should_send_to_cpp = False
        state.face_to_send = None
        
        if state.registration_complete:
            state.registration_complete = False
            register_manager.clear_all()
            self.logger.info("Listo para nuevos registros")
        
        if state.mode == "register" and state.register_state == "selecting":
            return self._handle_register_selecting(key, state, faces, register_manager)
        elif state.mode == "register" and state.register_state == "idle":
            return self._handle_register_idle(key, state, faces, register_manager)
        else:
            return self._handle_recognize_mode(key, state, register_manager)
    
    def _handle_register_selecting(
        self,
        key: int,
        state: AppState,
        faces: List[Tuple[int, Any, Tuple[int, int, int, int]]],
        register_manager: "RegisterManager"
    ) -> AppState:
        if key == 27:
            self.logger.info("Registro cancelado")
            state.register_state = "idle"
            state.selected_face_ids = []
            state.current_face_index = 0
            state.current_name = ""
            register_manager.clear_all()
        
        elif key == ord('3'):
            self.logger.info("Solicitud de salida")
            state.should_exit = True
            state.register_state = "idle"
            state.selected_face_ids = []
            state.current_name = ""
            register_manager.clear_all()
            
        elif key == 13:
            if state.current_name.strip():
                face_id = state.selected_face_ids[state.current_face_index]
                
                face_bbox = None
                for f_id, f_crop, f_bbox in faces:
                    if f_id == face_id:
                        face_bbox = f_bbox
                        break
                
                if face_bbox:
                    state.should_send_to_cpp = True
                    state.face_to_send = (face_id, face_bbox, state.current_name.strip())
                
                self.logger.info(f"Cara {face_id} registrada como: {state.current_name}")
                
                state.current_face_index += 1
                if state.current_face_index >= len(state.selected_face_ids):
                    state.register_state = "idle"
                    state.selected_face_ids = []
                    state.current_face_index = 0
                    state.current_name = ""
                    state.registration_complete = True
                    self.logger.info("Registro completado - Presiona tecla para continuar o '2' para reconocimiento")
                else:
                    state.current_name = ""
        
        elif key == 8:
            state.current_name = state.current_name[:-1]
        
        elif 32 <= key <= 126:
            if key != ord('3'):
                state.current_name += chr(key)
        
        return state
    
    def _handle_register_idle(
        self,
        key: int,
        state: AppState,
        faces: List[Tuple[int, Any, Tuple[int, int, int, int]]],
        register_manager: "RegisterManager"
    ) -> AppState:
        if key == ord('3'):
            self.logger.info("Solicitud de salida")
            state.should_exit = True
        
        elif key == ord('2'):
            state.mode = "recognize"
            state.register_state = "idle"
            state.selected_face_ids = []
            state.current_name = ""
            register_manager.clear_all()
            self.logger.info("Modo cambiado a: RECONOCIMIENTO")
        
        elif ord('0') <= key <= ord('9'):
            idx = key - ord('0')
            if idx < len(faces):
                face_id, face_crop, bbox = faces[idx]
                if face_id not in state.selected_face_ids:
                    state.selected_face_ids.append(face_id)
                    register_manager.lock_face(face_id, bbox)
                    self.logger.info(f"Cara {face_id} seleccionada")
        
        elif key == 13 and state.selected_face_ids:
            state.register_state = "selecting"
            state.current_face_index = 0
            self.logger.info(f"Iniciando registro de {len(state.selected_face_ids)} caras")
        
        return state
    
    def _handle_recognize_mode(
        self, 
        key: int, 
        state: AppState,
        register_manager: "RegisterManager"
    ) -> AppState:
        if key == ord('3'):
            self.logger.info("Solicitud de salida")
            state.should_exit = True
        
        elif key == ord('1'):
            state.mode = "register"
            state.register_state = "idle"
            state.selected_face_ids = []
            state.current_name = ""
            register_manager.clear_all()
            self.logger.info("Modo cambiado a: REGISTRO")
        
        elif key == ord('2'):
            state.mode = "recognize"
            state.register_state = "idle"
            self.logger.info("Modo cambiado a: RECONOCIMIENTO")
        
        return state
