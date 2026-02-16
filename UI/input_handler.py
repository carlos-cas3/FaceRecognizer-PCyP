import logging
from typing import Dict, Any, List, Tuple, Optional

from core import face_manager

logger = logging.getLogger(__name__)


class AppState:
    """Clase para representar el estado de la aplicación"""
    def __init__(self):
        self.mode = "recognize"  # "register" o "recognize"
        self.register_state = "idle"  # "idle", "selecting"
        self.selected_face_ids: List[int] = []
        self.current_face_index = 0
        self.current_name = ""
        self.should_exit = False
        self.should_send_to_cpp = False
        self.face_to_send: Optional[Tuple[int, Tuple[int, int, int, int], str]] = None


class InputHandler:
    """
    Maneja toda la entrada del teclado y transiciones de estado.
    Desacopla la lógica de input del loop principal.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_key(
        self,
        key: int,
        state: AppState,
        faces: List[Tuple[int, Any, Tuple[int, int, int, int]]],
        face_manager
    ) -> AppState:
        """
        Procesa una tecla presionada y retorna el nuevo estado.
        
        Args:
            key: Código de tecla de OpenCV
            state: Estado actual de la aplicación
            faces: Lista de caras detectadas
            locked_faces: Diccionario de caras bloqueadas
            
        Returns:
            AppState actualizado
        """
        # Resetear flags de acción
        state.should_send_to_cpp = False
        state.face_to_send = None
        
        if state.mode == "register" and state.register_state == "selecting":
            return self._handle_register_selecting(key, state, faces)
        elif state.mode == "register" and state.register_state == "idle":
            return self._handle_register_idle(key, state, faces, face_manager)
        else:
            return self._handle_recognize_mode(key, state)
    
    def _handle_register_selecting(
        self,
        key: int,
        state: AppState,
        faces: List[Tuple[int, Any, Tuple[int, int, int, int]]]
    ) -> AppState:
        """Maneja input durante el registro de nombre"""
        
        if key == 27:  # Escape - cancelar registro
            self.logger.info("Registro cancelado")
            state.register_state = "idle"
            state.selected_face_ids = []
            state.current_face_index = 0
            state.current_name = ""
            
        elif key == 13:  # Enter - confirmar nombre
            if state.current_name.strip():
                face_id = state.selected_face_ids[state.current_face_index]
                
                # Buscar bbox de esta cara
                face_bbox = None
                for f_id, f_crop, f_bbox in faces:
                    if f_id == face_id:
                        face_bbox = f_bbox
                        break
                
                if face_bbox:
                    # Marcar que se debe enviar a C++
                    state.should_send_to_cpp = True
                    state.face_to_send = (face_id, face_bbox, state.current_name.strip())
                
                self.logger.info(f"Cara {face_id} registrada como: {state.current_name}")
                
                # Avanzar al siguiente
                state.current_face_index += 1
                if state.current_face_index >= len(state.selected_face_ids):
                    state.register_state = "idle"
                    state.selected_face_ids = []
                    state.current_face_index = 0
                    self.logger.info("Registro completado")
                else:
                    state.current_name = ""
        
        elif key == 8:  # Backspace
            state.current_name = state.current_name[:-1]
        
        elif 32 <= key <= 126:  # Caracteres imprimibles
            state.current_name += chr(key)
        
        return state
    
    def _handle_register_idle(
        self,
        key: int,
        state: AppState,
        faces: List[Tuple[int, Any, Tuple[int, int, int, int]]],
        face_manager
    ) -> AppState:
        """Maneja input en modo registro (seleccionando caras)"""
        
        if key == ord('3'):  # Salir
            self.logger.info("Solicitud de salida")
            state.should_exit = True
        
        elif key == ord('2'):  # Cambiar a reconocimiento
            state.mode = "recognize"
            state.register_state = "idle"
            self.logger.info("Modo cambiado a: RECONOCIMIENTO")
        
        elif ord('0') <= key <= ord('9'):  # Seleccionar cara 0-9
            idx = key - ord('0')
            if idx < len(faces):
                face_id, face_crop, bbox = faces[idx]
                if face_id not in state.selected_face_ids:
                    state.selected_face_ids.append(face_id)
                    face_manager.lock_face(face_id, bbox)
                    self.logger.info(f"Cara {face_id} seleccionada")
        
        elif key == 13 and state.selected_face_ids:  # Enter - confirmar selección
            state.register_state = "selecting"
            state.current_face_index = 0
            self.logger.info(f"Iniciando registro de {len(state.selected_face_ids)} caras")
        
        return state
    
    def _handle_recognize_mode(self, key: int, state: AppState) -> AppState:
        """Maneja input en modo reconocimiento"""
        
        if key == ord('3'):  # Salir
            self.logger.info("Solicitud de salida")
            state.should_exit = True
        
        elif key == ord('1'):  # Cambiar a registro
            state.mode = "register"
            state.register_state = "idle"
            state.selected_face_ids = []
            state.current_name = ""
            self.logger.info("Modo cambiado a: REGISTRO")
        
        elif key == ord('2'):  # Ya estamos en reconocimiento
            state.mode = "recognize"
            state.register_state = "idle"
            self.logger.info("Modo cambiado a: RECONOCIMIENTO")
        
        return state