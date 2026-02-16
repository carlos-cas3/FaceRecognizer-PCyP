import cv2
from typing import List, Tuple

# Panel flotante para ingresar nombres durante el registro

class RegisterPanelDisplay:
    
    # Configuración de layout
    MARGIN_RIGHT = 20
    MARGIN_TOP = 60
    LINE_HEIGHT = 35
    PADDING = 10
    
    # Colores
    BG_COLOR = (0, 0, 0)           # Negro
    COLOR_TITLE = (0, 255, 255)     # Cian
    COLOR_TEXT = (255, 255, 255)    # Blanco
    COLOR_HINT = (200, 200, 200)    # Gris claro
    
    # Opacidad del fondo
    BG_ALPHA = 0.7
    
    @staticmethod
    def should_draw(mode: str, register_state: str) -> bool:
        return mode == "register" and register_state == "selecting"
    
    @staticmethod
    def draw(
        frame,
        width: int,
        selected_face_ids: List[int],
        current_face_index: int,
        current_name: str
    ):
        if not selected_face_ids:
            return
        
        # Preparar datos
        current_face_id = selected_face_ids[current_face_index]
        progress = f"{current_face_index + 1}/{len(selected_face_ids)}"
        
        # Definir textos
        texts = RegisterPanelDisplay._create_panel_texts(
            progress, current_face_id, current_name
        )
        
        # Calcular dimensiones
        max_width, total_height = RegisterPanelDisplay._calculate_dimensions(texts)
        
        # Dibujar fondo
        RegisterPanelDisplay._draw_background(
            frame, width, max_width, total_height
        )
        
        # Dibujar textos
        RegisterPanelDisplay._draw_texts(frame, width, texts)
    
    @staticmethod
    def _create_panel_texts(
        progress: str,
        face_id: int,
        name: str
    ) -> List[Tuple[str, float, Tuple[int, int, int]]]:
        return [
            (f"REGISTRO {progress}", 0.7, RegisterPanelDisplay.COLOR_TITLE),
            (f"ID: {face_id}", 0.9, RegisterPanelDisplay.COLOR_TITLE),
            (f"NOMBRE: {name}_", 0.7, RegisterPanelDisplay.COLOR_TEXT),
            ("", 0.5, RegisterPanelDisplay.COLOR_HINT),  # Línea vacía
            ("Enter: OK", 0.5, RegisterPanelDisplay.COLOR_HINT),
            ("Esc: Cancelar", 0.5, RegisterPanelDisplay.COLOR_HINT)
        ]
    
    @staticmethod
    def _calculate_dimensions(
        texts: List[Tuple[str, float, Tuple[int, int, int]]]
    ) -> Tuple[int, int]:
        max_width = 0
        
        for text, scale, _ in texts:
            if text:
                (text_w, _), _ = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    2
                )
                max_width = max(max_width, text_w)
        
        total_height = len(texts) * RegisterPanelDisplay.LINE_HEIGHT
        
        return max_width, total_height
    
    @staticmethod
    def _draw_background(frame, width: int, max_width: int, total_height: int):
        x1 = width - max_width - RegisterPanelDisplay.MARGIN_RIGHT - RegisterPanelDisplay.PADDING
        y1 = RegisterPanelDisplay.MARGIN_TOP - 30
        x2 = width - RegisterPanelDisplay.MARGIN_RIGHT + RegisterPanelDisplay.PADDING
        y2 = RegisterPanelDisplay.MARGIN_TOP + total_height + RegisterPanelDisplay.PADDING
        
        # Crear overlay con transparencia
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            RegisterPanelDisplay.BG_COLOR,
            -1
        )
        cv2.addWeighted(
            overlay,
            RegisterPanelDisplay.BG_ALPHA,
            frame,
            1 - RegisterPanelDisplay.BG_ALPHA,
            0,
            frame
        )
    
    @staticmethod
    def _draw_texts(
        frame,
        width: int,
        texts: List[Tuple[str, float, Tuple[int, int, int]]]
    ):
        y_pos = RegisterPanelDisplay.MARGIN_TOP
        
        for text, scale, color in texts:
            if text:
                (text_w, _), _ = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    2
                )
                x_pos = width - text_w - RegisterPanelDisplay.MARGIN_RIGHT
                
                cv2.putText(
                    frame,
                    text,
                    (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    color,
                    2
                )
            
            y_pos += RegisterPanelDisplay.LINE_HEIGHT