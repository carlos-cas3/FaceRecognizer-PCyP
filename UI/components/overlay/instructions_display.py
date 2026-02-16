import cv2

# Este componente se encarga de mostrar instrucciones en el footer de la pantalla

class InstructionsDisplay:
    # Configuración
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    COLOR = (200, 200, 200)  # Gris claro
    POSITION_X = 10
    OFFSET_FROM_BOTTOM = 20
    
    # Textos de instrucciones predefinidos
    INSTRUCTIONS = {
        'register_idle': "0-9:Seleccionar  Enter:Confirmar  2:Rec  3:Salir",
        'register_selecting': None,  # No mostrar (están en el panel)
        'recognize': "1:Registrar  2:Reconocer  3:Salir"
    }
    
    @staticmethod
    def get_instructions(mode: str, register_state: str) -> str:
        if mode == "register":
            if register_state == "idle":
                return InstructionsDisplay.INSTRUCTIONS['register_idle']
            elif register_state == "selecting":
                return InstructionsDisplay.INSTRUCTIONS['register_selecting']
            else:
                return ""
        else:
            return InstructionsDisplay.INSTRUCTIONS['recognize']
    
    @staticmethod
    def draw(frame, height: int, mode: str, register_state: str):
        instructions = InstructionsDisplay.get_instructions(mode, register_state)
        
        if not instructions:
            return
        
        y_position = height - InstructionsDisplay.OFFSET_FROM_BOTTOM
        
        cv2.putText(
            frame,
            instructions,
            (InstructionsDisplay.POSITION_X, y_position),
            InstructionsDisplay.FONT,
            InstructionsDisplay.FONT_SCALE,
            InstructionsDisplay.COLOR,
            InstructionsDisplay.FONT_THICKNESS
        )