import logging
from pathlib import Path
from typing import Optional

import yaml

from pipeline.camera import Camera
from pipeline.detector import FaceDetector
from pipeline.tracker import FaceTracker
from UI.renderer import UIRenderer
from UI.input_handler import InputHandler, AppState
from core.face_manager import FaceManager
from core.frame_manager import FrameManager
from core.app_orchestrator import ApplicationOrchestrator
from communication.zmq_client import ZMQClient

logger = logging.getLogger(__name__)

# Aplicación principal de reconocimiento facial.
# Inicializa componentes y delega el loop principal al orquestador.

class FaceRecognizerSimple:
    def __init__(self, config_path: str = "config_simple.yaml"):
        self.config_path = Path(config_path)
        self.config = None
        
        # Componentes del sistema
        self.camera: Optional[Camera] = None
        self.detector: Optional[FaceDetector] = None
        self.tracker: Optional[FaceTracker] = None
        self.zmq_client: Optional[ZMQClient] = None
        self.renderer: Optional[UIRenderer] = None
        self.input_handler: Optional[InputHandler] = None
        self.face_manager: Optional[FaceManager] = None
        self.frame_manager: Optional[FrameManager] = None
        
        # Orquestador
        self.orchestrator: Optional[ApplicationOrchestrator] = None
        
        # Estado de la aplicación
        self.state = AppState()
        
        self._load_config()
    
    def _load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuración cargada desde: {self.config_path}")
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            raise
    
    def initialize(self):
        logger.info("Inicializando sistema...")
        
        # === Inicializar Cámara ===
        cam_config = self.config['camera']
        self.camera = Camera(
            index=cam_config['index'],
            width=cam_config['resolution'][0],
            height=cam_config['resolution'][1]
        )
        
        # === Inicializar Detector y Tracker ===
        det_config = self.config['detection']
        self.detector = FaceDetector(
            model_path=det_config['model_path'],
            confidence=det_config['confidence']
        )
        self.tracker = FaceTracker(
            interval=det_config['detection_interval']
        )
        
        # === Inicializar ZMQ (opcional) ===
        zmq_config = self.config.get('zmq', {})
        if zmq_config.get('enabled', False):
            self.zmq_client = ZMQClient(
                send_endpoint=zmq_config['endpoint']  # Solo un endpoint
                # recv_endpoint ya no es necesario
            )
            self.zmq_client.connect()
        
        # === Inicializar Managers y UI ===
        self.renderer = UIRenderer()
        self.input_handler = InputHandler()
        self.face_manager = FaceManager(id_timeout=5.0, match_threshold=50)
        self.frame_manager = FrameManager()
        
        # === Inicializar Orquestador ===
        self.orchestrator = ApplicationOrchestrator(
            camera=self.camera,
            detector=self.detector,
            tracker=self.tracker,
            face_manager=self.face_manager,
            frame_manager=self.frame_manager,
            renderer=self.renderer,
            input_handler=self.input_handler,
            zmq_client=self.zmq_client
        )
        
        logger.info("Sistema inicializado correctamente")
    
    def run(self):
        self.orchestrator.start(self.state)
    
    def cleanup(self):
        logger.info("Limpiando recursos...")
        
        if self.orchestrator:
            self.orchestrator.stop()
        
        if self.camera:
            self.camera.release()
        
        if self.zmq_client:
            self.zmq_client.close()
        
        if self.renderer:
            self.renderer.cleanup()
        
        logger.info("Recursos liberados")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app = None
    try:
        app = FaceRecognizerSimple()
        app.initialize()
        app.run()
    except KeyboardInterrupt:
        logger.info("Interrupción por teclado")
    except Exception as e:
        logger.error(f"Error fatal: {e}", exc_info=True)
    finally:
        if app:
            app.cleanup()


if __name__ == "__main__":
    main()