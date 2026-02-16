import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MetricsManager:
    def __init__(self, log_interval: int = 30):
        self.log_interval = log_interval
        self.frame_count = 0
        self.start_time: Optional[float] = None
    
    def start(self):
        self.frame_count = 0
        self.start_time = time.time()
        logger.debug("Métricas iniciadas")
    
    def increment_frame(self):
        self.frame_count += 1
        
        # Loggear periódicamente
        if self.frame_count % self.log_interval == 0:
            self._log_metrics()
    
    def _log_metrics(self):
        if not self.start_time:
            return
        
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            fps = self.frame_count / elapsed
            logger.debug(f"FPS: {fps:.1f} | Frames: {self.frame_count}")
    
    def get_fps(self) -> float:
        if not self.start_time:
            return 0.0
        
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0
    
    def reset(self):
        self.frame_count = 0
        self.start_time = time.time()