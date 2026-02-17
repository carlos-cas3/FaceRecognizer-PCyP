import logging
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

class FrameProcessor:
    def __init__(self, tracker, detector):
        self.tracker = tracker
        self.detector = detector
    
    def detect_faces(self, frame) -> List[Tuple[int, Any, Tuple[int, int, int, int]]]:
        return self.tracker.process(frame, self.detector)
    
    def reset_tracker(self):
        self.tracker.reset()
        logger.debug("Tracker reseteado")
