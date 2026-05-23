"""
YOLO-based Face Detection Module.
Provides intelligent perception of sensitive regions in images.
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class FaceDetector:
    """
    YOLO-based face/person detector for sensitive region identification.
    Supports YOLOv8/v11 models.
    """
    
    def __init__(self, model_path: str = None, confidence: float = 0.5):
        self.confidence = confidence
        self.model = None
        self.model_path = model_path
        self._loaded = False
    
    def load(self, model_path: str = None) -> bool:
        """
        Load YOLO model.
        
        Args:
            model_path: Path to .pt model file
            
        Returns:
            True if loaded successfully
        """
        if model_path:
            self.model_path = model_path
            
        if not self.model_path:
            return False
            
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self._loaded = True
            return True
        except Exception as e:
            print(f"[FaceDetector] Failed to load model: {e}")
            self._loaded = False
            return False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def detect(
        self, 
        image_path: str = None, 
        image_array: np.ndarray = None,
        imgsz: int = 256
    ) -> List[dict]:
        """
        Detect faces/persons in image.
        
        Args:
            image_path: Path to image file
            image_array: Image as numpy array (BGR)
            imgsz: Inference image size
            
        Returns:
            List of detection dicts with keys: 'bbox', 'confidence', 'class_id'
        """
        if not self._loaded:
            return []
        
        source = image_path if image_path else image_array
        if source is None:
            return []
        
        try:
            results = self.model(source, imgsz=imgsz, conf=self.confidence, verbose=False)
            
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class_id': cls_id
                })
            
            return detections
        except Exception as e:
            print(f"[FaceDetector] Detection error: {e}")
            return []
    
    def detect_with_scaling(
        self,
        original_path: str,
        target_shape: Tuple[int, int],
        imgsz: int = 256
    ) -> List[dict]:
        """
        Detect on original image and scale bounding boxes to target shape.
        Useful when detection is done on original but applied to reconstructed image.
        
        Args:
            original_path: Path to original image for detection
            target_shape: (height, width) of target image
            imgsz: Inference size
            
        Returns:
            List of scaled detection dicts
        """
        import cv2
        
        orig = cv2.imread(original_path)
        if orig is None:
            return []
        
        h_orig, w_orig = orig.shape[:2]
        h_target, w_target = target_shape
        
        scale_x = w_target / w_orig
        scale_y = h_target / h_orig
        
        detections = self.detect(image_path=original_path, imgsz=imgsz)
        
        scaled = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            scaled.append({
                'bbox': (
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                ),
                'confidence': det['confidence'],
                'class_id': det['class_id']
            })
        
        return scaled
