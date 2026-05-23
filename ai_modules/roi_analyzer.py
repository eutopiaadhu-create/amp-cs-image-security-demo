"""
ROI (Region of Interest) Analyzer for sensitive region identification.
Analyzes detected regions and generates masks for differential encryption.
"""

import numpy as np
from typing import List, Tuple, Optional


class ROIAnalyzer:
    """
    Analyzes regions of interest for differential encryption strategy.
    Sensitive regions get higher security (higher compression ratio).
    """
    
    def __init__(self, padding: int = 10):
        self.padding = padding
    
    def create_mask(
        self, 
        detections: List[dict], 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create binary mask from detections.
        
        Args:
            detections: List of detection dicts with 'bbox' key
            image_shape: (height, width) of target image
            
        Returns:
            Binary mask where 1 = sensitive region, 0 = background
        """
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(width, x2 + self.padding)
            y2 = min(height, y2 + self.padding)
            
            mask[y1:y2, x1:x2] = 1
        
        return mask
    
    def get_roi_regions(
        self, 
        detections: List[dict], 
        image_shape: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Get list of ROI bounding boxes with padding.
        
        Returns:
            List of (x1, y1, x2, y2) tuples
        """
        height, width = image_shape
        regions = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(width, x2 + self.padding)
            y2 = min(height, y2 + self.padding)
            
            regions.append((x1, y1, x2, y2))
        
        return regions
    
    def compute_sensitivity_map(
        self,
        detections: List[dict],
        image_shape: Tuple[int, int],
        base_level: float = 0.5,
        sensitive_level: float = 1.0
    ) -> np.ndarray:
        """
        Create a sensitivity map for differential encryption.
        Higher values = more sensitive = stronger encryption.
        
        Args:
            detections: Detection results
            image_shape: (height, width)
            base_level: Sensitivity for background regions
            sensitive_level: Sensitivity for detected regions
            
        Returns:
            Float array with sensitivity values
        """
        height, width = image_shape
        sensitivity = np.full((height, width), base_level, dtype=np.float32)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(width, x2 + self.padding)
            y2 = min(height, y2 + self.padding)
            
            conf = det.get('confidence', 1.0)
            level = base_level + (sensitive_level - base_level) * conf
            sensitivity[y1:y2, x1:x2] = level
        
        return sensitivity
    
    def suggest_compression_ratios(
        self,
        detections: List[dict],
        image_shape: Tuple[int, int],
        base_cr: float = 0.5,
        sensitive_cr: float = 0.9
    ) -> dict:
        """
        Suggest compression ratios for different regions.
        
        Returns:
            Dict with 'background_cr' and 'sensitive_cr' values
        """
        has_sensitive = len(detections) > 0
        
        return {
            'background_cr': base_cr,
            'sensitive_cr': sensitive_cr if has_sensitive else base_cr,
            'has_sensitive_regions': has_sensitive,
            'num_sensitive_regions': len(detections)
        }
