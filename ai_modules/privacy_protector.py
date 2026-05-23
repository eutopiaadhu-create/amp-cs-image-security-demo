"""
Privacy Protection Module.
Applies visual privacy protection (mosaic/blur) to sensitive regions.
"""

import numpy as np
from typing import List, Tuple, Optional
import cv2


class PrivacyProtector:
    """
    Applies privacy protection to sensitive regions in images.
    Supports mosaic (pixelation) and Gaussian blur methods.
    """
    
    def __init__(self, method: str = 'mosaic', mosaic_size: int = 10, blur_ksize: int = 31):
        """
        Args:
            method: 'mosaic' or 'blur'
            mosaic_size: Block size for mosaic effect
            blur_ksize: Kernel size for Gaussian blur
        """
        self.method = method
        self.mosaic_size = mosaic_size
        self.blur_ksize = blur_ksize
    
    def apply_mosaic(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply mosaic (pixelation) to a region."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        roi = image[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        
        if roi_h < 2 or roi_w < 2:
            return image
        
        small_w = max(1, roi_w // self.mosaic_size)
        small_h = max(1, roi_h // self.mosaic_size)
        
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        
        result = image.copy()
        result[y1:y2, x1:x2] = pixelated
        return result
    
    def apply_blur(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply Gaussian blur to a region."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        result = image.copy()
        roi = result[y1:y2, x1:x2]
        
        if roi.shape[0] < 2 or roi.shape[1] < 2:
            return result
        
        ksize = self.blur_ksize
        if ksize % 2 == 0:
            ksize += 1
        
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        result[y1:y2, x1:x2] = blurred
        return result
    
    def protect(
        self, 
        image: np.ndarray, 
        detections: List[dict],
        method: str = None
    ) -> np.ndarray:
        """
        Apply privacy protection to all detected sensitive regions.
        
        Args:
            image: Input image (BGR or RGB numpy array)
            detections: List of detection dicts with 'bbox' key
            method: Override default method ('mosaic' or 'blur')
            
        Returns:
            Protected image
        """
        if not detections:
            return image.copy()
        
        use_method = method or self.method
        result = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            if use_method == 'mosaic':
                result = self.apply_mosaic(result, bbox)
            else:
                result = self.apply_blur(result, bbox)
        
        return result
    
    def protect_pil(self, pil_image, detections: List[dict], method: str = None):
        """
        Apply privacy protection to a PIL Image.
        
        Args:
            pil_image: PIL Image object
            detections: Detection results
            method: Protection method
            
        Returns:
            Protected PIL Image
        """
        from PIL import Image
        
        img_array = np.array(pil_image)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        protected_bgr = self.protect(img_bgr, detections, method)
        
        if len(protected_bgr.shape) == 3 and protected_bgr.shape[2] == 3:
            protected_rgb = cv2.cvtColor(protected_bgr, cv2.COLOR_BGR2RGB)
        else:
            protected_rgb = protected_bgr
        
        return Image.fromarray(protected_rgb)
