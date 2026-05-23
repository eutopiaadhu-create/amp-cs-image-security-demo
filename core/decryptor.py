"""
Decryption workflow orchestrator.
Coordinates extraction and reconstruction based on user permission level.
"""

import numpy as np
from typing import List, Optional
from PIL import Image
import os

from cs_crypto.measurement import MeasurementMatrix
from cs_crypto.scramble import IndexScrambler
from cs_crypto.embedding import DCTEmbedder
from cs_crypto.reconstruction import UserAReconstructor, UserBReconstructor
from ai_modules.face_detector import FaceDetector
from ai_modules.roi_analyzer import ROIAnalyzer
from ai_modules.privacy_protector import PrivacyProtector
from core.permission import UserLevel


class Decryptor:
    """
    Full decryption pipeline orchestrator.
    
    Pipeline:
    1. Extract encrypted data from stego image (DCT domain)
    2. Unscramble indices
    3. Remove encryption matrix R
    4. (UserB only) Remove permission matrix P
    5. SPL iterative reconstruction
    6. (Optional) Apply privacy protection based on permission level
    """
    
    def __init__(self, embedding_strength: float = 0.001):
        self.embedding_strength = embedding_strength
        self.scrambler = IndexScrambler(seed=42)
        self.embedder = DCTEmbedder(embedding_strength)
        self.face_detector = FaceDetector()
        self.roi_analyzer = ROIAnalyzer()
        self.privacy_protector = PrivacyProtector(method='mosaic', mosaic_size=10)
    
    def load_ai_model(self, model_path: str) -> bool:
        """Load YOLO model for privacy protection."""
        return self.face_detector.load(model_path)
    
    def decrypt_user_a(
        self,
        encryption_result: dict,
        output_dir: str = None
    ) -> List[np.ndarray]:
        """
        Decrypt for User A - recovers target images only.
        
        Args:
            encryption_result: Dict from Encryptor.encrypt()
            output_dir: Optional output directory
            
        Returns:
            List of 3 recovered target images
        """
        height = encryption_result['height']
        width = encryption_result['width']
        M = encryption_result['M']
        index = encryption_result['index']
        Y_scrambled = encryption_result['Y_scrambled']
        
        mm = MeasurementMatrix(height, width)
        Phi = mm.generate()
        R = mm.generate_encryption_matrix_R()
        
        reconstructor = UserAReconstructor()
        recovered = []
        
        for i, Y_scr in enumerate(Y_scrambled):
            print(f"  [UserA] Reconstructing target image {i+1}/3...")
            img = reconstructor.decrypt(Y_scr, Phi, R, index, height, width, self.scrambler)
            recovered.append(img)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, img in enumerate(recovered):
                Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
                    os.path.join(output_dir, f'recovered_A_{i+1}.png'))
        
        return recovered
    
    def decrypt_user_b(
        self,
        encryption_result: dict,
        output_dir: str = None
    ) -> dict:
        """
        Decrypt for User B - recovers both source and target images.
        
        Args:
            encryption_result: Dict from Encryptor.encrypt()
            output_dir: Optional output directory
            
        Returns:
            Dict with 'sources' and 'targets' lists
        """
        height = encryption_result['height']
        width = encryption_result['width']
        M = encryption_result['M']
        index = encryption_result['index']
        Y_scrambled = encryption_result['Y_scrambled']
        P_matrices = encryption_result['P_matrices']
        
        mm = MeasurementMatrix(height, width)
        Phi = mm.generate()
        R = mm.generate_encryption_matrix_R()
        
        reconstructor_a = UserAReconstructor()
        targets = []
        for i, Y_scr in enumerate(Y_scrambled):
            print(f"  [UserB] Reconstructing target image {i+1}/3...")
            img = reconstructor_a.decrypt(Y_scr, Phi, R, index, height, width, self.scrambler)
            targets.append(img)
        
        reconstructor_b = UserBReconstructor()
        sources = []
        for i, (Y_scr, P) in enumerate(zip(Y_scrambled, P_matrices)):
            print(f"  [UserB] Reconstructing source image {i+1}/3...")
            img = reconstructor_b.decrypt(Y_scr, Phi, R, P, index, height, width, self.scrambler)
            sources.append(img)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, img in enumerate(sources):
                Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
                    os.path.join(output_dir, f'recovered_B_source_{i+1}.png'))
            for i, img in enumerate(targets):
                Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
                    os.path.join(output_dir, f'recovered_B_target_{i+1}.png'))
        
        return {'sources': sources, 'targets': targets}
    
    def apply_privacy_protection(
        self,
        image: np.ndarray,
        original_path: str = None,
        user_level: UserLevel = UserLevel.USER_A
    ) -> np.ndarray:
        """
        Apply AI-driven privacy protection based on user level.
        Lower permission users see sensitive regions blurred/mosaic'd.
        """
        if user_level.value >= UserLevel.USER_B.value:
            return image
        
        if not self.face_detector.is_loaded:
            return image
        
        if original_path and os.path.exists(original_path):
            detections = self.face_detector.detect_with_scaling(
                original_path, image.shape[:2]
            )
        else:
            import cv2
            temp_path = '_temp_detect.png'
            cv2.imwrite(temp_path, image)
            detections = self.face_detector.detect(image_path=temp_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if detections:
            import cv2
            if len(image.shape) == 2:
                image_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = image.astype(np.uint8)
            
            protected = self.privacy_protector.protect(image_bgr, detections)
            
            if len(image.shape) == 2:
                return cv2.cvtColor(protected, cv2.COLOR_BGR2GRAY).astype(np.float64)
            return protected.astype(np.float64)
        
        return image
    
    def compute_metrics(self, original: np.ndarray, recovered: np.ndarray) -> dict:
        """Compute PSNR and SSIM quality metrics."""
        original = original.astype(np.float64)
        recovered = recovered.astype(np.float64)
        
        mse = np.mean((original - recovered) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(255.0 * 255.0 / mse)
        
        from skimage.metrics import structural_similarity as ssim
        try:
            ssim_val = ssim(
                np.clip(original, 0, 255).astype(np.uint8),
                np.clip(recovered, 0, 255).astype(np.uint8),
                data_range=255
            )
        except ImportError:
            ssim_val = self._simple_ssim(original, recovered)
        
        return {'psnr': psnr, 'ssim': ssim_val}
    
    def _simple_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Simplified SSIM calculation as fallback."""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        return float(numerator / denominator)
