"""
Encryption workflow orchestrator.
Coordinates the full encryption pipeline: CS measurement -> chaos encryption -> DCT embedding.
"""

import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
from scipy.fft import dctn, idctn

from cs_crypto.chaos import ChaosGenerator
from cs_crypto.measurement import MeasurementMatrix
from cs_crypto.scramble import IndexScrambler
from cs_crypto.embedding import DCTEmbedder


class Encryptor:
    """
    Full encryption pipeline orchestrator.
    
    Pipeline:
    1. Load and preprocess images
    2. Compute relationship matrices P (B-user keys)
    3. Generate chaotic sequences -> Phi, R
    4. Compress: Y = Phi * target_image
    5. Encrypt: Y = Y * R
    6. Scramble: Y = IndexScramble(Y)
    7. Embed into carrier via DCT
    """
    
    def __init__(self, compression_ratio: float = 0.8, embedding_strength: float = 0.001):
        self.compression_ratio = compression_ratio
        self.embedding_strength = embedding_strength
        self.scrambler = IndexScrambler(seed=42)
        self.embedder = DCTEmbedder(embedding_strength)
    
    def load_image_gray(self, path: str) -> np.ndarray:
        """Load image and convert to grayscale float64."""
        img = Image.open(path)
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img, dtype=np.float64)
    
    def load_image_color(self, path: str) -> np.ndarray:
        """Load image as RGB float64."""
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float64)
    
    def compute_relationship_matrix(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute relationship matrix P between source and target.
        P = pinv(S' * S) * S' * T
        """
        return np.linalg.pinv(source.T @ source) @ source.T @ target
    
    def encrypt(
        self,
        source_paths: List[str],
        target_paths: List[str],
        carrier_path: str,
        output_dir: str = None
    ) -> dict:
        """
        Execute full encryption pipeline.
        
        Args:
            source_paths: List of 3 source image paths
            target_paths: List of 3 target image paths
            carrier_path: Carrier image path
            output_dir: Optional output directory for saving intermediate results
            
        Returns:
            Dictionary containing all encryption artifacts needed for decryption
        """
        import os
        
        sources = [self.load_image_gray(p) for p in source_paths]
        targets = [self.load_image_gray(p) for p in target_paths]
        carrier = self.load_image_color(carrier_path)
        
        height, width = sources[0].shape
        
        P_matrices = []
        for s, t in zip(sources, targets):
            P = self.compute_relationship_matrix(s, t)
            P_matrices.append(P)
        
        mm = MeasurementMatrix(height, width, self.compression_ratio)
        Phi = mm.generate()
        M = mm.M
        R = mm.generate_encryption_matrix_R()
        
        Y_list = []
        for t in targets:
            Y = Phi @ t
            Y_list.append(Y)
        
        Y_encrypted = []
        for Y in Y_list:
            Y_enc = Y @ R
            Y_encrypted.append(Y_enc)
        
        m_high, m_width = Y_encrypted[0].shape
        index = self.scrambler.generate_index(m_high * m_width)
        
        Y_scrambled = []
        for Y_enc in Y_encrypted:
            Y_scr = self.scrambler.scramble(Y_enc, index)
            Y_scrambled.append(Y_scr)
        
        stego, dct_before, dct_after = self.embedder.embed(
            carrier, Y_scrambled, M, width
        )
        
        carrier_dct = DCTEmbedder.get_carrier_dct(carrier)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            for i, s in enumerate(sources):
                Image.fromarray(s.astype(np.uint8)).save(
                    os.path.join(output_dir, f'01_source_image_{i+1}.png'))
            
            for i, t in enumerate(targets):
                Image.fromarray(t.astype(np.uint8)).save(
                    os.path.join(output_dir, f'04_target_image_{i+1}.png'))
            
            for i, Y_scr in enumerate(Y_scrambled):
                from matplotlib import cm
                normalized = (Y_scr - Y_scr.min()) / (Y_scr.max() - Y_scr.min() + 1e-10)
                Image.fromarray((normalized * 255).astype(np.uint8)).save(
                    os.path.join(output_dir, f'07_encrypted_data_{i+1}.png'))
            
            Image.fromarray(carrier.astype(np.uint8)).save(
                os.path.join(output_dir, '10_carrier_original.png'))
            
            Image.fromarray(np.clip(stego, 0, 255).astype(np.uint8)).save(
                os.path.join(output_dir, '11_carrier_steganographic.png'))
            
            self._save_dct_visualization(dct_before, dct_after, output_dir)
        
        result = {
            'stego_image': stego,
            'carrier_dct': carrier_dct,
            'P_matrices': P_matrices,
            'index': index,
            'height': height,
            'width': width,
            'M': M,
            'Y_scrambled': Y_scrambled,
        }
        
        return result
    
    def _save_dct_visualization(self, dct_before, dct_after, output_dir):
        """Save DCT coefficient visualizations."""
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(np.log(np.abs(dct_before) + 1), cmap='gray')
        ax.set_title('DCT Before Embedding')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '91_dct_before_embedding.png'), dpi=100)
        plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(np.log(np.abs(dct_after) + 1), cmap='gray')
        ax.set_title('DCT After Embedding')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '92_dct_after_embedding.png'), dpi=100)
        plt.close()
        
        dct_diff = np.abs(dct_after - dct_before)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(np.log(dct_diff + 1), cmap='hot')
        ax.set_title('DCT Difference (Embedded Secret)')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '93_dct_difference_secret.png'), dpi=100)
        plt.close()
