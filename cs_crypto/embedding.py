"""
DCT Domain Embedding/Extraction for steganography.
Embeds encrypted data into carrier image's DCT coefficients.
"""

import numpy as np
from scipy.fft import dctn, idctn


class DCTEmbedder:
    """
    Embeds encrypted measurement data into carrier image using DCT domain.
    The embedding strength controls the trade-off between invisibility and robustness.
    """
    
    def __init__(self, embedding_strength: float = 0.001):
        self.strength = embedding_strength
    
    def embed(
        self, 
        carrier: np.ndarray, 
        encrypted_data: list, 
        M: int, 
        N: int
    ) -> tuple:
        """
        Embed encrypted data into carrier image's DCT domain.
        
        Args:
            carrier: Carrier image (H x W x 3, RGB)
            encrypted_data: List of 3 encrypted measurement matrices [Y1, Y2, Y3]
            M: Number of rows to embed
            N: Number of columns to embed
            
        Returns:
            Tuple of (stego_image, dct_before, dct_after) for visualization
        """
        carrier = carrier.astype(np.float64)
        
        R = carrier[:, :, 0]
        G = carrier[:, :, 1]
        B = carrier[:, :, 2]
        
        R_dct = dctn(R, norm='ortho')
        G_dct = dctn(G, norm='ortho')
        B_dct = dctn(B, norm='ortho')
        
        dct_before = R_dct.copy()
        
        Y1_dct = dctn(encrypted_data[0], norm='ortho')
        Y2_dct = dctn(encrypted_data[1], norm='ortho')
        Y3_dct = dctn(encrypted_data[2], norm='ortho')
        
        R_dct[:M, :N] -= Y1_dct * self.strength
        G_dct[:M, :N] -= Y2_dct * self.strength
        B_dct[:M, :N] -= Y3_dct * self.strength
        
        dct_after = R_dct.copy()
        
        R_new = idctn(R_dct, norm='ortho')
        G_new = idctn(G_dct, norm='ortho')
        B_new = idctn(B_dct, norm='ortho')
        
        stego = np.stack([R_new, G_new, B_new], axis=2)
        
        return stego, dct_before, dct_after
    
    def extract(
        self, 
        stego_image: np.ndarray, 
        carrier_dct_channels: tuple, 
        M: int, 
        N: int
    ) -> list:
        """
        Extract encrypted data from stego image.
        
        Args:
            stego_image: Steganographic image (H x W x 3)
            carrier_dct_channels: Tuple of (R_dct_orig, G_dct_orig, B_dct_orig)
            M: Number of rows embedded
            N: Number of columns embedded
            
        Returns:
            List of 3 extracted measurement matrices [Y1, Y2, Y3]
        """
        stego = stego_image.astype(np.float64)
        
        R_stego_dct = dctn(stego[:, :, 0], norm='ortho')
        G_stego_dct = dctn(stego[:, :, 1], norm='ortho')
        B_stego_dct = dctn(stego[:, :, 2], norm='ortho')
        
        R_orig, G_orig, B_orig = carrier_dct_channels
        
        Y1_dct = (R_stego_dct[:M, :N] - R_orig[:M, :N]) * (-1) / self.strength
        Y2_dct = (G_stego_dct[:M, :N] - G_orig[:M, :N]) * (-1) / self.strength
        Y3_dct = (B_stego_dct[:M, :N] - B_orig[:M, :N]) * (-1) / self.strength
        
        Y1 = idctn(Y1_dct, norm='ortho')
        Y2 = idctn(Y2_dct, norm='ortho')
        Y3 = idctn(Y3_dct, norm='ortho')
        
        return [Y1, Y2, Y3]
    
    @staticmethod
    def get_carrier_dct(carrier: np.ndarray) -> tuple:
        """Get DCT coefficients of original carrier for extraction."""
        carrier = carrier.astype(np.float64)
        R_dct = dctn(carrier[:, :, 0], norm='ortho')
        G_dct = dctn(carrier[:, :, 1], norm='ortho')
        B_dct = dctn(carrier[:, :, 2], norm='ortho')
        return (R_dct, G_dct, B_dct)
