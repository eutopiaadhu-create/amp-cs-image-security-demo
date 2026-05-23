"""
Index Scrambling/Unscrambling for image encryption.
Implements pixel-level permutation using random indices.
"""

import numpy as np


class IndexScrambler:
    """
    Performs index-based scrambling and unscrambling of image data.
    Uses a fixed random seed to generate reproducible permutation indices.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def generate_index(self, total_size: int) -> np.ndarray:
        """Generate random permutation index (equivalent to MATLAB randperm)."""
        rng = np.random.RandomState(self.seed)
        return rng.permutation(total_size)
    
    def scramble(self, image: np.ndarray, index: np.ndarray = None) -> np.ndarray:
        """
        Scramble image using index permutation.
        Corresponds to MATLAB IndexScramble.m
        """
        height, width = image.shape
        total = height * width
        
        if index is None:
            index = self.generate_index(total)
        
        vec = image.reshape(total, order='C')
        enc_vec = np.zeros(total)
        
        for k in range(total):
            enc_vec[k] = vec[index[k]]
        
        return enc_vec.reshape(height, width, order='C')
    
    def unscramble(self, encrypted_image: np.ndarray, index: np.ndarray = None) -> np.ndarray:
        """
        Reverse the scrambling operation.
        Corresponds to MATLAB UnIndexScramble.m
        """
        height, width = encrypted_image.shape
        total = height * width
        
        if index is None:
            index = self.generate_index(total)
        
        vec = encrypted_image.reshape(total, order='C')
        dec_vec = np.zeros(total)
        
        for k in range(total):
            dec_vec[index[k]] = vec[k]
        
        return dec_vec.reshape(height, width, order='C')
