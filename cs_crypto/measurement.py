"""
Measurement Matrix generation for Compressive Sensing.
Generates orthogonal measurement matrix Phi from chaotic sequences.
"""

import numpy as np
from scipy.linalg import orth
from .chaos import ChaosGenerator


class MeasurementMatrix:
    """
    Generates the measurement matrix Phi using chaotic sequences.
    Phi is derived from orthogonalizing a chaotic z-sequence matrix.
    """
    
    def __init__(self, height: int, width: int, compression_ratio: float = 0.8):
        self.height = height
        self.width = width
        self.M = round(compression_ratio * height)
        self._phi = None
    
    def generate(self) -> np.ndarray:
        """
        Generate measurement matrix Phi.
        Corresponds to MATLAB: Phi = orth(reshape(z, width, high))'; Phi = Phi(1:M, :)
        """
        gen = ChaosGenerator()
        z = gen.generate_phi_sequence(self.height, self.width)
        
        # Reshape in Fortran (column-major) order to match MATLAB
        z_matrix = z.reshape((self.width, self.height), order='F')
        
        # Orthogonalize and transpose
        phi_full = orth(z_matrix).T
        
        # Take first M rows
        self._phi = phi_full[:self.M, :]
        return self._phi
    
    @property
    def phi(self) -> np.ndarray:
        if self._phi is None:
            self.generate()
        return self._phi
    
    def generate_encryption_matrix_R(self) -> np.ndarray:
        """
        Generate encryption matrix R from second chaotic sequence.
        Corresponds to MATLAB: R = reshape(zzz, high, width)
        """
        gen = ChaosGenerator()
        z = gen.generate_r_sequence(self.height, self.width)
        R = z.reshape((self.height, self.width), order='F')
        return R
