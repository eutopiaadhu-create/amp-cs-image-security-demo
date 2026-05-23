"""
3D Chaotic Map Generator for encryption keys.
Implements the 3D chaotic system used in 2D-CS encryption scheme.
"""

import numpy as np
from typing import Tuple


class ChaosGenerator:
    """
    3D Chaotic Map Generator.
    
    Generates chaotic sequences using the 3D map:
        x(i) = y(i-1) - z(i-1)
        y(i) = sin(π * x(i-1) - a * y(i-1))
        z(i) = cos(w * acos(z(i-1)) + y(i-1))
    """
    
    def __init__(self, a: float = 10.0, w: float = 3.0):
        self.a = a
        self.w = w
    
    def generate(
        self, 
        length: int, 
        x0: float = 0.3, 
        y0: float = 0.4, 
        z0: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3D chaotic sequences.
        
        Args:
            length: Total length of sequences
            x0, y0, z0: Initial values
            
        Returns:
            Tuple of (x, y, z) sequences
        """
        x = np.zeros(length)
        y = np.zeros(length)
        z = np.zeros(length)
        
        x[0], y[0], z[0] = x0, y0, z0
        
        for i in range(1, length):
            x[i] = y[i-1] - z[i-1]
            y[i] = np.sin(np.pi * x[i-1] - self.a * y[i-1])
            z_prev = np.clip(z[i-1], -1.0, 1.0)
            z[i] = np.cos(self.w * np.arccos(z_prev) + y[i-1])
        
        return x, y, z
    
    def generate_phi_sequence(
        self, 
        height: int, 
        width: int
    ) -> np.ndarray:
        """
        Generate z sequence for measurement matrix Phi.
        Uses initial values (0.3, 0.4, 0.5).
        """
        T = height * width
        _, _, z = self.generate(T, x0=0.3, y0=0.4, z0=0.5)
        return z
    
    def generate_r_sequence(
        self, 
        height: int, 
        width: int
    ) -> np.ndarray:
        """
        Generate z sequence for encryption matrix R.
        Uses initial values (0.6, 0.2, 0.5).
        """
        T = height * width
        _, _, z = self.generate(T, x0=0.6, y0=0.2, z0=0.5)
        return z


def generate_chaos_keys(height: int, width: int) -> dict:
    """
    Convenience function to generate all chaos-based keys.
    
    Returns:
        Dictionary containing:
        - 'phi_z': z sequence for Phi matrix
        - 'r_z': z sequence for R matrix
    """
    gen = ChaosGenerator()
    return {
        'phi_z': gen.generate_phi_sequence(height, width),
        'r_z': gen.generate_r_sequence(height, width)
    }
