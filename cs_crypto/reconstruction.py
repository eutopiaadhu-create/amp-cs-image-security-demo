"""
SPL (Smoothed Projected Landweber) Reconstruction for Compressive Sensing.
Implements iterative reconstruction algorithm for recovering images from CS measurements.
"""

import numpy as np
from scipy.signal import wiener
from scipy.ndimage import convolve
from .wavelet import WaveletCDF97


class SPLReconstructor:
    """
    Smoothed Projected Landweber (SPL) iterative reconstruction.
    Equivalent to MATLAB SPLIterationUserA.m and SPLIterationUserB.m
    """
    
    def __init__(
        self,
        max_iterations: int = 50,
        lambda_val: float = 0.8,
        tolerance: float = 0.05,
        beta: float = 0.7,
        sigma_min: float = 0.01,
        num_levels: int = 3
    ):
        self.max_iterations = max_iterations
        self.lambda_val = lambda_val
        self.tolerance = tolerance
        self.beta = beta
        self.sigma_min = sigma_min
        self.num_levels = num_levels
        self.wavelet = WaveletCDF97()
    
    def _wiener_filter(self, x: np.ndarray, size: tuple = (3, 3)) -> np.ndarray:
        """Apply Wiener filter for noise reduction."""
        try:
            return wiener(x, size)
        except:
            return x
    
    def _smooth_function(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply smoothing function: delta = -x * exp(-|x|^2 / (2*sigma^2))
        """
        delta = -x * np.exp(-np.abs(x)**2 / (2 * sigma**2))
        return x + 4.3 * delta
    
    def _soft_threshold(self, x: np.ndarray, height: int, width: int) -> np.ndarray:
        """Apply soft thresholding for sparsity."""
        threshold = self.lambda_val * np.sqrt(2 * np.log(width * height)) * \
                   (np.median(np.abs(x.flatten())) / 0.6745)
        x[np.abs(x) < threshold] = 0
        return x
    
    def _pixel_error(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate pixel error between two images."""
        diff = x1 - x2
        return np.sqrt(np.mean(diff.flatten()**2))
    
    def _spl_iteration(
        self, 
        y: np.ndarray, 
        x: np.ndarray, 
        phi: np.ndarray, 
        sigma: float,
        height: int,
        width: int
    ) -> tuple:
        """
        Single SPL iteration step.
        """
        x_hat = x.copy()
        x_hat = self._wiener_filter(x_hat)
        x_hat = x_hat + phi.T @ (y - phi @ x_hat)
        
        x1 = x_hat.copy()
        x_check = x1.copy()
        
        x_check = self.wavelet.transform(x_check, self.num_levels)
        x_check = self._smooth_function(x_check, sigma)
        x_check = self._soft_threshold(x_check, height, width)
        x_bar = self.wavelet.transform(x_check, -self.num_levels)
        
        x_ret = x_bar + phi.T @ (y - phi @ x_bar)
        x2 = x_ret.copy()
        
        pe = self._pixel_error(x1, x2)
        return x_ret, pe
    
    def reconstruct(
        self, 
        y: np.ndarray, 
        phi: np.ndarray, 
        height: int, 
        width: int
    ) -> np.ndarray:
        """
        Reconstruct image from CS measurements using SPL algorithm.
        
        Args:
            y: Measurement matrix (M x N)
            phi: Measurement matrix (M x height)
            height: Original image height
            width: Original image width
            
        Returns:
            Reconstructed image
        """
        x = phi.T @ y
        
        sigma = 2 * np.max(np.abs(x.flatten()))
        d_prev = 0
        
        for i in range(self.max_iterations):
            x, d_cur = self._spl_iteration(y, x, phi, sigma, height, width)
            
            if d_prev != 0 and abs(d_cur - d_prev) < self.tolerance:
                break
            
            d_prev = d_cur
            
            if sigma > self.sigma_min:
                sigma = sigma * self.beta
        
        x, _ = self._spl_iteration(y, x, phi, sigma, height, width)
        
        return x


class UserAReconstructor(SPLReconstructor):
    """
    User A level reconstruction.
    Can only recover target images (no access to P matrix).
    """
    
    def decrypt(
        self,
        y_encrypted: np.ndarray,
        phi: np.ndarray,
        R: np.ndarray,
        index: np.ndarray,
        height: int,
        width: int,
        scrambler
    ) -> np.ndarray:
        """
        Decrypt and reconstruct for User A.
        
        Args:
            y_encrypted: Encrypted measurement data
            phi: Measurement matrix
            R: Encryption matrix R
            index: Scrambling index
            height, width: Image dimensions
            scrambler: IndexScrambler instance
        """
        y = scrambler.unscramble(y_encrypted, index)
        y = y @ np.linalg.pinv(R)
        
        return self.reconstruct(y, phi, height, width)


class UserBReconstructor(SPLReconstructor):
    """
    User B level reconstruction.
    Can recover both source and target images (has access to P matrix).
    """
    
    def decrypt(
        self,
        y_encrypted: np.ndarray,
        phi: np.ndarray,
        R: np.ndarray,
        P: np.ndarray,
        index: np.ndarray,
        height: int,
        width: int,
        scrambler
    ) -> np.ndarray:
        """
        Decrypt and reconstruct for User B.
        
        Args:
            y_encrypted: Encrypted measurement data
            phi: Measurement matrix
            R: Encryption matrix R
            P: Permission matrix P (source-target relationship)
            index: Scrambling index
            height, width: Image dimensions
            scrambler: IndexScrambler instance
        """
        y = scrambler.unscramble(y_encrypted, index)
        y = y @ np.linalg.pinv(R)
        y = y @ np.linalg.pinv(P)
        
        return self.reconstruct(y, phi, height, width)
