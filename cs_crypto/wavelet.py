"""
CDF 9/7 Wavelet Transform implementation.
Cohen-Daubechies-Feauveau 9/7 wavelet using lifting scheme.
"""

import numpy as np


class WaveletCDF97:
    """
    CDF 9/7 Wavelet Transform using lifting scheme.
    Equivalent to MATLAB waveletcdf97.m
    """
    
    LIFT_FILTER = np.array([
        -1.5861343420693648,
        -0.0529801185718856,
        0.8829110755411875,
        0.4435068520511142
    ])
    SCALE_FACTOR = 1.1496043988602418
    
    def __init__(self):
        self.s1 = self.LIFT_FILTER[0]
        self.s2 = self.LIFT_FILTER[1]
        self.s3 = self.LIFT_FILTER[2]
        self.s4 = self.LIFT_FILTER[3]
    
    def forward_1d(self, x: np.ndarray) -> np.ndarray:
        """1D forward CDF 9/7 transform."""
        n = len(x)
        if n < 2:
            return x.copy()
        
        x = x.astype(np.float64).copy()
        half = (n + 1) // 2
        
        even = x[0::2].copy()
        odd = x[1::2].copy()
        
        n_odd = len(odd)
        n_even = len(even)
        
        for i in range(n_odd):
            odd[i] += self.s1 * (even[i] + even[min(i+1, n_even-1)])
        
        for i in range(n_even):
            left = odd[max(i-1, 0)]
            right = odd[min(i, n_odd-1)] if i < n_odd else odd[n_odd-1]
            even[i] += self.s2 * (left + right)
        
        for i in range(n_odd):
            odd[i] += self.s3 * (even[i] + even[min(i+1, n_even-1)])
        
        for i in range(n_even):
            left = odd[max(i-1, 0)]
            right = odd[min(i, n_odd-1)] if i < n_odd else odd[n_odd-1]
            even[i] += self.s4 * (left + right)
        
        even *= self.SCALE_FACTOR
        odd /= self.SCALE_FACTOR
        
        result = np.zeros(n)
        result[:half] = even
        result[half:] = odd
        return result
    
    def inverse_1d(self, x: np.ndarray) -> np.ndarray:
        """1D inverse CDF 9/7 transform."""
        n = len(x)
        if n < 2:
            return x.copy()
        
        x = x.astype(np.float64).copy()
        half = (n + 1) // 2
        
        even = x[:half].copy() / self.SCALE_FACTOR
        odd = x[half:].copy() * self.SCALE_FACTOR
        
        n_odd = len(odd)
        n_even = len(even)
        
        for i in range(n_even):
            left = odd[max(i-1, 0)]
            right = odd[min(i, n_odd-1)] if i < n_odd else odd[n_odd-1]
            even[i] -= self.s4 * (left + right)
        
        for i in range(n_odd):
            odd[i] -= self.s3 * (even[i] + even[min(i+1, n_even-1)])
        
        for i in range(n_even):
            left = odd[max(i-1, 0)]
            right = odd[min(i, n_odd-1)] if i < n_odd else odd[n_odd-1]
            even[i] -= self.s2 * (left + right)
        
        for i in range(n_odd):
            odd[i] -= self.s1 * (even[i] + even[min(i+1, n_even-1)])
        
        result = np.zeros(n)
        result[0::2] = even
        result[1::2] = odd
        return result
    
    def forward_2d(self, image: np.ndarray, levels: int = 1) -> np.ndarray:
        """
        2D forward CDF 9/7 wavelet transform.
        
        Args:
            image: Input 2D array
            levels: Number of decomposition levels
        """
        result = image.astype(np.float64).copy()
        h, w = result.shape
        
        for _ in range(levels):
            for i in range(h):
                result[i, :w] = self.forward_1d(result[i, :w])
            
            for j in range(w):
                result[:h, j] = self.forward_1d(result[:h, j])
            
            h = (h + 1) // 2
            w = (w + 1) // 2
        
        return result
    
    def inverse_2d(self, coeffs: np.ndarray, levels: int = 1) -> np.ndarray:
        """
        2D inverse CDF 9/7 wavelet transform.
        
        Args:
            coeffs: Wavelet coefficients
            levels: Number of reconstruction levels
        """
        result = coeffs.astype(np.float64).copy()
        h_orig, w_orig = result.shape
        
        sizes = []
        h, w = h_orig, w_orig
        for _ in range(levels):
            sizes.append((h, w))
            h = (h + 1) // 2
            w = (w + 1) // 2
        
        for h, w in reversed(sizes):
            for j in range(w):
                result[:h, j] = self.inverse_1d(result[:h, j])
            
            for i in range(h):
                result[i, :w] = self.inverse_1d(result[i, :w])
        
        return result
    
    def transform(self, image: np.ndarray, levels: int) -> np.ndarray:
        """
        Perform wavelet transform (forward if levels > 0, inverse if levels < 0).
        Matches MATLAB waveletcdf97(X, Level) interface.
        """
        if levels >= 0:
            return self.forward_2d(image, levels)
        else:
            return self.inverse_2d(image, -levels)
