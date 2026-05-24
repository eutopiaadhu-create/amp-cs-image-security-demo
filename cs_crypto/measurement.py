"""
Measurement Matrix generation for Compressive Sensing.
Supports both the legacy whole-image flow and the new block-CS AMP flow.
"""

import os
import numpy as np
from scipy import io
from scipy.linalg import orth
from .chaos import ChaosGenerator


class MeasurementMatrix:
    """
    Generates measurement matrices for compressive sensing.
    """

    def __init__(
        self,
        height: int,
        width: int,
        compression_ratio: float = 0.6,
        block_size: int = 33,
        amp_cs_ratio: int = 60,
    ):
        self.height = height
        self.width = width
        self.compression_ratio = compression_ratio
        self.block_size = block_size
        self.amp_cs_ratio = amp_cs_ratio
        self.M = round(compression_ratio * height)
        self._phi = None
        self._block_sampling_matrix = None

    def generate(self) -> np.ndarray:
        gen = ChaosGenerator()
        z = gen.generate_phi_sequence(self.height, self.width)
        z_matrix = z.reshape((self.width, self.height), order='F')
        phi_full = orth(z_matrix).T
        self._phi = phi_full[:self.M, :]
        return self._phi

    @property
    def phi(self) -> np.ndarray:
        if self._phi is None:
            self.generate()
        return self._phi

    def generate_encryption_matrix_R(self) -> np.ndarray:
        gen = ChaosGenerator()
        z = gen.generate_r_sequence(self.height, self.width)
        return z.reshape((self.height, self.width), order='F')

    def _workspace_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _sampling_matrix_dir(self) -> str:
        path = os.path.join(self._workspace_root(), "dataset", "sampling_matrix")
        os.makedirs(path, exist_ok=True)
        return path

    def _generate_chaos_phi_block(self, block_size: int, cr: float) -> np.ndarray:
        N = block_size * block_size
        M = round(cr * N)

        x = np.zeros(N)
        y = np.zeros(N)
        z = np.zeros(N)
        x[0], y[0], z[0] = 0.3, 0.4, 0.5
        a, w = 10, 3

        for i in range(1, N):
            x[i] = y[i - 1] - z[i - 1]
            y[i] = np.sin(np.pi * x[i - 1] - a * y[i - 1])
            z_prev = np.clip(z[i - 1], -1.0, 1.0)
            z[i] = np.cos(w * np.arccos(z_prev) + y[i - 1])

        np.random.seed(42)
        random_matrix = np.random.randn(N, N)
        for i in range(N):
            random_matrix[i, :] *= z[i]

        U, _, _ = np.linalg.svd(random_matrix)
        return U[:M, :].astype(np.float32)

    def generate_block_sampling_matrix(self, cs_ratio: int = None) -> np.ndarray:
        if self._block_sampling_matrix is not None:
            return self._block_sampling_matrix

        ratio = cs_ratio or self.amp_cs_ratio
        matrix_path = os.path.join(self._sampling_matrix_dir(), f"{ratio}.mat")
        if os.path.exists(matrix_path):
            self._block_sampling_matrix = io.loadmat(matrix_path)["sampling_matrix"].astype(np.float32)
            return self._block_sampling_matrix

        A = self._generate_chaos_phi_block(self.block_size, ratio / 100.0)
        io.savemat(matrix_path, {"sampling_matrix": A})
        self._block_sampling_matrix = A
        return self._block_sampling_matrix

    def pad_image(self, image: np.ndarray):
        row, col = image.shape
        row_pad = (self.block_size - np.mod(row, self.block_size)) % self.block_size
        col_pad = (self.block_size - np.mod(col, self.block_size)) % self.block_size
        padded = np.concatenate((image, np.zeros([row, col_pad])), axis=1) if col_pad else image.copy()
        padded = np.concatenate((padded, np.zeros([row_pad, padded.shape[1]])), axis=0) if row_pad else padded
        row_new, col_new = padded.shape
        return padded, row_new, col_new

    def image_to_blocks(self, image: np.ndarray):
        padded, row_new, col_new = self.pad_image(image)
        block_rows = row_new // self.block_size
        block_cols = col_new // self.block_size

        # Match the exact block ordering used by AMPNetBasic.sampling()
        inputs = np.expand_dims((padded / 255.0).astype(np.float32), axis=0)
        inputs = np.concatenate(np.split(inputs, block_rows, axis=1), axis=0)
        inputs = np.concatenate(np.split(inputs, block_cols, axis=2), axis=0)
        img_col = np.transpose(np.reshape(inputs, [-1, self.block_size * self.block_size]), (1, 0)).astype(np.float32)

        return img_col, row_new, col_new, block_rows, block_cols

    def blocks_to_image(self, blocks: np.ndarray, height: int, width: int, row_new: int, col_new: int) -> np.ndarray:
        image = np.zeros([row_new, col_new], dtype=np.float32)
        count = 0
        for y in range(0, col_new - self.block_size + 1, self.block_size):
            for x in range(0, row_new - self.block_size + 1, self.block_size):
                image[x:x + self.block_size, y:y + self.block_size] = blocks[:, count].reshape([self.block_size, self.block_size])
                count += 1
        return image[:height, :width]

    def generate_block_encryption_matrix_R(self, num_blocks: int) -> np.ndarray:
        gen = ChaosGenerator()
        z = gen.generate_r_sequence(num_blocks, num_blocks)
        return z.reshape((num_blocks, num_blocks), order='F')
