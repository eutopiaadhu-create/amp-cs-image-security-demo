"""
Encryption workflow orchestrator.
Coordinates the full encryption pipeline: block CS measurement -> chaos encryption -> DCT embedding.
"""

import os
import numpy as np
from typing import List
from PIL import Image

from cs_crypto.measurement import MeasurementMatrix
from cs_crypto.scramble import IndexScrambler
from cs_crypto.embedding import DCTEmbedder


class Encryptor:
    def __init__(self, compression_ratio: float = 0.6, embedding_strength: float = 0.001):
        self.compression_ratio = compression_ratio
        self.embedding_strength = embedding_strength
        self.scrambler = IndexScrambler(seed=42)
        self.embedder = DCTEmbedder(embedding_strength)
        self._amp_sampling_matrix = None

    def load_image_gray(self, path: str) -> np.ndarray:
        img = Image.open(path)
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img, dtype=np.float64)

    def load_image_color(self, path: str) -> np.ndarray:
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float64)

    def compute_relationship_matrix(self, source_blocks: np.ndarray, target_blocks: np.ndarray) -> np.ndarray:
        # User B reconstructs the source from decrypted target measurements:
        # source_blocks ~= target_blocks @ P, therefore Y_source ~= Y_target @ P.
        return np.linalg.pinv(target_blocks.T @ target_blocks) @ target_blocks.T @ source_blocks

    def _get_amp_sampling_matrix(self) -> np.ndarray:
        if self._amp_sampling_matrix is None:
            from .decryptor import AMPReconstructor

            cs_ratio = int(round(self.compression_ratio * 100))
            self._amp_sampling_matrix = AMPReconstructor(cs_ratio=cs_ratio).get_model_sampling_matrix()
        return self._amp_sampling_matrix

    def _encrypt_image(self, image: np.ndarray, mm: MeasurementMatrix):
        X_col, row_new, col_new, block_rows, block_cols = mm.image_to_blocks(image)
        A = self._get_amp_sampling_matrix()
        Y = A @ X_col
        R = mm.generate_block_encryption_matrix_R(Y.shape[1])
        Y_enc = Y @ R
        index = self.scrambler.generate_index(Y_enc.size)
        Y_scr = self.scrambler.scramble(Y_enc, index)
        return X_col, Y_scr, R, index, row_new, col_new, block_rows, block_cols

    def encrypt(self, source_paths: List[str], target_paths: List[str], carrier_path: str, output_dir: str = None) -> dict:
        sources = [self.load_image_gray(p) for p in source_paths]
        targets = [self.load_image_gray(p) for p in target_paths]
        carrier = self.load_image_color(carrier_path)

        height, width = targets[0].shape
        for img in sources + targets:
            if img.shape != (height, width):
                raise ValueError("Scheme 1 requires source and target images to have the same size.")

        mm = MeasurementMatrix(height, width, compression_ratio=self.compression_ratio)

        Y_scrambled = []
        R_matrices = []
        indices = []
        source_Y_scrambled = []
        source_R_matrices = []
        source_indices = []
        P_matrices = []
        block_meta = None

        for s, t in zip(sources, targets):
            source_blocks, source_Y_scr, source_R, source_index, source_row_new, source_col_new, source_block_rows, source_block_cols = self._encrypt_image(s, mm)
            target_blocks, Y_scr, R, index, row_new, col_new, block_rows, block_cols = self._encrypt_image(t, mm)
            if (source_row_new, source_col_new, source_block_rows, source_block_cols) != (row_new, col_new, block_rows, block_cols):
                raise ValueError("Scheme 1 requires matching source/target block metadata.")
            P_matrices.append(self.compute_relationship_matrix(source_blocks, target_blocks))
            Y_scrambled.append(Y_scr)
            R_matrices.append(R)
            indices.append(index)
            source_Y_scrambled.append(source_Y_scr)
            source_R_matrices.append(source_R)
            source_indices.append(source_index)
            block_meta = (row_new, col_new, block_rows, block_cols)

        m_high, m_width = Y_scrambled[0].shape
        embedded_payloads = [
            np.concatenate((target_y, source_y), axis=1)
            for target_y, source_y in zip(Y_scrambled, source_Y_scrambled)
        ]
        M = m_high
        embed_width = embedded_payloads[0].shape[1]
        if carrier.shape[0] < M or carrier.shape[1] < embed_width:
            raise ValueError(
                f"Carrier image is too small for target+source measurements; need at least {M}x{embed_width}, got {carrier.shape[0]}x{carrier.shape[1]}."
            )
        stego, dct_before, dct_after = self.embedder.embed(carrier, embedded_payloads, M, embed_width)
        carrier_dct = DCTEmbedder.get_carrier_dct(carrier)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, s in enumerate(sources):
                Image.fromarray(s.astype(np.uint8)).save(os.path.join(output_dir, f'01_source_image_{i+1}.png'))
            for i, t in enumerate(targets):
                Image.fromarray(t.astype(np.uint8)).save(os.path.join(output_dir, f'04_target_image_{i+1}.png'))
            for i, payload in enumerate(embedded_payloads):
                normalized = (payload - payload.min()) / (payload.max() - payload.min() + 1e-10)
                Image.fromarray((normalized * 255).astype(np.uint8)).save(os.path.join(output_dir, f'07_encrypted_data_{i+1}.png'))
            Image.fromarray(carrier.astype(np.uint8)).save(os.path.join(output_dir, '10_carrier_original.png'))
            Image.fromarray(np.clip(stego, 0, 255).astype(np.uint8)).save(os.path.join(output_dir, '11_carrier_steganographic.png'))

        return {
            'stego_image': stego,
            'carrier_dct': carrier_dct,
            'P_matrices': P_matrices,
            'indices': indices,
            'R_matrices': R_matrices,
            'source_Y_scrambled': source_Y_scrambled,
            'source_indices': source_indices,
            'source_R_matrices': source_R_matrices,
            'height': height,
            'width': width,
            'M': M,
            'measurement_width': m_width,
            'embed_width': embed_width,
            'Y_scrambled': Y_scrambled,
            'block_meta': block_meta,
            'compression_ratio': self.compression_ratio,
        }
