"""
AMP-based reconstruction for compressive sensing decryption.
"""

import os
import sys
import numpy as np
from typing import List
from PIL import Image
from scipy import io

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_DIR = os.path.join(WORKSPACE_ROOT, "net")
if NET_DIR not in sys.path:
    sys.path.insert(0, NET_DIR)

from train_AMP_Net import AMPNetBasic, BLOCK_SIZE, DEFAULT_CS_RATIO, DEFAULT_PHASES
from cs_crypto.scramble import IndexScrambler
from cs_crypto.embedding import DCTEmbedder
from ai_modules.face_detector import FaceDetector
from ai_modules.roi_analyzer import ROIAnalyzer
from ai_modules.privacy_protector import PrivacyProtector
from core.permission import UserLevel

import torch


class AMPReconstructor:
    def __init__(self, cs_ratio: int = DEFAULT_CS_RATIO, phases: int = DEFAULT_PHASES):
        self.cs_ratio = cs_ratio
        self.phases = phases
        self._model = None

    def _workspace_root(self):
        return WORKSPACE_ROOT

    def _load_sampling_matrix(self):
        path = os.path.join(self._workspace_root(), "dataset", "sampling_matrix", f"{self.cs_ratio}.mat")
        if not os.path.exists(path):
            raise FileNotFoundError(f"采样矩阵未找到: {path}")
        return io.loadmat(path)["sampling_matrix"]

    def _load_model(self):
        if self._model is not None:
            return self._model
        model_path = os.path.join(self._workspace_root(), "models", "best_model (1).pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"AMP 权重未找到: {model_path}")

        A = self._load_sampling_matrix().astype(np.float32)
        model = AMPNetBasic(self.phases, A)
        state_dict = torch.load(
            model_path,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        self._model = model
        return self._model

    def get_model_sampling_matrix(self) -> np.ndarray:
        model = self._load_model()
        return model.A.detach().cpu().numpy().astype(np.float32)

    def reconstruct(self, measurements: np.ndarray, block_rows: int, block_cols: int, row_new: int, col_new: int, height: int, width: int) -> np.ndarray:
        model = self._load_model()
        device = next(model.parameters()).device
        with torch.no_grad():
            y = torch.from_numpy(measurements.astype(np.float32)).to(device)
            X = torch.matmul(model.Q, y)
            identity = torch.eye(BLOCK_SIZE * BLOCK_SIZE, device=device)

            for n in range(self.phases):
                step = model.steps[n]
                denoiser = model.denoisers[n]
                z = torch.matmul(torch.transpose(model.A, 0, 1), y - torch.matmul(model.A, X))
                z = step * z + X
                noise = denoiser(X)
                X = z - torch.matmul((step * torch.matmul(torch.transpose(model.A, 0, 1), model.A)) - identity, noise)
                X = model.together(X, 1, block_rows, block_cols)
                X = torch.cat(torch.split(X, split_size_or_sections=BLOCK_SIZE, dim=1), dim=0)
                X = torch.cat(torch.split(X, split_size_or_sections=BLOCK_SIZE, dim=2), dim=0)
                X = torch.transpose(torch.reshape(X, [-1, BLOCK_SIZE * BLOCK_SIZE]), 0, 1)

            X = model.together(X, 1, block_rows, block_cols)
            image = torch.squeeze(X).detach().cpu().numpy()
        image = image[:height, :width] * 255.0
        return np.clip(image, 0, 255).astype(np.float64)


class UserAReconstructor:
    def __init__(self):
        self.amp = AMPReconstructor()

    def decrypt(self, y_encrypted, R, index, block_rows, block_cols, row_new, col_new, height, width, scrambler):
        y = scrambler.unscramble(y_encrypted, index)
        y = y @ np.linalg.pinv(R)
        return self.amp.reconstruct(y, block_rows, block_cols, row_new, col_new, height, width)


class UserBReconstructor:
    def __init__(self):
        self.amp = AMPReconstructor()

    def decrypt(self, y_encrypted, R, P, index, block_rows, block_cols, row_new, col_new, height, width, scrambler):
        y = scrambler.unscramble(y_encrypted, index)
        y = y @ np.linalg.pinv(R)
        y = y @ P
        return self.amp.reconstruct(y, block_rows, block_cols, row_new, col_new, height, width)


class Decryptor:
    def __init__(self, embedding_strength: float = 0.001):
        self.embedding_strength = embedding_strength
        self.scrambler = IndexScrambler(seed=42)
        self.embedder = DCTEmbedder(embedding_strength)
        self.face_detector = FaceDetector()
        self.roi_analyzer = ROIAnalyzer()
        self.privacy_protector = PrivacyProtector(method='mosaic', mosaic_size=10)

    def load_ai_model(self, model_path: str) -> bool:
        return self.face_detector.load(model_path)

    def decrypt_user_a(self, encryption_result: dict, output_dir: str = None) -> List[np.ndarray]:
        height = encryption_result['height']
        width = encryption_result['width']
        Y_scrambled = encryption_result['Y_scrambled']
        R_matrices = encryption_result['R_matrices']
        indices = encryption_result['indices']
        row_new, col_new, block_rows, block_cols = encryption_result['block_meta']

        reconstructor = UserAReconstructor()
        recovered = []
        for i, (Y_scr, R, index) in enumerate(zip(Y_scrambled, R_matrices, indices)):
            print(f"  [UserA] Reconstructing target image {i+1}/3...")
            img = reconstructor.decrypt(Y_scr, R, index, block_rows, block_cols, row_new, col_new, height, width, self.scrambler)
            recovered.append(img)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, img in enumerate(recovered):
                Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(output_dir, f'recovered_A_{i+1}.png'))
        return recovered

    def decrypt_user_b(self, encryption_result: dict, output_dir: str = None) -> dict:
        height = encryption_result['height']
        width = encryption_result['width']
        Y_scrambled = encryption_result['Y_scrambled']
        R_matrices = encryption_result['R_matrices']
        indices = encryption_result['indices']
        P_matrices = encryption_result.get('P_matrices', [])
        source_Y_scrambled = encryption_result.get('source_Y_scrambled')
        source_R_matrices = encryption_result.get('source_R_matrices')
        source_indices = encryption_result.get('source_indices')
        row_new, col_new, block_rows, block_cols = encryption_result['block_meta']

        reconstructor_a = UserAReconstructor()
        targets = []
        for i, (Y_scr, R, index) in enumerate(zip(Y_scrambled, R_matrices, indices)):
            print(f"  [UserB] Reconstructing target image {i+1}/3...")
            img = reconstructor_a.decrypt(Y_scr, R, index, block_rows, block_cols, row_new, col_new, height, width, self.scrambler)
            targets.append(img)

        sources = []
        if source_Y_scrambled is not None and source_R_matrices is not None and source_indices is not None:
            for i, (Y_scr, R, index) in enumerate(zip(source_Y_scrambled, source_R_matrices, source_indices)):
                print(f"  [UserB] Reconstructing source image {i+1}/3 from direct measurements...")
                img = reconstructor_a.decrypt(Y_scr, R, index, block_rows, block_cols, row_new, col_new, height, width, self.scrambler)
                sources.append(img)
        else:
            reconstructor_b = UserBReconstructor()
            for i, (Y_scr, R, P, index) in enumerate(zip(Y_scrambled, R_matrices, P_matrices, indices)):
                print(f"  [UserB] Reconstructing source image {i+1}/3 from P mapping...")
                img = reconstructor_b.decrypt(Y_scr, R, P, index, block_rows, block_cols, row_new, col_new, height, width, self.scrambler)
                sources.append(img)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, img in enumerate(sources):
                Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(output_dir, f'recovered_B_source_{i+1}.png'))
            for i, img in enumerate(targets):
                Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(output_dir, f'recovered_B_target_{i+1}.png'))
        return {'sources': sources, 'targets': targets}

    def apply_privacy_protection(self, image: np.ndarray, original_path: str = None, user_level: UserLevel = UserLevel.USER_A) -> np.ndarray:
        if user_level.value >= UserLevel.USER_B.value:
            return image
        if not self.face_detector.is_loaded:
            return image
        if original_path and os.path.exists(original_path):
            detections = self.face_detector.detect_with_scaling(original_path, image.shape[:2])
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
        original = original.astype(np.float64)
        recovered = recovered.astype(np.float64)
        mse = np.mean((original - recovered) ** 2)
        psnr = float('inf') if mse == 0 else 10 * np.log10(255.0 * 255.0 / mse)
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_val = ssim(
                np.clip(original, 0, 255).astype(np.uint8),
                np.clip(recovered, 0, 255).astype(np.uint8),
                data_range=255,
            )
        except ImportError:
            ssim_val = self._simple_ssim(original, recovered)
        return {'psnr': psnr, 'ssim': ssim_val}

    def _simple_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
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
