import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
NET_DIR = os.path.join(ROOT, "net")
if NET_DIR not in sys.path:
    sys.path.insert(0, NET_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.decryptor import AMPReconstructor, Decryptor
from cs_crypto.measurement import MeasurementMatrix
from cs_crypto.scramble import IndexScrambler
from train_AMP_Net import AMPNetBasic, DEFAULT_CS_RATIO, DEFAULT_PHASES

BLOCK_SIZE = 33


def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float64)


def load_model_and_matrix(cs_ratio: int, phases: int, device: str):
    amp = AMPReconstructor(cs_ratio=cs_ratio, phases=phases)
    A = amp._load_sampling_matrix().astype(np.float32)
    model = AMPNetBasic(phases, A).to(device)
    model_path = os.path.join(ROOT, "models", "best_model (1).pkl")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return amp, model, A


def summarize(name: str, arr: np.ndarray):
    arr = np.asarray(arr)
    print(f"[{name}] shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, std={arr.std():.4f}")


def reconstruct_like_training(model, measurements: np.ndarray, phases: int, block_rows: int, block_cols: int, height: int, width: int, device: str) -> np.ndarray:
    with torch.no_grad():
        y = torch.from_numpy(measurements.astype(np.float32)).to(device)
        X = torch.matmul(model.Q, y)
        identity = torch.eye(BLOCK_SIZE * BLOCK_SIZE, device=device)

        for n in range(phases):
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


def main():
    parser = argparse.ArgumentParser(description="AMP local self check")
    parser.add_argument("--target", required=True, help="target gray image path")
    parser.add_argument("--source", help="optional source gray image path")
    parser.add_argument("--cs-ratio", type=int, default=DEFAULT_CS_RATIO)
    parser.add_argument("--phases", type=int, default=DEFAULT_PHASES)
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "output", "amp_self_check"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    target = load_gray(args.target)
    summarize("target", target)
    Image.fromarray(np.clip(target, 0, 255).astype(np.uint8)).save(os.path.join(args.output_dir, "00_target.png"))

    amp, model, A = load_model_and_matrix(args.cs_ratio, args.phases, device)
    amp_model = amp._load_model()
    decryptor = Decryptor()
    mm = MeasurementMatrix(target.shape[0], target.shape[1], compression_ratio=args.cs_ratio / 100.0)

    print("check A consistency")
    model_A = model.A.detach().cpu().numpy().astype(np.float32)
    amp_model_A = amp_model.A.detach().cpu().numpy().astype(np.float32)
    print("A diff L1", float(np.mean(np.abs(model_A - A))))
    print("model vs amp_model A diff L1", float(np.mean(np.abs(model_A - amp_model_A))))
    print("model vs amp_model Q diff L1", float(np.mean(np.abs(model.Q.detach().cpu().numpy() - amp_model.Q.detach().cpu().numpy()))))

    print("step1: direct AMP from image")
    X_col, row_new, col_new, block_rows, block_cols = mm.image_to_blocks(target)
    padded_target, _, _ = mm.pad_image(target)
    inputs = torch.from_numpy((padded_target / 255.0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out1 = model(inputs, args.phases)
    rec1 = torch.squeeze(out1).detach().cpu().numpy()
    rec1 = np.clip(rec1[:target.shape[0], :target.shape[1]] * 255.0, 0, 255)
    summarize("stage1_rec", rec1)
    print("stage1 metrics", decryptor.compute_metrics(target, rec1))
    Image.fromarray(rec1.astype(np.uint8)).save(os.path.join(args.output_dir, "01_stage1_direct_amp.png"))

    print("step2: measurement -> AMP")
    y = model_A @ X_col
    summarize("measurement_y", y)
    with torch.no_grad():
        internal_y = model.sampling(inputs)
    internal_y_np = internal_y.detach().cpu().numpy()
    summarize("internal_y", internal_y_np)
    print("measurement diff L1", float(np.mean(np.abs(internal_y_np - y))))

    rec2 = amp.reconstruct(y, block_rows, block_cols, row_new, col_new, target.shape[0], target.shape[1])
    summarize("stage2_rec", rec2)
    print("stage2 metrics", decryptor.compute_metrics(target, rec2))

    rec2b = amp.reconstruct(internal_y_np, block_rows, block_cols, row_new, col_new, target.shape[0], target.shape[1])
    summarize("stage2b_rec", rec2b)
    print("stage2b metrics", decryptor.compute_metrics(target, rec2b))

    rec2c = reconstruct_like_training(model, internal_y_np, args.phases, block_rows, block_cols, target.shape[0], target.shape[1], device)
    summarize("stage2c_rec", rec2c)
    print("stage2c metrics", decryptor.compute_metrics(target, rec2c))

    rec2d = reconstruct_like_training(amp_model, internal_y_np, args.phases, block_rows, block_cols, target.shape[0], target.shape[1], device)
    summarize("stage2d_rec", rec2d)
    print("stage2d metrics", decryptor.compute_metrics(target, rec2d))

    Image.fromarray(np.clip(rec2, 0, 255).astype(np.uint8)).save(os.path.join(args.output_dir, "02_stage2_measurement_amp.png"))
    Image.fromarray(np.clip(rec2b, 0, 255).astype(np.uint8)).save(os.path.join(args.output_dir, "02b_stage2_internal_y_amp.png"))
    Image.fromarray(np.clip(rec2c, 0, 255).astype(np.uint8)).save(os.path.join(args.output_dir, "02c_stage2_training_like_amp.png"))
    Image.fromarray(np.clip(rec2d, 0, 255).astype(np.uint8)).save(os.path.join(args.output_dir, "02d_stage2_ampmodel_training_like_amp.png"))

    print("step3: measurement + scramble + inverse")
    scrambler = IndexScrambler(seed=42)
    R = mm.generate_block_encryption_matrix_R(y.shape[1])
    idx = scrambler.generate_index(y.size)
    y_scr = scrambler.scramble(y @ R, idx)
    y_uns = scrambler.unscramble(y_scr, idx) @ np.linalg.pinv(R)
    summarize("y_scr", y_scr)
    summarize("y_uns", y_uns)
    rec3 = amp.reconstruct(y_uns, block_rows, block_cols, row_new, col_new, target.shape[0], target.shape[1])
    summarize("stage3_rec", rec3)
    print("stage3 metrics", decryptor.compute_metrics(target, rec3))
    Image.fromarray(np.clip(rec3, 0, 255).astype(np.uint8)).save(os.path.join(args.output_dir, "03_stage3_scramble_amp.png"))

    if args.source:
        source = load_gray(args.source)
        summarize("source", source)
        Image.fromarray(np.clip(source, 0, 255).astype(np.uint8)).save(os.path.join(args.output_dir, "99_source.png"))

    print(f"done, outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
