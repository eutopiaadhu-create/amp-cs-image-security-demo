import os
import glob
import numpy as np
import torch
from scipy import io
from skimage.io import imsave

from train_AMP_Net import AMPNetBasic, DEFAULT_CS_RATIO, DEFAULT_PHASES, MODEL_NAME
from utils import imread_CS_py, psnr, compute_ssim


def _net_root():
    return os.path.dirname(os.path.abspath(__file__))


def _workspace_root():
    return os.path.dirname(_net_root())


def _sampling_matrix_path(cs_ratio):
    candidates = [
        os.path.join(_workspace_root(), "dataset", "sampling_matrix", f"{cs_ratio}.mat"),
        os.path.join(_net_root(), "dataset", "sampling_matrix", f"{cs_ratio}.mat"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def load_sampling_matrix(cs_ratio):
    path = _sampling_matrix_path(cs_ratio)
    if not os.path.exists(path):
        raise FileNotFoundError(f"sampling matrix not found: {path}")
    return io.loadmat(path)["sampling_matrix"]


def resolve_model_path(cs_ratio, phases):
    candidates = [
        os.path.join(_workspace_root(), "models", "best_model (1).pkl"),
        os.path.join(_workspace_root(), "results", MODEL_NAME, str(cs_ratio), str(phases), "best_model.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def resolve_test_dir():
    candidates = [
        os.path.join(_workspace_root(), "dataset", "Set11"),
        os.path.join(_workspace_root(), "dataset", "bsds500", "test"),
        os.path.join(_net_root(), "dataset", "Set11"),
        os.path.join(_net_root(), "dataset", "bsds500", "test"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError("No test dataset found in dataset/Set11 or dataset/bsds500/test")


def evaluate(model, phases, save_dir, device):
    paths = sorted(glob.glob(os.path.join(resolve_test_dir(), "*.tif")))
    if not paths:
        raise FileNotFoundError("No .tif test images found")

    os.makedirs(save_dir, exist_ok=True)
    psnrs = []
    ssims = []

    model.eval()
    with torch.no_grad():
        for img_path in paths:
            Iorg, row, col, Ipad, row_new, col_new = imread_CS_py(img_path)
            Ipad = Ipad / 255.0
            inputs = torch.from_numpy(Ipad.astype("float32")).to(device)
            inputs = torch.unsqueeze(torch.unsqueeze(inputs, dim=0), dim=0)
            outputs = model(inputs, phases)
            output = torch.squeeze(outputs).detach().cpu().numpy()
            recovered = output[0:row, 0:col] * 255.0
            recovered = np.clip(recovered, 0, 255).astype(np.uint8)

            rec_psnr = psnr(recovered, Iorg)
            rec_ssim = compute_ssim(recovered, Iorg)
            psnrs.append(rec_psnr)
            ssims.append(rec_ssim)

            name = os.path.splitext(os.path.basename(img_path))[0]
            imsave(os.path.join(save_dir, f"{name}_{rec_psnr:.4f}_{rec_ssim:.4f}.png"), recovered)

    return float(np.mean(psnrs)), float(np.mean(ssims))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cs_ratio = DEFAULT_CS_RATIO
    phases = DEFAULT_PHASES

    model_path = resolve_model_path(cs_ratio, phases)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model weights not found: {model_path}")

    A = load_sampling_matrix(cs_ratio)
    model = AMPNetBasic(phases, A).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    save_dir = os.path.join(_workspace_root(), "results", MODEL_NAME, str(cs_ratio), str(phases), "generated_images")
    psnr_val, ssim_val = evaluate(model, phases, save_dir, device)
    print(f"CS_ratio={cs_ratio}, phases={phases}, PSNR={psnr_val:.4f} dB, SSIM={ssim_val:.4f}")


if __name__ == "__main__":
    main()
