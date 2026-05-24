import os
import glob
import numpy as np
import torch
from torch import nn
from scipy import io

from dataset import dataset
from utils import imread_CS_py, psnr

BLOCK_SIZE = 33
DEFAULT_CS_RATIO = 60
DEFAULT_PHASES = 9
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 1e-4
MODEL_NAME = "AMP_Net_K"


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1, bias=False),
        )

    def forward(self, inputs):
        inputs = torch.unsqueeze(torch.reshape(torch.transpose(inputs, 0, 1), [-1, BLOCK_SIZE, BLOCK_SIZE]), dim=1)
        output = self.D(inputs)
        output = torch.transpose(torch.reshape(torch.squeeze(output), [-1, BLOCK_SIZE * BLOCK_SIZE]), 0, 1)
        return output


class AMPNetBasic(nn.Module):
    def __init__(self, layer_num, A):
        super().__init__()
        self.layer_num = layer_num
        self.denoisers = []
        self.steps = []
        # Keep A and Q on independent storage. np.transpose(A) is a view, and
        # shared storage lets load_state_dict(Q) overwrite A when loading on CPU.
        A_tensor = torch.from_numpy(np.array(A, dtype=np.float32, copy=True))
        Q_tensor = torch.from_numpy(np.array(np.transpose(A), dtype=np.float32, copy=True))
        self.register_parameter("A", nn.Parameter(A_tensor, requires_grad=False))
        self.register_parameter("Q", nn.Parameter(Q_tensor, requires_grad=True))
        for n in range(layer_num):
            self.denoisers.append(Denoiser())
            self.register_parameter(f"step_{n + 1}", nn.Parameter(torch.tensor(1.0), requires_grad=False))
            self.steps.append(getattr(self, f"step_{n + 1}"))
        for n, denoiser in enumerate(self.denoisers):
            self.add_module(f"denoiser_{n + 1}", denoiser)

    def forward(self, inputs, output_layers):
        block_rows = int(inputs.shape[2] / BLOCK_SIZE)
        block_cols = int(inputs.shape[3] / BLOCK_SIZE)
        batch_size = inputs.shape[0]

        y = self.sampling(inputs)
        X = torch.matmul(self.Q, y)
        identity = torch.eye(BLOCK_SIZE * BLOCK_SIZE, device=inputs.device)

        for n in range(output_layers):
            step = self.steps[n]
            denoiser = self.denoisers[n]

            z = self.block1(X, y, step)
            noise = denoiser(X)
            X = z - torch.matmul((step * torch.matmul(torch.transpose(self.A, 0, 1), self.A)) - identity, noise)

            X = self.together(X, batch_size, block_rows, block_cols)
            X = torch.cat(torch.split(X, split_size_or_sections=BLOCK_SIZE, dim=1), dim=0)
            X = torch.cat(torch.split(X, split_size_or_sections=BLOCK_SIZE, dim=2), dim=0)
            X = torch.transpose(torch.reshape(X, [-1, BLOCK_SIZE * BLOCK_SIZE]), 0, 1)

        X = self.together(X, batch_size, block_rows, block_cols)
        return torch.unsqueeze(X, dim=1)

    def sampling(self, inputs):
        inputs = torch.squeeze(inputs, dim=1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=BLOCK_SIZE, dim=1), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=BLOCK_SIZE, dim=2), dim=0)
        inputs = torch.transpose(torch.reshape(inputs, [-1, BLOCK_SIZE * BLOCK_SIZE]), 0, 1)
        outputs = torch.matmul(self.A, inputs)
        return outputs

    def block1(self, X, y, step):
        outputs = torch.matmul(torch.transpose(self.A, 0, 1), y - torch.matmul(self.A, X))
        outputs = step * outputs + X
        return outputs

    def together(self, inputs, batch_size, block_rows, block_cols):
        inputs = torch.reshape(torch.transpose(inputs, 0, 1), [-1, BLOCK_SIZE, BLOCK_SIZE])
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=block_rows * batch_size, dim=0), dim=2)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=batch_size, dim=0), dim=1)
        return inputs


def _net_root():
    return os.path.dirname(os.path.abspath(__file__))


def _workspace_root():
    return os.path.dirname(_net_root())


def sampling_matrix_dir():
    candidates = [
        os.path.join(_workspace_root(), "dataset", "sampling_matrix"),
        os.path.join(_net_root(), "dataset", "sampling_matrix"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def load_sampling_matrix(cs_ratio):
    path = sampling_matrix_dir()
    file_path = os.path.join(path, f"{cs_ratio}.mat")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"sampling matrix not found: {file_path}")
    return io.loadmat(file_path)["sampling_matrix"]


def results_dir():
    path = os.path.join(_workspace_root(), "results", MODEL_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def train_epoch(model, optimizer, train_loader, epoch, cs_ratio, phases, device):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        data = torch.unsqueeze(data, dim=1).float().to(device)
        outputs = model(data, phases)
        loss = torch.mean((outputs - data) ** 2)
        loss.backward()
        optimizer.step()

        if batch_idx % 25 == 0:
            print(f"CS_ratio: {cs_ratio} epoch: {epoch:03d} step: {batch_idx:04d} loss: {loss.item():.6f}")


def evaluate(model, phases, device):
    model.eval()
    test_dir_candidates = [
        os.path.join(_workspace_root(), "dataset", "Set11"),
        os.path.join(_workspace_root(), "dataset", "bsds500", "test"),
        os.path.join(_net_root(), "dataset", "Set11"),
        os.path.join(_net_root(), "dataset", "bsds500", "test"),
    ]
    test_dir = next((d for d in test_dir_candidates if os.path.isdir(d)), None)
    if test_dir is None:
        raise FileNotFoundError("No evaluation dataset found in dataset/Set11 or dataset/bsds500/test")

    paths = sorted(glob.glob(os.path.join(test_dir, "*.tif")))
    if not paths:
        raise FileNotFoundError(f"No .tif files found in {test_dir}")

    psnrs = []
    with torch.no_grad():
        for img_path in paths:
            Iorg, row, col, Ipad, row_new, col_new = imread_CS_py(img_path)
            Ipad = Ipad / 255.0
            inputs = torch.from_numpy(Ipad.astype("float32")).to(device)
            inputs = torch.unsqueeze(torch.unsqueeze(inputs, dim=0), dim=0)
            outputs = model(inputs, phases)
            outputs = torch.squeeze(outputs).detach().cpu().numpy()
            images_recovered = outputs[0:row, 0:col] * 255.0
            psnrs.append(psnr(images_recovered, Iorg))
    return float(np.mean(psnrs))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cs_ratio = DEFAULT_CS_RATIO
    phases = DEFAULT_PHASES
    epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE
    learning_rate = DEFAULT_LEARNING_RATE

    print(f"Using device: {device}")
    print(f"Training {MODEL_NAME} with CS_ratio={cs_ratio}, phases={phases}")

    data_root_candidates = [os.path.join(_workspace_root(), "dataset"), os.path.join(_net_root(), "dataset")]
    data_root = next((d for d in data_root_candidates if os.path.isdir(d)), data_root_candidates[0])
    train_set = dataset(root=data_root)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    A = load_sampling_matrix(cs_ratio)
    model = AMPNetBasic(phases, A).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    save_dir = ensure_dir(os.path.join(results_dir(), str(cs_ratio), str(phases)))
    best_path = os.path.join(save_dir, "best_model.pkl")
    best_psnr = float("-inf")

    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, train_loader, epoch, cs_ratio, phases, device)
        val_psnr = evaluate(model, phases, device)
        print(f"Epoch {epoch:03d}: val PSNR = {val_psnr:.4f} dB")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    print(f"Training complete. Best PSNR: {best_psnr:.4f} dB")


if __name__ == "__main__":
    main()
