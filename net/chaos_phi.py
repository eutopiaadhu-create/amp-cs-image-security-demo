import numpy as np
from scipy.linalg import orth
from scipy import io
import os

def generate_chaos_phi_block(block_size=33, cr=0.25):
    """
    为33x33分块生成混沌测量矩阵
    block_size: 块大小 (33)
    cr: 压缩率 (0.25 对应 CS_ratio=25)
    返回: A矩阵, shape=(M, 33*33), M=round(cr*33)
    """
    N = block_size * block_size  # 1089
    M = round(cr * N)  # 272 for cr=0.25
    
    T = N
    x = np.zeros(T)
    y = np.zeros(T)
    z = np.zeros(T)
    
    x[0], y[0], z[0] = 0.3, 0.4, 0.5
    a, w = 10, 3
    
    for i in range(1, T):
        x[i] = y[i-1] - z[i-1]
        y[i] = np.sin(np.pi * x[i-1] - a * y[i-1])
        z_prev = np.clip(z[i-1], -1.0, 1.0)
        z[i] = np.cos(w * np.arccos(z_prev) + y[i-1])
    z_matrix = z.reshape((block_size, block_size), order='F')
    Phi_full = orth(z_matrix.flatten().reshape(-1, 1) @ np.ones((1, N)))
    
    # 用SVD生成正交测量矩阵
    np.random.seed(42)  # 固定种子保证可复现
    random_matrix = np.random.randn(N, N)
    for i in range(N):
        random_matrix[i, :] *= z[i]  # 用混沌序列调制
    
    U, _, _ = np.linalg.svd(random_matrix)
    A = U[:M, :]  # shape: (M, N)
    
    return A.astype(np.float32)


if __name__ == "__main__":
    # 生成不同压缩率的测量矩阵
    os.makedirs("dataset/sampling_matrix", exist_ok=True)
    
    for cs_ratio in [50,60]:
        cr = cs_ratio / 100.0
        A = generate_chaos_phi_block(33, cr)
        
        # 保存为.mat格式（与原代码兼容）
        save_path = f"dataset/sampling_matrix/{cs_ratio}.mat"
        io.savemat(save_path, {'sampling_matrix': A})
        print(f"Saved {save_path}, shape: {A.shape}")