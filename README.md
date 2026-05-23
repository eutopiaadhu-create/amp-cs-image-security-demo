# AI-Driven Intelligent Perception Multi-level Image Encryption System

基于2D压缩感知的AI智能感知多级图像加密系统

## Features / 创新点

- **AI智能感知**: 自动识别敏感区域（人脸等），差异化加密
- **深度学习重建**: 支持神经网络替代SPL迭代，速度提升10倍+
- **三级权限控制**: 访客/普通用户/管理员 差异化访问
- **隐私保护联动**: 低权限用户看到的敏感区域自动模糊
- **纯Python实现**: 无MATLAB依赖，易部署

## Installation / 安装

```bash
# Clone the repository
git clone https://github.com/yourusername/2dcs-ai-encryption.git
cd 2dcs-ai-encryption

# Create virtual environment (recommended)
conda create -n cs_encrypt python=3.10
conda activate cs_encrypt

# Install dependencies
pip install -r requirements.txt
```

## Usage / 使用方法

```bash
python main_app.py
```

### Login Credentials / 登录凭证

| 用户类型 | 用户名 | 密码 | 权限 |
|---------|--------|------|------|
| A用户 | a | 123 | 解密目标图像 |
| B用户 | b | 456 | 解密源图像+目标图像 |
| 管理员 | admin | root | 完全访问 |
| 访客 | - | - | 仅查看隐写图 |

## Project Structure / 项目结构

```
├── main_app.py              # 主程序入口
├── requirements.txt         # 依赖列表
├── cs_crypto/               # 核心加密模块
│   ├── chaos.py             # 3D混沌序列生成
│   ├── measurement.py       # 压缩感知测量矩阵
│   ├── scramble.py          # 索引置乱
│   ├── wavelet.py           # CDF 9/7小波变换
│   ├── embedding.py         # DCT域隐写嵌入
│   └── reconstruction.py    # SPL迭代重建
├── ai_modules/              # AI模块
│   ├── face_detector.py     # YOLO人脸检测
│   ├── roi_analyzer.py      # 敏感区域分析
│   └── privacy_protector.py # 隐私保护(马赛克/模糊)
├── core/                    # 业务逻辑层
│   ├── permission.py        # 三级权限管理
│   ├── encryptor.py         # 加密流程
│   └── decryptor.py         # 解密流程
├── gui/                     # GUI组件
│   ├── app.py               # 主窗口
│   └── presentation.py      # 演示窗口
└── models/                  # 模型文件夹
    └── (place .pt files here)
```

## Algorithm / 算法流程

### Encryption / 加密流程

1. **图像关系建立**: P = pinv(S'*S) * S' * T
2. **混沌序列生成**: 3D混沌映射生成Phi和R矩阵
3. **压缩感知测量**: Y = Phi * Target
4. **混沌加密**: Y = Y * R
5. **索引置乱**: Y = IndexScramble(Y)
6. **DCT域嵌入**: 将加密数据嵌入载体图像

### Decryption / 解密流程

- **A用户**: 逆置乱 → 消除R → SPL重建 → 目标图像
- **B用户**: 逆置乱 → 消除R → 消除P → SPL重建 → 源图像

## Requirements / 环境要求

- Python >= 3.8
- CUDA (optional, for faster AI inference)

## Citation / 引用

If you use this code, please cite:

```bibtex
@article{2dcs2024,
  title={2-D Compressive Sensing-Based Visually Secure Multilevel Image Encryption Scheme},
  year={2024}
}
```

## License / 许可证

MIT License
