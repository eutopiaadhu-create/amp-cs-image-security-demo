"""
Generate presentation-ready experiment artifacts for the AMP-CS demo.

Outputs:
- comparison_grid.png: visual comparison figure for slides.
- experiment_summary.csv / experiment_summary.md: metrics table.
- wrong_key_target_*.png: wrong-index reconstruction examples.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
NET_DIR = ROOT / "net"
if str(NET_DIR) not in sys.path:
    sys.path.insert(0, str(NET_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.encryptor import Encryptor
from core.decryptor import Decryptor, AMPReconstructor
from cs_crypto.scramble import IndexScrambler


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float64)


def save_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def psnr_ssim(original: np.ndarray, recovered: np.ndarray) -> dict:
    original = original.astype(np.float64)
    recovered = recovered.astype(np.float64)
    mse = np.mean((original - recovered) ** 2)
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 * 255.0 / mse)
    try:
        from skimage.metrics import structural_similarity as ssim

        if original.ndim == 3:
            ssim_val = ssim(
                np.clip(original, 0, 255).astype(np.uint8),
                np.clip(recovered, 0, 255).astype(np.uint8),
                channel_axis=2,
                data_range=255,
            )
        else:
            ssim_val = ssim(
                np.clip(original, 0, 255).astype(np.uint8),
                np.clip(recovered, 0, 255).astype(np.uint8),
                data_range=255,
            )
    except Exception:
        ssim_val = float("nan")
    return {"psnr": float(psnr), "ssim": float(ssim_val)}


def avg_metric(metrics: list[dict], key: str) -> float:
    vals = [m[key] for m in metrics if m and np.isfinite(m[key])]
    return float(np.mean(vals)) if vals else float("nan")


def default_input_dir() -> Path:
    return Path.home() / "Desktop" / "2DCS_AI_Output"


def resolve_inputs(input_dir: Path) -> tuple[list[Path], list[Path], Path]:
    sources = [input_dir / f"01_source_image_{i}.png" for i in range(1, 4)]
    targets = [input_dir / f"04_target_image_{i}.png" for i in range(1, 4)]
    carrier = input_dir / "10_carrier_original.png"
    missing = [p for p in [*sources, *targets, carrier] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing demo input files:\n" + "\n".join(str(p) for p in missing))
    return sources, targets, carrier


def wrong_key_reconstructions(result: dict, target_paths: list[Path], output_dir: Path) -> list[dict]:
    amp = AMPReconstructor()
    wrong_scrambler = IndexScrambler(seed=999)
    metrics = []
    height = result["height"]
    width = result["width"]
    row_new, col_new, block_rows, block_cols = result["block_meta"]

    for i, (y_scr, r_matrix, target_path) in enumerate(
        zip(result["Y_scrambled"], result["R_matrices"], target_paths), start=1
    ):
        wrong_index = wrong_scrambler.generate_index(y_scr.size)
        y_wrong = wrong_scrambler.unscramble(y_scr, wrong_index) @ np.linalg.pinv(r_matrix)
        rec_wrong = amp.reconstruct(y_wrong, block_rows, block_cols, row_new, col_new, height, width)
        out_path = output_dir / f"wrong_key_target_{i}.png"
        save_gray(out_path, rec_wrong)
        metrics.append(psnr_ssim(load_gray(target_path), rec_wrong))
    return metrics


def make_noise_cipher(reference_path: Path, output_path: Path) -> None:
    img = Image.open(reference_path).convert("L")
    rng = np.random.default_rng(20240524)
    noise = rng.integers(0, 256, size=(img.height, img.width), dtype=np.uint8)
    Image.fromarray(noise).save(output_path)


def find_font(size: int):
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def tile_image(path: Path, title: str, subtitle: str, size=(320, 260)) -> Image.Image:
    font_title = find_font(20)
    font_sub = find_font(14)
    card_w, card_h = size
    canvas = Image.new("RGB", (card_w, card_h), "#111827")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=18, outline="#334155", width=2, fill="#111827")

    img = Image.open(path).convert("RGB")
    preview = ImageOps.contain(img, (card_w - 36, card_h - 82), method=Image.Resampling.LANCZOS)
    x = (card_w - preview.width) // 2
    y = 58 + (card_h - 82 - preview.height) // 2
    canvas.paste(preview, (x, y))

    draw.text((18, 14), title, fill="#e5edf7", font=font_title)
    draw.text((18, 38), subtitle, fill="#8fb3d9", font=font_sub)
    return canvas


def make_comparison_grid(output_dir: Path) -> None:
    items = [
        ("原始载体", "普通可见图像", output_dir / "10_carrier_original.png"),
        ("传统密文示意", "噪声外观，容易引起注意", output_dir / "traditional_cipher_noise.png"),
        ("本系统输出", "隐写载体，视觉正常", output_dir / "11_carrier_steganographic.png"),
        ("原始目标图", "待保护图像", output_dir / "04_target_image_1.png"),
        ("正确密钥恢复", "A/B用户可重建", output_dir / "recovered_B_target_1.png"),
        ("错误密钥恢复", "索引错误，无法恢复", output_dir / "wrong_key_target_1.png"),
    ]
    tiles = [tile_image(path, title, sub) for title, sub, path in items]
    gap = 18
    grid_w = 3 * 320 + 4 * gap
    grid_h = 2 * 260 + 3 * gap + 46
    grid = Image.new("RGB", (grid_w, grid_h), "#0b1118")
    draw = ImageDraw.Draw(grid)
    draw.text((gap, 14), "传统加密外观 vs 本系统隐写输出", fill="#f8fafc", font=find_font(24))
    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, 3)
        x = gap + col * (320 + gap)
        y = 58 + row * (260 + gap)
        grid.paste(tile, (x, y))
    grid.save(output_dir / "comparison_grid.png")


def write_tables(output_dir: Path, rows: list[dict]) -> None:
    csv_path = output_dir / "experiment_summary.csv"
    md_path = output_dir / "experiment_summary.md"
    fields = ["实验项", "指标", "结果", "说明"]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| 实验项 | 指标 | 结果 | 说明 |\n")
        f.write("|---|---|---|---|\n")
        for row in rows:
            f.write(f"| {row['实验项']} | {row['指标']} | {row['结果']} | {row['说明']} |\n")


def run(args) -> Path:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources, targets, carrier = resolve_inputs(input_dir)
    encryptor = Encryptor(compression_ratio=args.compression_ratio)
    decryptor = Decryptor()

    t0 = time.perf_counter()
    enc_start = time.perf_counter()
    result = encryptor.encrypt([str(p) for p in sources], [str(p) for p in targets], str(carrier), str(output_dir))
    enc_time = time.perf_counter() - enc_start

    a_start = time.perf_counter()
    decryptor.decrypt_user_a(result, str(output_dir))
    a_time = time.perf_counter() - a_start

    b_start = time.perf_counter()
    decryptor.decrypt_user_b(result, str(output_dir))
    b_time = time.perf_counter() - b_start

    wrong_start = time.perf_counter()
    wrong_metrics = wrong_key_reconstructions(result, targets, output_dir)
    wrong_time = time.perf_counter() - wrong_start
    total_time = time.perf_counter() - t0

    make_noise_cipher(targets[0], output_dir / "traditional_cipher_noise.png")
    make_comparison_grid(output_dir)

    source_metrics = [
        psnr_ssim(load_gray(src), load_gray(output_dir / f"recovered_B_source_{i}.png"))
        for i, src in enumerate(sources, start=1)
    ]
    target_metrics = [
        psnr_ssim(load_gray(tgt), load_gray(output_dir / f"recovered_B_target_{i}.png"))
        for i, tgt in enumerate(targets, start=1)
    ]
    carrier_metrics = psnr_ssim(load_rgb(output_dir / "10_carrier_original.png"), load_rgb(output_dir / "11_carrier_steganographic.png"))

    rows = [
        {
            "实验项": "目标图重建质量",
            "指标": "PSNR / SSIM",
            "结果": f"{avg_metric(target_metrics, 'psnr'):.2f} dB / {avg_metric(target_metrics, 'ssim'):.4f}",
            "说明": "正确密钥下AMP重建目标图。",
        },
        {
            "实验项": "源图重建质量",
            "指标": "PSNR / SSIM",
            "结果": f"{avg_metric(source_metrics, 'psnr'):.2f} dB / {avg_metric(source_metrics, 'ssim'):.4f}",
            "说明": "方案1：B用户使用源图真实measurement重建。",
        },
        {
            "实验项": "隐写载体不可感知性",
            "指标": "Carrier PSNR / SSIM",
            "结果": f"{carrier_metrics['psnr']:.2f} dB / {carrier_metrics['ssim']:.4f}",
            "说明": "原始载体与隐写载体的差异。",
        },
        {
            "实验项": "错误密钥敏感性",
            "指标": "Wrong-key PSNR / SSIM",
            "结果": f"{avg_metric(wrong_metrics, 'psnr'):.2f} dB / {avg_metric(wrong_metrics, 'ssim'):.4f}",
            "说明": "使用错误置乱索引时无法正确恢复。",
        },
        {
            "实验项": "运行效率",
            "指标": "总耗时 / 加密 / A重建 / B重建",
            "结果": f"{total_time:.2f}s / {enc_time:.2f}s / {a_time:.2f}s / {b_time:.2f}s",
            "说明": "普通演示环境下的端到端运行时间。",
        },
        {
            "实验项": "嵌入容量",
            "指标": "Payload尺寸",
            "结果": f"{result['M']} x {result['embed_width']} x 3",
            "说明": f"目标+源measurement横向拼接，单图measurement宽度为{result['measurement_width']}。",
        },
    ]
    write_tables(output_dir, rows)

    print(f"Artifacts saved to: {output_dir}")
    print(f"- {output_dir / 'comparison_grid.png'}")
    print(f"- {output_dir / 'experiment_summary.csv'}")
    print(f"- {output_dir / 'experiment_summary.md'}")
    print(f"- {output_dir / 'wrong_key_target_1.png'}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate demo experiment artifacts.")
    parser.add_argument("--input-dir", default=str(default_input_dir()), help="Directory containing demo source/target/carrier images.")
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "demo_experiments"), help="Directory to save generated artifacts.")
    parser.add_argument("--compression-ratio", type=float, default=0.6)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
