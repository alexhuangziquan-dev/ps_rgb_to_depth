# -*- coding: utf-8 -*-
"""
主程序：从 RGB(光度立体)计算相对深度 w，并按 config.scale 输出真实深度矩阵。
- 读取 YAML 配置（含新增参数 scale）
- 运行现有的深度重建流程获得相对深度 w_relative
- 应用 w_real = w_relative * scale
- 保存 w_real 到输出目录（.npy + 可视化 .png）
注意：
- 这里的重建函数 demo_reconstruct() 仅作占位示例。
  在你的仓库中，把它替换为你原本用于得到“相对深度 w”的函数/流程即可。
"""

import os
import argparse
import yaml
import numpy as np
from pathlib import Path

try:
    import imageio.v2 as imageio  # imageio>=2
except Exception:
    import imageio  # 兼容老版本

# =========================
# 你的实际重建函数应替换这里
# =========================
def demo_reconstruct(input_dir: str, cfg: dict) -> np.ndarray:
    """
    占位示例：返回一张相对深度 w（H×W，float32）。
    请用你仓库中真实的重建流程替代这段实现：
    例如：w = your_module.estimate_depth(images, lights, cfg)
    """
    # 为了可运行演示，这里构造一张 256x256 的渐变“相对深度”
    H, W = 256, 256
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    w_relative = (x + y) / 2.0  # 0~1 的相对值
    return w_relative.astype(np.float32)


def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)


def save_numpy(path: str, arr: np.ndarray):
    np.save(path, arr)


def save_png(path: str, arr: np.ndarray):
    """
    保存为 16bit PNG（常见深度可视化/交换格式）。
    - 若你的真实单位是米，且数值较小，你也可以改乘 1000 存成毫米并转 uint16。
    - 下面采用自适应线性拉伸，仅用于可视化（不会覆盖 .npy 的真实数值）。
    """
    a = np.asarray(arr, dtype=np.float32)
    amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
    if amax - amin < 1e-12:
        norm = np.zeros_like(a, dtype=np.uint16)
    else:
        # 可视化映射到 0~65535
        norm = np.clip((a - amin) / (amax - amin), 0.0, 1.0)
        norm = (norm * 65535.0 + 0.5).astype(np.uint16)
    imageio.imwrite(path, norm)


def parse_args():
    p = argparse.ArgumentParser(description="ps_rgb_to_depth: 输出真实深度（按 scale 换算）")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="YAML 配置文件路径")
    p.add_argument("--input", type=str, required=False, default="./train_images", help="输入图像目录")
    p.add_argument("--out", type=str, required=False, default="./prediction", help="输出目录")
    return p.parse_args()


def main():
    args = parse_args()

    # 读取配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    input_dir = args.input if args.input is not None else cfg.get("input", "./train_images")
    out_dir   = args.out if args.out is not None else cfg.get("output_dir", "./prediction")
    ensure_dir(out_dir)

    # ======================== 你的原有流程 =========================
    # 在这里调用你原本的重建函数，得到「相对深度」w_relative
    # 例如：w_relative = estimate_depth(input_dir, lights=cfg["lights"], ...)
    w_relative = demo_reconstruct(input_dir, cfg)   # <<< 用你的真实函数替换这行
    # ============================================================

    # >>>>>>>>>>>>>>>>>>>>>>> SCALE 应用开始 <<<<<<<<<<<<<<<<<<<<<<<
    # 读取 scale（默认 1.0），将相对深度换算为真实深度
    scale = float(cfg.get("scale", 1.0))
    w_real = (w_relative.astype(np.float32)) * scale
    # <<<<<<<<<<<<<<<<<<<<<<< SCALE 应用结束 <<<<<<<<<<<<<<<<<<<<<<<

    # 保存：真实深度矩阵（数值不做拉伸），以及仅供查看的可视化 PNG
    npy_path  = os.path.join(out_dir, "w_real.npy")
    png_path  = os.path.join(out_dir, "w_real_vis.png")

    save_numpy(npy_path, w_real)  # 真实值（后续计算请以此为准）
    save_png(png_path, w_real)    # 仅可视化

    # 如果你的原工程需要把 w 变量继续向下游传递/返回，
    # 也请确保使用 w_real（而不是 w_relative）
    print(f"[OK] 已保存真实深度矩阵: {npy_path}")
    print(f"[OK] 可视化图: {png_path}")
    print(f"[INFO] scale = {scale}, 深度范围: min={float(np.nanmin(w_real)):.6g}, max={float(np.nanmax(w_real)):.6g}")


if __name__ == "__main__":
    main()
