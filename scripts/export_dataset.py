import os, argparse, glob, yaml, re, sys
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import ensure_dir, load_rgb_for_pipeline, save_png, load_mask
from src.rgb_photometric_stereo import build_S_from_config, estimate_normals_from_rgb
from src.integrate import integrate_depth_from_normals

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def list_images(input_path):
    if os.path.isdir(input_path):
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
        return sorted(files)
    else:
        return [input_path]

def next_index(dest_images, dest_w, zpad):
    import re
    pattern = re.compile(rf"^(\d{{{zpad}}})\." )
    indices = []
    for d in [dest_images, dest_w]:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            m = pattern.match(name)
            if m:
                indices.append(int(m.group(1)))
    if not indices:
        return 0
    return max(indices) + 1

def main(cfg, args):
    # 解析导出目标
    dest_root = args.dest if args.dest else cfg["export"]["dest_root"]
    zpad = args.zpad if args.zpad is not None else cfg["export"]["zpad"]
    image_ext = args.image_ext if args.image_ext else cfg["export"].get("image_ext", "png")
    overwrite = args.overwrite if args.overwrite is not None else cfg["export"].get("overwrite", False)

    dest_images = os.path.join(dest_root, "images")
    dest_w = os.path.join(dest_root, "w")
    ensure_dir(dest_images); ensure_dir(dest_w)

    # 光照矩阵
    S = build_S_from_config(cfg)

    # 输入文件列表
    files = list_images(args.input if args.input else cfg["image_path"])  # 支持单图或目录
    if len(files) == 0:
        raise FileNotFoundError("No input image found.")

    # 起始编号
    idx = args.start if args.start is not None else next_index(dest_images, dest_w, zpad)

    for path in files:
        # 1) 载入并预处理（中心裁剪→可选缩放）
        img_linear, img_vis_u8 = load_rgb_for_pipeline(
            path,
            center_crop_square=cfg.get("center_crop_square", True),
            resize_after_crop=tuple(cfg["resize_after_crop"]) if cfg.get("resize_after_crop", None) else None,
            srgb_linearize=cfg["srgb_linearize"],
            wb_gains=tuple(cfg["wb_gains"])
        )
        mask = load_mask(cfg.get("mask_path", None),
                         center_crop_square=cfg.get("center_crop_square", True),
                         resize_after_crop=tuple(cfg["resize_after_crop"]) if cfg.get("resize_after_crop", None) else None)

        # 2) 法向（内部）
        normals = estimate_normals_from_rgb(
            img_linear, S, mask=mask,
            shadow_threshold=cfg["shadow_threshold"],
            min_valid_channels=cfg["min_valid_channels"]
        )

        # 3) 深度
        depth = integrate_depth_from_normals(normals, method=cfg["integration_method"])

        # 4) 保存到指定目录
        name = f"{idx:0{zpad}d}"
        img_out = os.path.join(dest_images, f"{name}.{image_ext}")
        w_out = os.path.join(dest_w, f"{name}.npy")

        if (not overwrite) and (os.path.exists(img_out) or os.path.exists(w_out)):
            print(f"[SKIP] {name} exists (overwrite=false)")
            idx += 1
            continue

        save_png(img_out, img_vis_u8)
        np.save(w_out, depth.astype(np.float32))
        print(f"[OK] {name}: image -> {img_out}, depth -> {w_out}")
        idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--input", type=str, help="输入：单张图片路径或目录（RGB彩色图）")
    parser.add_argument("--dest", type=str, help="输出根目录（内部含 images/ 与 w/）")
    parser.add_argument("--start", type=int, help="起始编号（默认自动接档）")
    parser.add_argument("--zpad", type=int, help="编号位数，默认取配置 export.zpad")
    parser.add_argument("--image_ext", type=str, help="images 保存格式，如 png/jpg")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名文件")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg, args)
