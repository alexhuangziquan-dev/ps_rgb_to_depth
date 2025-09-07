import os, argparse, time, yaml
import numpy as np
from .utils import ensure_dir, load_rgb_for_pipeline, load_mask, save_png, list_images
from .rgb_photometric_stereo import build_S_from_config, estimate_normals_from_rgb
from .integrate import integrate_depth_from_normals

def process_one(image_path, cfg, S, mask, export_dirs, idx):
    """Process a single image path and write cropped image + depth to export_dirs with sequential idx."""
    img_linear, img_vis_u8 = load_rgb_for_pipeline(
        image_path,
        center_crop_square=cfg.get("center_crop_square", True),
        resize_after_crop=tuple(cfg["resize_after_crop"]) if cfg.get("resize_after_crop", None) else None,
        srgb_linearize=cfg["srgb_linearize"],
        wb_gains=tuple(cfg["wb_gains"])
    )

    normals = estimate_normals_from_rgb(
        img_linear, S, mask=mask,
        shadow_threshold=cfg["shadow_threshold"],
        min_valid_channels=cfg["min_valid_channels"]
    )
    depth = integrate_depth_from_normals(normals, method=cfg["integration_method"])

    name = f"{idx:0{export_dirs['zpad']}d}"
    img_out = os.path.join(export_dirs["images"], f"{name}.{export_dirs['image_ext']}")
    w_out = os.path.join(export_dirs["w"], f"{name}.npy")
    save_png(img_out, img_vis_u8)
    np.save(w_out, depth.astype(np.float32))
    return name, img_out, w_out

def main(cfg, cli_input=None):
    # Determine input: file or directory
    input_path = cli_input if cli_input is not None else cfg["image_path"]
    file_list = list_images(input_path)
    if len(file_list) == 0:
        raise FileNotFoundError(f"No input image(s) found at {input_path}")

    # Output root: outputs/run_TIMESTAMP with subfolders images/ and w/
    stamp = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg["save_root"], stamp)
    images_dir = os.path.join(out_dir, "images")
    w_dir = os.path.join(out_dir, "w")
    ensure_dir(images_dir); ensure_dir(w_dir)

    # Export options
    zpad = int(cfg.get("export", {}).get("zpad", 6))
    image_ext = str(cfg.get("export", {}).get("image_ext", "png"))
    export_dirs = {"images": images_dir, "w": w_dir, "zpad": zpad, "image_ext": image_ext}

    # Optional mask (same for all inputs)
    mask = load_mask(cfg.get("mask_path", None),
                     center_crop_square=cfg.get("center_crop_square", True),
                     resize_after_crop=tuple(cfg["resize_after_crop"]) if cfg.get("resize_after_crop", None) else None)

    # Lighting matrix
    S = build_S_from_config(cfg)

    # Process batch
    manifest = []
    for idx, p in enumerate(file_list):
        name, img_out, w_out = process_one(p, cfg, S, mask, export_dirs, idx)
        manifest.append({"index": name, "src": os.path.abspath(p), "image_out": os.path.abspath(img_out), "depth_out": os.path.abspath(w_out)})
        print(f"[OK] {name}: image -> {img_out}, depth -> {w_out}")

    # Write manifest
    import json
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Processed {len(file_list)} file(s). Outputs in: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--input", type=str, help="单张图片路径或目录（RGB彩色图）")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg, cli_input=args.input)
