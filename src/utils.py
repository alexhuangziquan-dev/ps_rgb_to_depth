import os
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import glob

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def to_linear_srgb(img):
    """img: float32 [0,1], sRGB -> linear"""
    a = 0.055
    thr = 0.04045
    lin = np.where(img <= thr, img / 12.92, ((img + a) / (1 + a)) ** 2.4)
    return lin.astype(np.float32)

def center_crop_pil(im: Image.Image) -> Image.Image:
    """中心裁剪为正方形（取 min(w,h) 的边长）"""
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return im.crop((left, top, left + side, top + side))

def load_rgb_for_pipeline(path, center_crop_square=True, resize_after_crop=(64,64),
                          srgb_linearize=True, wb_gains=(1.0,1.0,1.0)):
    """返回: (img_linear[H,W,3], img_vis_u8[H,W,3])"""
    im = Image.open(path).convert("RGB")
    if center_crop_square:
        im = center_crop_pil(im)
    if resize_after_crop is not None:
        im = im.resize((resize_after_crop[1], resize_after_crop[0]), Image.BILINEAR)
    # sRGB 可视版本
    arr_srgb = np.asarray(im).astype(np.uint8)  # [H,W,3] uint8
    # 算法用线性版本
    arr_lin = arr_srgb.astype(np.float32) / 255.0
    if srgb_linearize:
        arr_lin = to_linear_srgb(arr_lin)
    wb = np.array(wb_gains, dtype=np.float32).reshape(1,1,3)
    arr_lin = np.clip(arr_lin * wb, 0.0, 1.0).astype(np.float32)
    return arr_lin, arr_srgb

def load_mask(mask_path, center_crop_square, resize_after_crop):
    if mask_path is None:
        return None
    im = Image.open(mask_path).convert("L")
    if center_crop_square:
        im = center_crop_pil(im)
    if resize_after_crop is not None:
        im = im.resize((resize_after_crop[1], resize_after_crop[0]), Image.NEAREST)
    arr = np.asarray(im).astype(np.float32) / 255.0
    msk = (arr > 0.5).astype(np.bool_)
    return msk

def save_png(path, arr):
    """arr: [H,W]或[H,W,3]，float(0~1)或uint8"""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    imageio.imwrite(path, arr)


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def list_images(input_path):
    """Return sorted list of image file paths from a file or a directory."""
    if os.path.isdir(input_path):
        files = []
        for ext in IMG_EXTS:
            files.extend(sorted(glob.glob(os.path.join(input_path, f"*{ext}"))))
        return sorted(files)
    else:
        return [input_path]
