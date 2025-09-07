# ps_rgb_single — 单张RGB=三光源 的光度立体 (含裁剪与数据集导出)

你的系统将 R/G/B 通道分别置于**已知光照方向/强度**。本项目把单张彩色图视作 3 次观测，
使用经典朗伯模型逐像素解法向（内部）并积分得到**相对深度**（仅保存 `depth.npy`）。

## 主要特性
- 输入默认 640×480，**中心裁剪为正方形**（480×480），可选再缩放到配置分辨率；
- 仅输出深度（`depth.npy`），不保存法向/反照率；
- 附带 `scripts/export_dataset.py`，将**裁剪后的图**写入 `images/`，**深度**写入 `w/`，
  用 `000000, 000001, ...` 连续编号。

## 安装
```bash
pip install -r requirements.txt
```

## 配置光照
编辑 `configs/default.yaml`：
```yaml
lights:
  R: { dir: [sx, sy, sz], intensity: eR }
  G: { dir: [sx, sy, sz], intensity: eG }
  B: { dir: [sx, sy, sz], intensity: eB }
```
- `dir` 为每个通道的光照方向（程序会归一化）；
- `intensity` 为相对强度系数；
- 三个方向应**分布开**，避免矩阵 `S` 条件数过大。

## 运行（单图）
```bash
python -m src.main --config configs/default.yaml
```
- 结果输出到 `outputs/run_YYYYmmdd_HHMMSS/depth.npy`。
- 如需 PNG 调试，把 `save_visual_png: true`。

## 批量导出数据集
把输入图放入某个目录，例如 `raw_imgs/`：
```bash
python scripts/export_dataset.py --config configs/default.yaml        --input raw_imgs        --dest  dataset_out
```
- 导出到 `dataset_out/images/*.png` 与 `dataset_out/w/*.npy`；
- 自动从现有编号**接档**，或用 `--start 100` 指定起始编号。

## 备注
- 深度为**相对深度**（缺全局尺度与常数）；可用标定平面/已知点做后对齐；
- 默认进行 sRGB→线性与白平衡增益；如不需要可在配置中关闭/修改；
- 若运行时提示 `S` 条件数过大，请增大 RGB 三方向夹角或调整强度比例。


## 批量处理（主程序 main.py）
支持把 `--input` 指向**目录**，主程序会按顺序处理该目录下的所有图片：
```bash
python -m src.main --config configs/default.yaml --input path/to/your_images_dir
```
- 输出结构：`outputs/run_YYYYmmdd_HHMMSS/{images/, w/, manifest.json}`；
- 裁剪后的图片保存在 `images/000000.png`，对应深度保存在 `w/000000.npy`；
- `manifest.json` 记录了编号与原始文件路径的映射。

> 你也可以继续使用 `scripts/export_dataset.py` 达到同样的效果（并具备“接档编号”等附加控制）。
