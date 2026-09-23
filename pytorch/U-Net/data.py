import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
import torch

def merge_masks(mask_paths):
    """把一张图下的所有单核掩膜合并成一张二值掩膜。
     返回 float32 数组，形状 (H, W)，取值只可能是 0.0 或 1.0。"""
    merged = None
    for mask_path in mask_paths:
        one = np.array(Image.open(mask_path).convert("L"))
        merged = one if merged is None else np.maximum(merged, one)

    # 阈值 127：只有明显亮起来的像素才算前景；再转成 0/1
    return (merged>127).astype(np.float32)

class DSBDataset(Dataset):
    """每个样本返回 (图像张量, 掩膜张量)。

       图像形状 (3, image_size, image_size)，掩膜形状 (1, image_size, image_size)。
       """
    def __init__(self, case_dirs, image_size=256, augment=False):
        self.case_dirs = list(case_dirs)
        self.image_size = image_size
        self.augment = augment        # True 时启用数据增强（只用在训练集）
    def __len__(self):
        return len(self.case_dirs)
    def __getitem__(self, index):
        case_dir = self.case_dirs[index]
        # 1. 原图：源文件是 RGBA，先转成 RGB，否则会多出一个透明通道
        image_path = next((case_dir / "images").glob("*.png"))
        image = Image.open(image_path).convert("RGB")
        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        # to_tensor 会把 0-255 压到 0-1，并把通道维挪到最前面，得到 (3, H, W)
        image = TF.to_tensor(image)

        # 2. 掩膜：先合并成一张，再缩放
        mask_paths = sorted((case_dir / "masks").glob("*.png"))
        mask = merge_masks(mask_paths)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        # 掩膜必须用最近邻插值：双线性会在边界插出 0.5 这类中间值，破坏二值性
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        # 3. 数据增强：只在 augment=True 时生效，验证集保持原样
        if self.augment:
            # 翻转、旋转属于几何变换，图和掩膜必须用同一个随机判定，否则两者会错位
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)

            # k 只抽一次，图和掩膜共用同一个 k；张量形状是 (C, H, W)，所以在第 1、2 维上旋转
            k = random.randint(0, 3)
            if k:
                image = torch.rot90(image, k, dims=(1, 2))
                mask = torch.rot90(mask, k, dims=(1, 2))

            # 颜色扰动只作用于图像，掩膜不参与：模拟不同染色深浅和不同设备的亮度差异
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            image = TF.adjust_contrast(TF.adjust_brightness(image, brightness), contrast)

        return image, mask

def build_datasets(data_root, image_size, val_ratio, seed, augment=False):
    """把样本按固定随机种子划分成训练集和验证集。"""
    case_dirs = sorted(Path(data_root).iterdir())
    if not case_dirs:
        raise FileNotFoundError(f"目录里没有样本：{data_root}")

    # 用固定种子打乱索引，保证每次运行划分结果一致
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(case_dirs), generator=generator).tolist()

    val_size = int(len(case_dirs) * val_ratio)
    val_dirs = [case_dirs[i] for i in order[:val_size]]
    train_dirs = [case_dirs[i] for i in order[val_size:]]

    # 只给训练集开增强；验证集必须保持原始形态，否则评估结果不可信
    train_set = DSBDataset(train_dirs, image_size, augment=augment)
    val_set = DSBDataset(val_dirs, image_size, augment=False)
    return train_set, val_set


def build_loaders(args):
    """按命令行参数构造训练集和验证集的 DataLoader。"""
    train_set, val_set = build_datasets(
        args.data_root, args.image_size, args.val_ratio, args.seed, args.augment)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    print(f"训练集 {len(train_set)} 张，验证集 {len(val_set)} 张")
    return train_loader, val_loader
