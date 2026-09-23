"""DSB2018 细胞核分割训练脚本。

用法示例（在 pytorch/U-Net 目录下执行）：
    python train.py --epochs 3 --loss bce            # 先跑 3 轮，确认脚本能通
    python train.py --loss bce                       # v1：纯 BCE，完整训练
    python train.py --loss bce_dice --run-name v2    # v2：BCE + Dice 对照
    python train.py --loss bce --augment --run-name v3-bce-cosine-aug   # v3：在 v1b 基础上加数据增强
"""

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn as nn

import wandb

from data import build_loaders
from net import UNet, combined_loss, dice_coeff

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "data" / "dsb2018" / "stage1_train"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


def parse_args():
    """所有超参数都从命令行传入，换参数不用改代码。"""
    parser = argparse.ArgumentParser(description="DSB2018 U-Net 训练脚本")

    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                        help="stage1_train 目录")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="权重和 results.json 的保存目录")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--base-channels", type=int, default=64,
                        help="U-Net 的基础通道数，显存不够可以降到 32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="数据加载进程数，Windows 用 0，Linux 可以调大")

    parser.add_argument("--augment", action="store_true",
                        help="训练集启用数据增强（翻转/旋转/亮度对比度），验证集始终不增强")

    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "bce_dice"],
                        help="bce 表示只用交叉熵，bce_dice 表示两者组合")

    parser.add_argument("--project", type=str, default="dsb2018-unet",
                        help="W&B 项目名")
    parser.add_argument("--run-name", type=str, default="v1-bce",
                        help="W&B run 名称，用来区分不同实验")

    return parser.parse_args()


def train_one_epoch(net, loader, criterion, optimizer, device):
    """跑完一轮训练，返回这一轮的平均损失和训练集 Dice。"""
    net.train()          # 训练模式：BatchNorm 更新统计量，Dropout 生效
    total_loss = 0.0
    dice_sum = 0.0
    total = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = net(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 顺手记录训练集 Dice：detach 断开计算图，避免这个指标额外占显存
        dice_sum += dice_coeff(logits.detach(), masks).item() * images.size(0)
        total += images.size(0)

    return total_loss / len(loader), dice_sum / total


@torch.no_grad()
def evaluate(net, loader, device):
    """在验证集上算平均 Dice。

    加 @torch.no_grad() 是因为评估不需要梯度，关掉之后更快也更省显存。
    """
    net.eval()           # 评估模式：BatchNorm 用固定统计量，Dropout 关闭
    dice_sum = 0.0
    total = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = net(images)

        # dice_coeff 返回的是"这一批的平均值"，乘以本批张数才能正确加权
        # （最后一个 batch 可能不满，直接平均会让结果偏一点）
        dice_sum += dice_coeff(logits, masks).item() * images.size(0)
        total += images.size(0)

    return dice_sum / total


def main():
    args = parse_args()

    # 固定随机种子，让权重初始化和数据划分可复现
    torch.manual_seed(args.seed)
    random.seed(args.seed)   # 数据增强用 random 模块抽随机数，同样要固定种子

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(args)

    net = UNet(base_channels=args.base_channels).to(device)
    print("参数量:", sum(p.numel() for p in net.parameters()))

    # 两种损失函数二选一，通过命令行切换
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = combined_loss

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    best_dice = -1.0     # 从 -1 开始，"第一轮一定比它好"，保证 best_state 一定会被赋值
    best_state = None
    history = []         # 每轮的指标，训练结束后写进 results.json

    for epoch in range(args.epochs):
        train_loss, train_dice = train_one_epoch(net, train_loader, criterion, optimizer, device)
        val_dice = evaluate(net, val_loader, device)

        scheduler.step()  # 每跑完一轮，学习率按余弦曲线降一点
        current_lr = optimizer.param_groups[0]["lr"]

        # 只保留验证集表现最好的那一份权重，避免"最后一轮反而最差"
        if val_dice > best_dice:
            best_dice = val_dice
            best_state = copy.deepcopy(net.state_dict())

        print(
            f"Epoch [{epoch + 1}/{args.epochs}]  "
            f"Loss: {train_loss:.4f}  "
            f"Train Dice: {train_dice:.4f}  "
            f"Val Dice: {val_dice:.4f}  "
            f"Best: {best_dice:.4f}  "
            f"LR: {current_lr:.6f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_dice": val_dice,
            "learning_rate": current_lr,
        })

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/dice": train_dice,
            "val/dice": val_dice,
            "learning_rate": current_lr,
        })

    print("训练结束")

    # 保存验证集上表现最好的那份权重
    weight_path = output_dir / "unet_dsb2018.pth"
    torch.save(best_state, weight_path)

    # 本地也留一份训练记录：W&B 连不上网时照样能画曲线、写文档
    results = {
        "config": vars(args),
        "history": history,
        "best_val_dice": best_dice,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"最佳验证 Dice: {best_dice:.4f}")
    print(f"权重已保存到: {weight_path}")
    print(f"训练记录已保存到: {results_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
