"""Fashion-MNIST 训练脚本。

把 pytorch/Fashion-MNIST.ipynb 里的流程改写成可以从命令行运行的脚本，
方便之后整体上传到服务器上跑同一份实验。

用法示例（在 pytorch 目录下执行）：
    python fashion_mnist/train.py --epochs 2     # 先跑 2 轮验证脚本没问题
    python fashion_mnist/train.py                # 用默认参数完整训练 35 轮
"""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "data"    # 即 pytorch/data，本地数据已经在这里
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"       # 权重和 results.json 存放目录


class Net(nn.Module):
    """两组 3x3 卷积堆叠 + BN 的 CNN，结构和 notebook 里保持一致。

    28x28 的输入经过两次 2x2 池化变成 7x7，最后接一个较小的全连接分类头。
    """

    def __init__(self):
        super().__init__()

        # 第一组卷积：28x28 -> 14x14
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # padding=1 让卷积前后尺寸不变
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        # 第二组卷积：14x14 -> 7x7
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        # 分类头：把 64x7x7 的特征摊平后经 128 维隐层输出 10 类
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


def parse_args():
    """所有超参数都从命令行传入，换参数不用改代码。"""
    parser = argparse.ArgumentParser(description="Fashion-MNIST CNN 训练脚本")

    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                        help="数据集存放目录，默认取脚本上一级目录的 data")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="权重和 results.json 的保存目录")

    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="数据加载进程数，Windows 用 0，Linux 服务器可以调大")

    parser.add_argument("--project", type=str, default="fashion-mnist-cnn",
                        help="W&B 项目名")
    parser.add_argument("--run-name", type=str, default="deep-aug-cosine-script",
                        help="W&B run 名称，和 notebook 里的实验区分开")

    return parser.parse_args()


def build_loaders(args):
    """准备数据，返回训练/验证/测试三个 DataLoader。

    训练集做数据增强，验证集和测试集只做标准化，否则评估结果会忽高忽低、不可信。
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),          # 随机平移：先补边再随机裁回原尺寸
        transforms.RandomHorizontalFlip(),             # 随机左右翻转，衣服翻过来还是同一类
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),    # FashionMNIST 的真实均值/标准差
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    # 同一份训练数据加载两份：一份带增强用于训练，一份不带增强用于验证
    train_full = torchvision.datasets.FashionMNIST(
        root=args.data_root, train=True, transform=train_transform, download=True)
    val_full = torchvision.datasets.FashionMNIST(
        root=args.data_root, train=True, transform=test_transform, download=True)
    test_set = torchvision.datasets.FashionMNIST(
        root=args.data_root, train=False, transform=test_transform, download=True)

    # 先切分索引，再把索引分别装进两份数据集，这样验证集不会被增强干扰
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(train_full), generator=generator).tolist()
    val_size = int(len(train_full) * args.val_ratio)
    train_set = Subset(train_full, perm[val_size:])
    val_set = Subset(val_full, perm[:val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    print(f"训练集 {len(train_set)} 张，验证集 {len(val_set)} 张，测试集 {len(test_set)} 张")
    return train_loader, val_loader, test_loader


def train_one_epoch(net, loader, criterion, optimizer, device):
    """跑完一轮训练，返回这一轮的平均损失。"""
    net.train()   # 训练模式：BatchNorm 更新统计量，Dropout 生效
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(net, loader, device):
    """返回模型在给定数据集上的准确率（%）。

    评估时必须切到 eval() 模式：BatchNorm 用固定的统计量，Dropout 关闭。
    """
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def main():
    args = parse_args()

    # 固定随机种子，让权重初始化、数据划分的结果可复现
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 打印环境信息：换一台机器（比如上服务器）时先看这几行，快速判断环境是否正常
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"使用设备: {device}")

    # 输出目录不存在就创建，保证脚本在本地和服务器上都能直接跑
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    train_loader, val_loader, test_loader = build_loaders(args)

    net = Net().to(device)
    print(net)
    print("参数量:", sum(p.numel() for p in net.parameters()))
    wandb.watch(net, log="all", log_freq=500)

    # 交叉熵加标签平滑，优化器加权重衰减，都是抑制过拟合的手段
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率按余弦曲线从初值平滑降到接近 0：前期步子大、后期步子小，收敛更稳
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_state = None
    history = []     # 每轮的指标，训练结束后写进 results.json

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)
        val_acc = evaluate(net, val_loader, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # 只保留验证集表现最好的那一份权重，避免“最后一轮反而最差”
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(net.state_dict())

        print(
            f"Epoch [{epoch + 1}/{args.epochs}], "
            f"Loss: {avg_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%, "
            f"Best Val: {best_val_acc:.2f}%, "
            f"LR: {current_lr:.6f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr,
        })

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "val/accuracy": val_acc,
            "learning_rate": current_lr,
        })

    print("Finished Training")

    # 测试集相当于期末考卷，只在训练全部结束后评一次
    net.load_state_dict(best_state)
    test_acc = evaluate(net, test_loader, device)
    print(f"验证集最佳准确率: {best_val_acc:.2f}%")
    print(f"测试集准确率: {test_acc:.2f}%")

    wandb.log({
        "best_val/accuracy": best_val_acc,
        "test/accuracy": test_acc,
    })

    # 保存验证集上表现最好的那份权重
    weight_path = output_dir / "fashion_mnist_cnn.pth"
    torch.save(best_state, weight_path)

    # 本地也留一份训练记录：W&B 连不上网时照样能画曲线、写文档
    results = {
        "config": vars(args),
        "history": history,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"权重已保存到: {weight_path}")
    print(f"训练记录已保存到: {results_path}")

    artifact = wandb.Artifact(name="fashion-mnist-cnn", type="model")
    artifact.add_file(str(weight_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
