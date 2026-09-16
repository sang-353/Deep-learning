"""PneumoniaMNIST 肺炎二分类训练脚本。

把 pytorch/pneumonia_mnist/pneumonia_mnist.ipynb 里的流程改写成可以从命令行运行的脚本，
目的是把同一份代码原样搬到服务器上跑：服务器上没有图形界面，点不了 notebook。

训练集固定使用随机裁剪做数据增强：notebook 里做过开关对照实验，
关掉之后测试 AUC 和准确率都下降，所以这里不再保留开关。

用法示例（在 pytorch 目录下执行）：
    python pneumonia_mnist/train.py --epochs 2      # 先跑 2 轮确认脚本没问题
    python pneumonia_mnist/train.py                 # 用默认参数完整训练
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
from medmnist import PneumoniaMNIST
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "data"      # 数据已经下载在 pneumonia_mnist/data
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"  # 权重和 results.json 存放目录

# 在训练集上统计出来的像素均值和标准差；训练集和测试集用同一组，保证两者尺度一致
IMG_MEAN = (0.5719,)
IMG_STD = (0.1684,)


class Net(nn.Module):
    """两组 3x3 卷积堆叠 + BN 的 CNN，结构和 notebook 里保持一致。

    28x28 的输入经过两次 2x2 池化变成 7x7，最后接一个分类头输出 2 类（正常 / 肺炎）。
    """

    def __init__(self):
        super().__init__()

        # 第一组卷积：28x28 -> 14x14
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
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

        # 分类头：把 64x7x7 的特征摊平后经 512 维隐层输出 2 类
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


def to_label_scalar(target):
    """把 MedMNIST 形状为 (1,) 的标签数组转成标量。

    不转换的话交叉熵会报 "0D or 1D target tensor expected"；
    更麻烦的是 (batch,) 的预测和 (batch, 1) 的标签直接比较会广播成 (batch, batch)，
    准确率算错却不会报错。在数据出口统一处理，后面 loss、准确率、AUC 都自然正确。
    """
    return int(target[0])


def parse_args():
    """所有超参数都从命令行传入，换参数不用改代码。"""
    parser = argparse.ArgumentParser(description="PneumoniaMNIST 肺炎二分类训练脚本")

    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                        help="medmnist 数据缓存目录，默认取脚本同级目录的 data")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="权重和 results.json 的保存目录")

    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="数据加载进程数，Windows 用 0，Linux 服务器可以调大")

    parser.add_argument("--project", type=str, default="pneumonia-mnist-cnn",
                        help="W&B 项目名")
    parser.add_argument("--run-name", type=str, default="baseline-script",
                        help="W&B run 名称，本地和服务器两次跑用不同名字区分")

    return parser.parse_args()


def build_loaders(args):
    """准备数据，返回训练/验证/测试三个 DataLoader。

    PneumoniaMNIST 官方已经划好了 train/val/test，不需要自己再切分。
    训练集可以做数据增强，验证集和测试集只做标准化，否则评估结果忽高忽低、不可信。
    """
    train_transform = transforms.Compose([
        # 随机平移：先补 4 像素黑边再随机裁回 28x28。
        # 对照实验里关掉它之后测试 AUC 和准确率都下降，所以固定开启
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

    train_set = PneumoniaMNIST(root=args.data_root, split="train",
                               transform=train_transform, target_transform=to_label_scalar,
                               download=True)
    val_set = PneumoniaMNIST(root=args.data_root, split="val",
                             transform=test_transform, target_transform=to_label_scalar,
                             download=True)
    test_set = PneumoniaMNIST(root=args.data_root, split="test",
                              transform=test_transform, target_transform=to_label_scalar,
                              download=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # 两类数量差别很大，打印出来便于对照"全猜多数类"的准确率
    label_counts = np.bincount(np.load(Path(args.data_root) / "pneumoniamnist.npz")["train_labels"].ravel())
    print(f"训练集 {len(train_set)} 张（正常 {label_counts[0]} / 肺炎 {label_counts[1]}），"
          f"验证集 {len(val_set)} 张，测试集 {len(test_set)} 张")

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
    """返回模型在给定数据集上的 (准确率%，肺炎类 AUC)。

    准确率只数分对了几张，AUC 看的是"肺炎的分数有没有排在正常前面"。
    两类不平衡时准确率会骗人（什么都不学、全猜肺炎就有 74.2%），所以用 AUC 挑模型。
    """
    net.eval()    # 评估模式：BatchNorm 用固定的统计量，Dropout 关闭
    correct = 0
    total = 0
    all_labels = []    # 整个数据集所有样本的真实标签
    all_scores = []    # 整个数据集所有样本的"肺炎"分数

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            # 第 1 列是肺炎的分数；AUC 要的是连续分数，不能只给 argmax 之后的类别
            pneumonia_score = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 必须整个数据集收集完再算：AUC 是数据集级别的指标，逐 batch 算没有意义，
            # 而且某个 batch 只出现一类时 sklearn 会直接报错
            all_labels.append(labels.cpu())
            all_scores.append(pneumonia_score.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_scores = torch.cat(all_scores).numpy()

    accuracy = 100 * correct / total
    auc = roc_auc_score(all_labels, all_scores)
    return accuracy, auc


def main():
    args = parse_args()

    # 固定随机种子，让权重初始化、数据打乱的结果可复现
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

    best_val_auc = 0.0
    best_state = None
    history = []     # 每轮的指标，训练结束后写进 results.json

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)
        val_acc, val_auc = evaluate(net, val_loader, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # 只保留验证集 AUC 最高的那一份权重：不平衡数据上按准确率挑会偏向多数类
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(net.state_dict())

        print(
            f"Epoch [{epoch + 1}/{args.epochs}], "
            f"Loss: {avg_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%, "
            f"Val AUC: {val_auc:.4f}, "
            f"Best Val AUC: {best_val_auc:.4f}, "
            f"LR: {current_lr:.6f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "val_auc": val_auc,
            "learning_rate": current_lr,
        })

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "val/accuracy": val_acc,
            "val/auc": val_auc,
            "learning_rate": current_lr,
        })

    print("Finished Training")

    # 测试集相当于期末考卷，只在训练全部结束后评一次
    net.load_state_dict(best_state)
    test_acc, test_auc = evaluate(net, test_loader, device)
    print(f"验证集最佳 AUC: {best_val_auc:.4f}")
    print(f"测试集 AUC: {test_auc:.4f}")
    print(f"测试集准确率: {test_acc:.2f}%")

    wandb.log({
        "best_val/auc": best_val_auc,
        "test/auc": test_auc,
        "test/accuracy": test_acc,
    })

    # 保存验证集 AUC 最高的那份权重
    weight_path = output_dir / "pneumonia_cnn.pth"
    torch.save(best_state, weight_path)

    # 本地也留一份训练记录：W&B 连不上网时照样能画曲线、写文档。
    # 里面记了设备信息，事后能区分这次是在本地还是服务器上跑的
    results = {
        "config": vars(args),
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "history": history,
        "best_val_auc": best_val_auc,
        "test_auc": test_auc,
        "test_accuracy": test_acc,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"权重已保存到: {weight_path}")
    print(f"训练记录已保存到: {results_path}")

    artifact = wandb.Artifact(name="pneumonia-cnn", type="model")
    artifact.add_file(str(weight_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
