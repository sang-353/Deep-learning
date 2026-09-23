"""生成第一阶段交付文档所需的结果图。

包含 7 张图：
    1. fig_compare_tasks.png   三个 CV 任务的测试指标对比
    2. fig_samples.png         PneumoniaMNIST 数据样例（正常 / 肺炎）
    3. fig_class_dist.png      训练集两类样本数量分布
    4. fig_train_curve.png     训练过程（损失、验证 AUC、验证准确率、学习率）
    5. fig_roc.png             测试集 ROC 曲线
    6. fig_confusion.png       测试集混淆矩阵
    7. fig_score_dist.png      两类样本的预测分数分布

训练曲线读 outputs_server/results.json，模型指标用 outputs_server/pneumonia_cnn.pth
在测试集上重新推理得到，与文档中报告的服务器端结果保持一致。

用法（在 pytorch 目录下执行）：
    python pneumonia_mnist/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from medmnist import PneumoniaMNIST
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

# 中文字体：Windows 自带微软雅黑，服务器可用黑体替代
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train import IMG_MEAN, IMG_STD, Net, to_label_scalar  # noqa: E402

DATA_ROOT = SCRIPT_DIR / "data"
RESULT_FILE = SCRIPT_DIR / "outputs_server" / "results.json"
WEIGHT_FILE = SCRIPT_DIR / "outputs_server" / "pneumonia_cnn.pth"
FIGURE_DIR = SCRIPT_DIR.parent.parent / "docs" / "images"

CLASS_NAMES = ["正常", "肺炎"]


def load_array(name):
    """从 medmnist 的 npz 里取数组，并统一成 (N, 28, 28) 的灰度形式。"""
    data = np.load(DATA_ROOT / "pneumoniamnist.npz")
    arr = data[name]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def figure_task_compare():
    """图 1：三个任务的测试指标对比。数值取自各任务的测试结果。"""
    labels = ["MNIST\n（官方教程）", "Fashion-MNIST\n基线", "Fashion-MNIST\n改进",
              "CIFAR-10\n基线", "CIFAR-10\n改进"]
    values = [98.65, 90.77, 93.28, 48.00, 91.91]
    colors = ["#8c8c8c", "#f0a35e", "#e2703a", "#9ec3e8", "#3a7bd5"]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.2f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("测试集准确率（%）")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_compare_tasks.png")
    plt.close(fig)


def figure_samples():
    """图 2：数据样例，上排正常、下排肺炎。"""
    images = load_array("test_images")
    labels = load_array("test_labels").ravel()

    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.2), dpi=150)
    for row, cls in enumerate([0, 1]):
        picked = np.where(labels == cls)[0][:5]
        for col, index in enumerate(picked):
            ax = axes[row, col]
            ax.imshow(images[index], cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_title(CLASS_NAMES[cls], fontsize=10, loc="left")
    fig.suptitle("PneumoniaMNIST 数据样例（上排：正常；下排：肺炎）", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_samples.png")
    plt.close(fig)


def figure_class_dist():
    """图 3：训练集两类样本数量分布。"""
    labels = load_array("train_labels").ravel()
    counts = [int((labels == i).sum()) for i in range(2)]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(5.0, 3.4), dpi=150)
    bars = ax.bar(CLASS_NAMES, counts, color=["#9ec3e8", "#e2703a"], width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 60,
                f"{count} 张\n{count / total * 100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("样本数量（张）")
    ax.set_ylim(0, max(counts) * 1.25)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(f"训练集共 {total} 张，肺炎样本约为正常样本的 {counts[1] / counts[0]:.1f} 倍", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_class_dist.png")
    plt.close(fig)


def figure_train_curve():
    """图 4：训练过程曲线，来自服务器端训练的逐轮记录。"""
    history = json.loads(RESULT_FILE.read_text(encoding="utf-8"))["history"]
    epochs = [row["epoch"] for row in history]
    loss = [row["train_loss"] for row in history]
    auc = [row["val_auc"] for row in history]
    acc = [row["val_accuracy"] for row in history]
    lr = [row["learning_rate"] for row in history]
    best_epoch = epochs[int(np.argmax(auc))]

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.0), dpi=150)

    axes[0, 0].plot(epochs, loss, color="#3a7bd5")
    axes[0, 0].set_title("训练损失", fontsize=10)
    axes[0, 0].set_xlabel("训练轮数")
    axes[0, 0].set_ylabel("loss")

    axes[0, 1].plot(epochs, auc, color="#e2703a")
    axes[0, 1].axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].annotate(f"最佳 {max(auc):.4f}\n第 {best_epoch} 轮",
                        xy=(best_epoch, max(auc)), xytext=(best_epoch - 22, max(auc) - 0.008),
                        fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"))
    axes[0, 1].set_title("验证集 AUC", fontsize=10)
    axes[0, 1].set_xlabel("训练轮数")

    axes[1, 0].plot(epochs, acc, color="#2e9e5b")
    axes[1, 0].set_title("验证集准确率", fontsize=10)
    axes[1, 0].set_xlabel("训练轮数")
    axes[1, 0].set_ylabel("准确率（%）")

    axes[1, 1].plot(epochs, lr, color="#7a5fbf")
    axes[1, 1].set_title("学习率（余弦退火）", fontsize=10)
    axes[1, 1].set_xlabel("训练轮数")

    for ax in axes.ravel():
        ax.grid(linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_train_curve.png")
    plt.close(fig)


def predict_test_set():
    """用保存的最优权重在测试集上推理，返回标签、分数与指标。"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    test_set = PneumoniaMNIST(root=str(DATA_ROOT), split="test", transform=transform,
                              target_transform=to_label_scalar, download=False)
    loader = DataLoader(test_set, batch_size=64, shuffle=False)

    net = Net()
    net.load_state_dict(torch.load(WEIGHT_FILE, map_location="cpu"))
    net.eval()

    labels_list, scores_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = net(images)
            scores_list.append(torch.softmax(outputs, dim=1)[:, 1])
            labels_list.append(labels)

    labels = torch.cat(labels_list).numpy()
    scores = torch.cat(scores_list).numpy()
    auc = roc_auc_score(labels, scores)
    predicted = (scores >= 0.5).astype(int)
    accuracy = 100 * (predicted == labels).mean()
    print(f"测试集：{len(labels)} 张，AUC={auc:.4f}，准确率={accuracy:.2f}%")
    return labels, scores


def figure_roc(labels, scores):
    """图 5：测试集 ROC 曲线。"""
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    ax.plot(fpr, tpr, color="#3a7bd5", linewidth=2, label=f"基线模型（AUC = {auc:.4f}）")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="随机猜测")
    ax.set_xlabel("假正例率（1 - 特异度）")
    ax.set_ylabel("真正例率（敏感度）")
    ax.set_title("测试集 ROC 曲线", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_roc.png")
    plt.close(fig)


def figure_confusion(labels, scores):
    """图 6：测试集混淆矩阵（阈值为 0.5）。"""
    predicted = (scores >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predicted)

    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=150)
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]}\n{matrix[i, j] / matrix.sum() * 100:.1f}%",
                    ha="center", va="center", fontsize=11,
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("测试集混淆矩阵（判定阈值 0.5）", fontsize=11)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_confusion.png")
    plt.close(fig)


def figure_score_dist(labels, scores):
    """图 7：两类样本的预测分数分布。"""
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=150)
    bins = np.linspace(0, 1, 26)
    ax.hist(scores[labels == 1], bins=bins, alpha=0.65, color="#e2703a", label="真实为肺炎")
    ax.hist(scores[labels == 0], bins=bins, alpha=0.65, color="#9ec3e8", label="真实为正常")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.text(0.51, ax.get_ylim()[1] * 0.9, "判定阈值 0.5", fontsize=9, color="gray")
    ax.set_xlabel("模型输出的肺炎概率")
    ax.set_ylabel("样本数量（张）")
    ax.set_title("测试集预测分数分布", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_score_dist.png")
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figure_task_compare()
    figure_samples()
    figure_class_dist()
    figure_train_curve()

    labels, scores = predict_test_set()
    figure_roc(labels, scores)
    figure_confusion(labels, scores)
    figure_score_dist(labels, scores)

    print("已生成图片：")
    for path in sorted(FIGURE_DIR.glob("fig_*.png")):
        print(" -", path.name)


if __name__ == "__main__":
    main()
