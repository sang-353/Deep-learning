"""生成第二阶段 U-Net 分割任务所需的结果图。

包含 5 张图：
    1. fig_unet_samples.png   DSB2018 数据样例（原图 / 合并掩膜 / 叠加）
    2. fig_unet_curves.png    四次实验的验证集 Dice 曲线对比
    3. fig_unet_gap.png       数据增强前后：训练与验证 Dice 曲线及两者差距
    4. fig_unet_pred.png      最终模型在验证集上的预测结果（4 个样本）
    5. fig_unet_fail.png      验证集中 Dice 最低的 3 个样本（失败案例分析）

数据来源：
    - v1（纯 BCE、无余弦）的 results.json 已被后续运行覆盖，逐轮记录从 W&B
      本地保存的训练日志中解析（见 LOG_V1）；
    - 其余三次运行读各自输出目录下的 results.json；
    - 预测图用 outputs/v3-bce-cosine-aug/unet_dsb2018.pth 在验证集上重新推理得到。

用法（在 pytorch/U-Net 目录下执行）：
    python make_figures.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# 中文字体：Windows 自带微软雅黑，服务器可用黑体替代
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data import build_datasets  # noqa: E402
from net import UNet, dice_coeff  # noqa: E402

DATA_ROOT = SCRIPT_DIR.parent / "data" / "dsb2018" / "stage1_train"
FIGURE_DIR = SCRIPT_DIR.parent.parent / "docs" / "images"

IMAGE_SIZE = 256
VAL_RATIO = 0.15
SEED = 42

# v1（纯 BCE、无余弦）的训练日志：该次运行的 results.json 被后续运行覆盖，只剩这份日志
LOG_V1 = SCRIPT_DIR / "wandb" / "run-20260919_163029-7uwrp4od" / "files" / "output.log"

# 四组对照实验的曲线来源与配色，顺序即图例顺序（也决定了下文按下标取用的位置）
# 下标：0 = v1， 1 = v1b， 2 = v3， 3 = v4
RUNS = [
    ("v1　纯 BCE", ("log", LOG_V1), "#8c8c8c"),
    ("v1b　BCE + 余弦", ("json", SCRIPT_DIR / "outputs" / "v1b-bce-cosine" / "results.json"), "#7a5fbf"),
    ("v3　BCE + 余弦 + 增强", ("json", SCRIPT_DIR / "outputs" / "v3-bce-cosine-aug" / "results.json"), "#3a7bd5"),
    ("v4　BCE + Dice + 余弦", ("json", SCRIPT_DIR / "outputs" / "v4-bce-dice-cosine" / "results.json"), "#e2703a"),
]

WEIGHT_FILE = SCRIPT_DIR / "outputs" / "v3-bce-cosine-aug" / "unet_dsb2018.pth"


def load_history(source):
    """按来源读取逐轮记录，返回 [{'epoch', 'val_dice', ...}, ...]。"""
    kind, path = source
    if not path.exists():
        print(f"[跳过] 找不到记录：{path}")
        return None

    if kind == "json":
        return json.loads(path.read_text(encoding="utf-8"))["history"]

    # 训练日志的每行形如：
    # Epoch [1/50]  Loss: 0.2489  Val Dice: 0.5892  Best: 0.5892
    pattern = re.compile(r"Epoch \[(\d+)/\d+\]\s+Loss:\s+([\d.]+).*?Val Dice:\s+([\d.]+)")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            rows.append({
                "epoch": int(match.group(1)),
                "train_loss": float(match.group(2)),
                "val_dice": float(match.group(3)),
            })
    return rows or None


def overlay(image, mask_true, mask_pred=None):
    """把掩膜叠在原图上：真实标签用绿色、预测用红色，两者重合处呈黄色。"""
    canvas = np.zeros((*mask_true.shape, 3), dtype=np.float32)
    canvas[..., 1] = mask_true
    if mask_pred is not None:
        canvas[..., 0] = mask_pred
    return canvas


def to_numpy(image, mask):
    """图像张量 (3, H, W) 与掩膜张量 (1, H, W) 转成绘图用的 numpy 数组。"""
    return image.permute(1, 2, 0).numpy(), mask[0].numpy()


def figure_samples():
    """图 1：数据样例，展示原图、合并后的掩膜与两者的叠加。"""
    train_set, _ = build_datasets(DATA_ROOT, IMAGE_SIZE, VAL_RATIO, SEED)

    fig, axes = plt.subplots(3, 3, figsize=(9.0, 9.0), dpi=150)
    for row in range(3):
        image, mask = train_set[row]
        image, mask = to_numpy(image, mask)
        for col, (panel, kwargs) in enumerate([
            (image, dict(cmap=None)),
            (mask, dict(cmap="gray", vmin=0, vmax=1)),
            (image, dict(cmap=None)),
        ]):
            ax = axes[row, col]
            ax.imshow(panel, **kwargs)
            if col == 2:
                ax.imshow(overlay(image, mask), alpha=0.45)
            ax.axis("off")
            if row == 0:
                ax.set_title(["原图", "合并后的掩膜", "叠加（绿色为细胞核）"][col], fontsize=11)

    fig.suptitle("DSB2018 细胞核分割数据样例（单个样本含数十个细胞核）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_unet_samples.png")
    plt.close(fig)


def figure_curves():
    """图 2：四次实验的验证集 Dice 曲线对比。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)

    for label, source, color in RUNS:
        history = load_history(source)
        if history is None:
            continue
        epochs = [row["epoch"] for row in history]
        dice = [row["val_dice"] for row in history]
        ax.plot(epochs, dice, color=color, linewidth=1.6, label=label)
        print(f"{label}：最佳 {max(dice):.4f}（第 {epochs[int(np.argmax(dice))]} 轮）")

    ax.set_xlabel("训练轮数")
    ax.set_ylabel("验证集 Dice")
    ax.set_title("四次实验的验证集 Dice 变化", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_unet_curves.png")
    plt.close(fig)


def figure_gap():
    """图 3：数据增强前后训练与验证 Dice 的对比，以及两者的差距。"""
    before = load_history(RUNS[1][1])   # v1b：无增强
    after = load_history(RUNS[2][1])    # v3：有增强

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), dpi=150)
    gaps = []
    tail_means = []

    for ax, history, title, color in [
        (axes[0], before, "v1b　无数据增强", "#7a5fbf"),
        (axes[1], after, "v3　启用数据增强", "#3a7bd5"),
    ]:
        epochs = [row["epoch"] for row in history]
        train_dice = [row["train_dice"] for row in history]
        val_dice = [row["val_dice"] for row in history]
        ax.plot(epochs, train_dice, color=color, linewidth=1.5, label="训练集")
        ax.plot(epochs, val_dice, color=color, linewidth=1.5, linestyle="--", label="验证集")
        ax.fill_between(epochs, val_dice, train_dice, color=color, alpha=0.12)
        ax.set_xlabel("训练轮数")
        ax.set_ylabel("Dice")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.6, 0.95)
        ax.grid(linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.legend(fontsize=9, loc="lower right")
        gaps.append(np.mean(train_dice[-10:]) - np.mean(val_dice[-10:]))
        tail_means.append((np.mean(train_dice[-10:]), np.mean(val_dice[-10:])))

    bars = axes[2].bar(["无增强\n(v1b)", "有增强\n(v3)"], gaps, color=["#7a5fbf", "#3a7bd5"], width=0.5)
    for bar, gap in zip(bars, gaps):
        axes[2].text(bar.get_x() + bar.get_width() / 2, gap + 0.002, f"{gap:.4f}",
                     ha="center", va="bottom", fontsize=9)
    axes[2].set_ylabel("末 10 轮 训练与验证 Dice 之差")
    axes[2].set_title("过拟合差距对比", fontsize=10)
    axes[2].set_ylim(0, max(gaps) * 1.35)
    axes[2].grid(axis="y", linestyle=":", alpha=0.5)
    axes[2].set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_unet_gap.png")
    plt.close(fig)

    for (label, _, _), (train_mean, val_mean), gap in zip([RUNS[1], RUNS[2]], tail_means, gaps):
        print(f"{label} 末10轮：训练 {train_mean:.4f}，验证 {val_mean:.4f}，差距 {gap:.4f}")


def predict_val_set(limit=12):
    """用最终权重在验证集上推理，返回逐样本的 Dice、图像与掩膜。"""
    _, val_set = build_datasets(DATA_ROOT, IMAGE_SIZE, VAL_RATIO, SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(base_channels=64).to(device)
    net.load_state_dict(torch.load(WEIGHT_FILE, map_location=device))
    net.eval()

    results = []
    with torch.no_grad():
        for index in range(len(val_set)):
            image, mask = val_set[index]
            logits = net(image.unsqueeze(0).to(device))
            score = dice_coeff(logits, mask.unsqueeze(0).to(device)).item()
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            results.append({
                "index": index,
                "dice": score,
                "image": image.permute(1, 2, 0).numpy(),
                "mask": mask[0].numpy(),
                "prob": prob,
            })

    mean_dice = float(np.mean([row["dice"] for row in results]))
    print(f"验证集 {len(results)} 张，平均 Dice = {mean_dice:.4f}（设备：{device}）")
    return results


def figure_predictions(results, picks, filename, title):
    """画出若干样本的四联图：原图 / 真实标签 / 预测概率 / 叠加。"""
    fig, axes = plt.subplots(len(picks), 4, figsize=(11.0, 2.7 * len(picks)), dpi=150)
    if len(picks) == 1:
        axes = axes[np.newaxis, :]

    for row, pick in enumerate(picks):
        item = results[pick]
        predicted = (item["prob"] > 0.5).astype(np.float32)

        panels = [
            (item["image"], dict(cmap=None), "原图"),
            (item["mask"], dict(cmap="gray", vmin=0, vmax=1), "真实标签"),
            (item["prob"], dict(cmap="viridis", vmin=0, vmax=1), "预测概率"),
            (item["image"], dict(cmap=None), f"叠加（Dice = {item['dice']:.4f}）"),
        ]
        for col, (panel, kwargs, label) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel, **kwargs)
            if col == 3:
                ax.imshow(overlay(item["image"], item["mask"], predicted), alpha=0.45)
            ax.axis("off")
            if row == 0:
                ax.set_title(label, fontsize=10)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figure_samples()
    figure_curves()
    figure_gap()

    results = predict_val_set()
    order = sorted(range(len(results)), key=lambda i: results[i]["dice"], reverse=True)
    figure_predictions(results, order[:4], "fig_unet_pred.png",
                       "最终模型在验证集上的分割结果（绿色为真实标签，红色为预测，黄色为重合）")
    figure_predictions(results, order[-3:], "fig_unet_fail.png",
                       "验证集中表现最差的 3 个样本（真实标签与预测差异明显）")

    print("已生成图片：")
    for name in ["fig_unet_samples.png", "fig_unet_curves.png", "fig_unet_gap.png",
                 "fig_unet_pred.png", "fig_unet_fail.png"]:
        print(" -", name)


if __name__ == "__main__":
    main()
