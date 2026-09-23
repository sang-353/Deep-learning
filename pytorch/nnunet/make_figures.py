"""生成第二阶段 nnU-Net 任务（MSD Task02 Heart）所需的结果图。

包含 3 张图：
    1. fig_nnunet_folds.png     五折验证 Dice，含官方值与考核门槛参考线
    2. fig_nnunet_cases.png     20 例逐例 Dice（按所属折着色）
    3. fig_nnunet_progress.png  训练过程曲线（由 nnU-Net 生成的 progress.png 复制）

数据来源：
    - 逐折与逐例 Dice：各折 fold_i/validation/summary.json；
    - 交叉验证整体指标：crossval_results_folds_0_1_2_3_4/summary.json；

用法（在 pytorch/nnunet 目录下执行）：
    python make_figures.py
"""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体：Windows 自带微软雅黑，其它平台可用黑体替代
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent.parent / "docs" / "images"

HEART_DIR = SCRIPT_DIR / "nnUNet_results" / "Dataset002_Heart"
TRAINER_DIR = HEART_DIR / "nnUNetTrainer_100epochs__nnUNetPlans_6gb__3d_fullres"
CROSSVAL_SUMMARY = TRAINER_DIR / "crossval_results_folds_0_1_2_3_4" / "summary.json"


FOLDS = [0, 1, 2, 3, 4]
OFFICIAL_DICE = 0.93                 # MSD 官方排行榜（Phase 1）中 nnU-Net 的成绩
PASS_DICE = OFFICIAL_DICE * 0.95     # 考核门槛：官方精度的 95%



def load_json(path):
    """读取 JSON 文件，文件不存在时返回 None。"""
    if not Path(path).exists():
        print(f"[跳过] 找不到文件：{path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fold_results():
    """读取每折的验证结果，返回 (折号, Dice) 与逐例 Dice 两个列表。"""
    fold_dice = []
    case_dice = []
    for fold in FOLDS:
        summary = load_json(TRAINER_DIR / f"fold_{fold}" / "validation" / "summary.json")
        if summary is None:
            continue
        fold_dice.append((fold, summary["foreground_mean"]["Dice"]))
        for case in summary["metric_per_case"]:
            case_dice.append({
                "fold": fold,
                "name": Path(case["reference_file"]).name,
                "dice": case["metrics"]["1"]["Dice"],
            })
    return fold_dice, case_dice


def figure_folds(fold_dice, overall_dice):
    """五折验证 Dice 条形图，并标出官方值与考核门槛。"""
    labels = [f"fold {f}" for f, _ in fold_dice] + ["五折均值"]
    values = [d for _, d in fold_dice] + [overall_dice]
    colors = ["#3a7bd5"] * len(fold_dice) + ["#e2703a"]

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.4f}",
                ha="center", va="bottom", fontsize=9)

    ax.axhline(OFFICIAL_DICE, color="#c0392b", linestyle="--", linewidth=1.2,
               label=f"官方公布 Dice = {OFFICIAL_DICE:.2f}")
    ax.axhline(PASS_DICE, color="#f39c12", linestyle=":", linewidth=1.6,
               label=f"考核门槛（官方 95%）= {PASS_DICE:.4f}")
    ax.set_ylim(0.86, 0.96)
    ax.set_ylabel("验证集 Dice")
    ax.set_title(f"MSD Task02 Heart 五折交叉验证结果（达到官方的 {overall_dice / OFFICIAL_DICE * 100:.1f}%）")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_nnunet_folds.png")
    plt.close(fig)


def figure_cases(case_dice):
    """20 例逐例 Dice 条形图，按折着色，并标出均值。"""
    cases = sorted(case_dice, key=lambda c: c["dice"], reverse=True)
    values = [c["dice"] for c in cases]
    mean_dice = sum(values) / len(values)

    # 每折一个颜色，便于看出某一折的整体水平
    palette = ["#3a7bd5", "#7a5fbf", "#2e9e6b", "#e2703a", "#c0392b"]
    colors = [palette[c["fold"]] for c in cases]

    fig, ax = plt.subplots(figsize=(10.0, 4.0), dpi=150)
    bars = ax.bar(range(len(cases)), values, color=colors)
    for bar, case in zip(bars, cases):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, f"{case['dice']:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=90)
    ax.axhline(mean_dice, color="#333333", linestyle="--", linewidth=1.2,
               label=f"20 例均值 = {mean_dice:.4f}")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c["name"].replace(".nii.gz", "") for c in cases], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.80, 0.99)
    ax.set_ylabel("Dice")
    ax.set_title("每例的交叉验证 Dice（每个病例均由未见过它的那一折模型预测）")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[f]) for f in FOLDS]
    ax.legend(handles + [plt.Line2D([0], [0], color="#333333", linestyle="--")],
              [f"fold {f}" for f in FOLDS] + ["均值"], fontsize=8, ncol=3)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_nnunet_cases.png")
    plt.close(fig)


def copy_progress():
    """把 nnU-Net 生成的 fold 0 训练曲线复制为文档插图。"""
    source = TRAINER_DIR / "fold_0" / "progress.png"
    if not source.exists():
        print(f"[跳过] 找不到训练曲线：{source}")
        return
    shutil.copyfile(source, FIGURE_DIR / "fig_nnunet_progress.png")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fold_dice, case_dice = load_fold_results()
    if not fold_dice:
        print("[中止] 没有读到任何一折的验证结果，请确认结果目录是否存在")
        return

    overall = load_json(CROSSVAL_SUMMARY)
    overall_dice = overall["foreground_mean"]["Dice"] if overall else \
        sum(d for _, d in fold_dice) / len(fold_dice)

    figure_folds(fold_dice, overall_dice)
    figure_cases(case_dice)
    copy_progress()

    print(f"五折均值 = {overall_dice:.4f}，逐例共 {len(case_dice)} 例")
    print("已生成图片：")
    for name in ["fig_nnunet_folds.png", "fig_nnunet_cases.png",
                 "fig_nnunet_progress.png"]:
        print(" -", name)


if __name__ == "__main__":
    main()
