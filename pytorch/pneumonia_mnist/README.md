# PneumoniaMNIST 肺炎二分类训练脚本

把 `pytorch/pneumonia_mnist/pneumonia_mnist.ipynb` 里的训练流程改写成可以从命令行运行的脚本。目的是：后续把整个目录搬到服务器上时，跑的是完全同一份代码。

## 目录内容

| 文件 | 说明 |
| --- | --- |
| `train.py` | 训练脚本，所有超参数都从命令行传入 |
| `requirements.txt` | 依赖及版本记录 |
| `pneumonia_mnist.ipynb` | 过程记录（含踩过的坑），保留不动 |
| `data/` | medmnist 下载的数据（`pneumoniamnist.npz`） |
| `outputs/` | 跑完才出现，里面是权重和 `results.json` |

## 运行环境

- conda 环境名：`pytorch`，Python 3.11.16
- 主要依赖：torch 2.14.0+cu126、torchvision 0.29.0+cu126、medmnist 3.0.2、scikit-learn 1.9.1、wandb 0.30.0
- 本地硬件：RTX 4060 Laptop（8GB 显存）

## 怎么跑

```powershell
conda activate pytorch
cd E:\code\PycharmProjects\pythonProject\pytorch

python pneumonia_mnist\train.py --epochs 2     # 先快速跑 2 轮，确认脚本没问题
python pneumonia_mnist\train.py                # 用默认参数完整训练 35 轮
```

数据默认去 `pneumonia_mnist/data` 找。官方已经划好了 train/val/test，脚本不自己切分验证集。

训练集固定做随机裁剪增强（做过开关对照实验：关掉之后测试 AUC 从 0.9687 降到 0.9519、准确率从 91.35% 降到 88.30%，因此不再保留开关）；验证集和测试集只做标准化。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--epochs` | 35 | 训练轮数 |
| `--batch-size` | 64 | 每批样本数 |
| `--lr` | 0.001 | 初始学习率（余弦退火逐步降到接近 0） |
| `--weight-decay` | 5e-5 | 权重衰减，抑制过拟合 |
| `--label-smoothing` | 0.1 | 标签平滑，避免模型过度自信 |
| `--seed` | 42 | 随机种子 |
| `--data-root` | `pneumonia_mnist/data` | 数据目录 |
| `--output-dir` | `pneumonia_mnist/outputs` | 权重和 `results.json` 的保存目录 |
| `--num-workers` | 0 | 数据加载进程数，Windows 用 0，Linux 服务器可以调大 |
| `--project` | pneumonia-mnist-cnn | W&B 项目名 |
| `--run-name` | baseline-script | W&B run 名称，本地和服务器两次跑用不同名字区分 |

## 跑完会得到什么

- `outputs/pneumonia_cnn.pth`：验证集 AUC 最高的那份权重（不是最后一轮的）
- `outputs/results.json`：每一轮的 loss / 验证准确率 / 验证 AUC / 学习率，以及最终的测试指标，还记了这次跑在哪台设备上
- W&B 的本地记录写在执行命令时所在的目录下：在 `pytorch` 目录下跑脚本，run 记到 `pytorch/wandb/`；在 notebook 里跑则记到 `pneumonia_mnist/wandb/`。两处都已排除在版本控制之外
- W&B 上一条新 run：训练曲线、验证指标、权重与梯度直方图，权重也会作为 Artifact 上传

## 目前的结果

| 版本 | 测试 AUC | 测试准确率 | 备注 |
| --- | --- | --- | --- |
| notebook，开启随机裁剪 | 0.9687 | 91.35% | 本地，35 轮 |
| notebook，关闭随机裁剪 | 0.9519 | 88.30% | 本地，35 轮，数据增强的对照实验 |
| 脚本版 `train.py`（开启随机裁剪） | 0.9692 | 90.38% | 本地，35 轮，`--run-name baseline-local` |
| 服务器版 | 待补 | 待补 | 上服务器跑完再补这一行 |

三点结论：

1. **脚本化和 notebook 的结果基本一致。** 测试 AUC 0.9692 对 0.9687，只差 0.0005；准确率 90.38% 对 91.35%，差 0.96 个百分点（约 6 张图）。AUC 几乎相同说明模型的排序能力没变，准确率那点差异来自分数落在 0.5 阈值附近的几张图被分到了另一侧——两次运行时数据打乱顺序和随机裁剪的位置不完全一样，这个量级的波动属于正常范围，说明脚本化没有改变实验逻辑。
2. **随机裁剪这组增强是有效的。** 关掉之后测试 AUC 下降 1.68 个百分点，准确率下降 3.05 个百分点，所以脚本里固定开启，不再保留开关。
3. **`--epochs 2` 的冒烟测试也留下了一条 W&B run**（测试 AUC 0.9542）。它只代表两轮训练，不是基线成绩，不要把那条 run 的数字当结果用。

## 上服务器时要注意

- 服务器上跑脚本，不用 notebook。
- `torch` 和 `torchvision` 不要直接照 `requirements.txt` 装：本地装的是 Windows 的 CUDA 12.6 版本，服务器要按 PyTorch 官网给出的对应 CUDA 命令安装，其余包再按 `requirements.txt` 装。
- 数据可以先在本地下好，把 `data/pneumoniamnist.npz` 传到服务器，用 `--data-root` 指向它所在的目录；服务器能联网时让它自己下也行。
- W&B 连不上网时设离线模式：Linux 用 `export WANDB_MODE=offline`。
- 训练用 `tmux` 挂着跑，SSH 断开后训练不会被中断。
- Linux 上可以把 `--num-workers` 调大（比如 4）加快数据加载。
