# Fashion-MNIST 训练脚本

把 `pytorch/Fashion-MNIST.ipynb` 里的训练流程改写成可以从命令行运行的脚本。目的是：后续把整个目录搬到服务器上时，跑的是完全同一份代码。

## 目录内容

| 文件 | 说明 |
| --- | --- |
| `train.py` | 训练脚本，所有超参数都从命令行传入 |
| `requirements.txt` | 依赖及版本记录 |
| `outputs/` | 跑完才出现，里面是权重和 `results.json` |

## 运行环境

- conda 环境名：`pytorch`，Python 3.11.16
- 主要依赖：torch 2.14.0+cu126、torchvision 0.29.0+cu126、wandb 0.30.0
- 本地硬件：RTX 4060 Laptop（8GB 显存）

**注意环境不要搞混**：命令行里直接敲 `python` 用的是 conda 的 base 环境（Python 3.14），和本目录使用的 `pytorch` 环境（Python 3.11）不是同一个。目前两边装的 torch 版本恰好相同，所以都能跑，但跑之前还是要确认已经激活 `pytorch` 环境，避免出现"在 notebook 里能跑、在命令行里报错"这类难查的问题。

## 怎么跑

```powershell
conda activate pytorch
cd E:\code\PycharmProjects\pythonProject\pytorch

python fashion_mnist\train.py --epochs 2   # 先快速跑 2 轮，确认脚本没问题
python fashion_mnist\train.py              # 用默认参数完整训练 35 轮
```

数据集默认去 `pytorch/data` 找，本地已经下载过。换到别的机器第一次跑时，torchvision 会自动下载到 `--data-root` 指定的目录。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--epochs` | 35 | 训练轮数 |
| `--batch-size` | 64 | 每批样本数 |
| `--lr` | 0.001 | 初始学习率（余弦退火逐步降到接近 0） |
| `--weight-decay` | 5e-5 | 权重衰减，抑制过拟合 |
| `--label-smoothing` | 0.1 | 标签平滑，避免模型过度自信 |
| `--val-ratio` | 0.1 | 从训练集里划出多少比例做验证集 |
| `--seed` | 42 | 随机种子 |
| `--data-root` | `pytorch/data` | 数据集目录 |
| `--output-dir` | `fashion_mnist/outputs` | 权重和 `results.json` 的保存目录 |
| `--num-workers` | 0 | 数据加载进程数，Windows 用 0，Linux 服务器可以调大 |
| `--project` | fashion-mnist-cnn | W&B 项目名 |
| `--run-name` | deep-aug-cosine-script | W&B run 名称 |

## 跑完会得到什么

- `outputs/fashion_mnist_cnn.pth`：验证集上表现最好的那份权重（不是最后一轮的）
- `outputs/results.json`：每一轮的 loss / 验证准确率 / 学习率，以及最终的测试准确率
- W&B 上一条新 run：训练曲线、验证指标、权重与梯度直方图，权重也会作为 Artifact 上传

`results.json` 主要是为服务器训练准备的：服务器上 W&B 可能连不上网，本地留下这份记录，照样能画曲线、写文档。

## 目前的结果

| 版本 | 测试集准确率 | 备注 |
| --- | --- | --- |
| baseline（notebook） | 90.77% | 无数据增强、无验证集 |
| 改进版（notebook） | 93.37%（验证集 94.15%） | 加深网络 + 数据增强 + 余弦退火 |
| 脚本版（`train.py`） | 93.28%（验证集 94.23%） | 35 轮，默认随机种子，最佳验证轮次是第 33 轮 |

两次实验的差距：测试集 0.09 个百分点（93.28% 对 93.37%），验证集 0.08 个百分点。这点差距来自数据增强和打乱顺序的随机性，属于正常范围，说明脚本化没有改变实验逻辑，脚本版可以替代 notebook 版。

本次运行的 W&B 记录：https://wandb.ai/3531427435-none/fashion-mnist-cnn/runs/f2dtrzwk

## 和 notebook 的关系

`train.py` 是 `pytorch/Fashion-MNIST.ipynb` 的工程化版本，网络结构、数据增强、超参数都保持一致。notebook 保留不动，它是过程记录（包括当时踩过的坑），是之后写交付文档的素材。

## 上服务器时要注意

- 服务器上用这个脚本跑，不用 notebook。
- `torch` 和 `torchvision` 不要直接照 `requirements.txt` 装：本地装的是 Windows 的 CUDA 12.6 版本，服务器要按 PyTorch 官网给出的对应 CUDA 命令安装，其余包再按 `requirements.txt` 装。
- `--data-root` 指向服务器上放数据的路径。
- W&B 连不上网时设离线模式：Windows 用 `set WANDB_MODE=offline`，Linux 用 `export WANDB_MODE=offline`。
- 训练用 `tmux` 之类的工具挂着跑，这样 SSH 断开后训练不会被中断。
