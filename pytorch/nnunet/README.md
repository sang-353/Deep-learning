# nnU-Net 学习记录

本目录记录 nnU-Net 的学习与复现过程，目标是学明白 nnU-Net 的**自适应预处理、模型集成、后处理**三条设计思路，并理解它成为医学影像分割领域基准的原因。

- 学习任务：MSD Task04 Hippocampus（数据集 ID = 4），数据小，用于快速迭代
- 考核任务：MSD Task02 Heart（数据集 ID = 2），用于最终对标官方精度
- 运行环境：Windows + conda 环境 `nnunet`，RTX 4060 Laptop（8GB 显存）
- 命令记录日期：2026-09-20

## 一、目录约定

训练数据、预处理结果、模型权重分三个目录存放，nnU-Net 通过环境变量找它们：

| 环境变量 | 路径 | 内容 |
| --- | --- | --- |
| `nnUNet_raw` | `pytorch\nnunet\nnUNet_raw` | 原始数据（nnU-Net 格式） |
| `nnUNet_preprocessed` | `pytorch\nnunet\nnUNet_preprocessed` | 指纹、plan、预处理后的数据 |
| `nnUNet_results` | `pytorch\nnunet\nnUNet_results` | 训练出的模型权重与日志 |

三个目录已在 `.gitignore` 中排除，不进版本库。终端输出存档在 `pytorch\nnunet\logs\`。

## 二、Day 0 完整命令记录

### 1. 建环境

```powershell
conda create -n nnunet python=3.11 -y
conda activate nnunet
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install nnunetv2
```

官方文档特别强调：**必须先装好 PyTorch，再装 nnunetv2**，顺序不能反。

### 2. 设置环境变量

会话级（只在当前终端有效，每次新开终端都要执行）：

```powershell
conda activate nnunet
$env:nnUNet_raw          = 'E:\code\PycharmProjects\pythonProject\pytorch\nnunet\nnUNet_raw'
$env:nnUNet_preprocessed = 'E:\code\PycharmProjects\pythonProject\pytorch\nnunet\nnUNet_preprocessed'
$env:nnUNet_results      = 'E:\code\PycharmProjects\pythonProject\pytorch\nnunet\nnUNet_results'
Get-ChildItem Env:nnUNet*
```

持久化（写用户环境变量，**必须关掉终端重开才生效**）：

```powershell
[Environment]::SetEnvironmentVariable('nnUNet_raw','E:\code\PycharmProjects\pythonProject\pytorch\nnunet\nnUNet_raw','User')
[Environment]::SetEnvironmentVariable('nnUNet_preprocessed','E:\code\PycharmProjects\pythonProject\pytorch\nnunet\nnUNet_preprocessed','User')
[Environment]::SetEnvironmentVariable('nnUNet_results','E:\code\PycharmProjects\pythonProject\pytorch\nnunet\nnUNet_results','User')
```

读回验证（这条读的是注册表，不依赖当前窗口）：

```powershell
[Environment]::GetEnvironmentVariable('nnUNet_raw','User')
```

### 3. 下载数据

```powershell
curl.exe -L -o C:\Users\35314\Task04_Hippocampus.tar https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar
curl.exe -L -o C:\Users\35314\Task02_Heart.tar       https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar
```

下载体积：Hippocampus 约 30 MB，Heart 约 0.42 GB（AWS 直链，来自 MSD 官网）。

### 4. 解压并转换格式

```powershell
New-Item -ItemType Directory -Force E:\code\PycharmProjects\pythonProject\pytorch\data\msd | Out-Null
tar -xf C:\Users\35314\Task04_Hippocampus.tar -C E:\code\PycharmProjects\pythonProject\pytorch\data\msd

nnUNetv2_convert_MSD_dataset -h
nnUNetv2_convert_MSD_dataset -i E:\code\PycharmProjects\pythonProject\pytorch\data\msd\Task04_Hippocampus

Get-ChildItem $env:nnUNet_raw
```

`-i` 指向解压出来的任务目录；数据集 ID 由任务编号自动决定（Task04 → Dataset004），除非用 `-overwrite_id` 覆盖。

### 5. 提取指纹、生成 plan、预处理

```powershell
nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity
```

Windows 上如果多进程报错或卡住，加上进程数限制重跑：

```powershell
nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity -np 4
```

## 三、Day 0 实际结果（2026-09-20）

全部来自 `logs\20260920-day0-环境与数据准备.txt` 的原始输出。

- 转换成功，生成 `nnUNet_raw\Dataset004_Hippocampus`，数据集 ID 确认为 4
- 完整性校验通过：`verify_dataset_integrity Done.`
- 共 **260 例**数据，指纹提取 8 秒（31 it/s）
- 预处理：2D 配置 260 例用 2 分 05 秒，3D fullres 配置 260 例用 22 秒
- plan 文件保存到 `nnUNet_preprocessed\Dataset004_Hippocampus\nnUNetPlans.json`

自动生成的两个配置（括号里是论文 Table 1 对 Hippocampus 的记录，两者完全一致）：

| 项目 | 2D 配置 | 3D fullres 配置 |
| --- | --- | --- |
| 输入 patch | 56 × 40 | 40 × 56 × 40 |
| batch size | 366 | 9 |
| 目标间距 | [1.0, 1.0] | [1.0, 1.0, 1.0] |
| 归一化 | ZScoreNormalization | ZScoreNormalization |
| 网络深度 | 4 级 | 4 级 |
| 每级通道数 | 32 / 64 / 128 / 256 | 32 / 64 / 128 / 256 |
| 归一化层 | InstanceNorm2d | InstanceNorm3d |
| batch dice | 是 | 否 |

**自适应生效的直接证据（第一次观测）**：

```
Dropping 3d_lowres config because the image size difference to 3d_fullres is too small.
3d_fullres: [36. 50. 35.], 3d_lowres: [36, 50, 35]
```

nnU-Net 判断低分辨率配置相对全分辨率没有降采样空间，于是直接放弃了级联方案；后续预处理阶段也相应地跳过了 `3d_lowres`。也就是说这份 plan 的网络数量、patch、batch 都是按这份数据的统计量算出来的，不是固定模板。

另一条官方提示（信息记录，本次不影响结果）：

```
INFO: You are using the old nnU-Net default planner. We have updated our recommendations.
```

官方现在有更新的推荐预设，本次按默认 planner 跑，是为与论文报告的数字保持一致。

## 四、踩过的坑

1. **环境变量为空**：报错 `Could not find a dataset with the ID 4 ... nnUNet_raw=None`。原因是变量只在旧终端窗口里设过，窗口一关就失效；`[Environment]::SetEnvironmentVariable` 的 User 级设置也不会改动已经打开的窗口。解决：要么每次新终端重设会话级变量，要么设成持久化后重开终端。
2. **装包顺序**：先 PyTorch 后 nnunetv2。
3. **先转换再跑**：`nnUNet_raw` 下没有 `DatasetXXX` 目录时，`plan_and_preprocess` 会直接失败，但报错信息是"找不到数据集 ID"，容易被误认为 ID 写错了。

## 五、下一步

- 精读 `dataset_fingerprint.json` 与 `nnUNetPlans.json`，把"统计量 → 决定"的对应关系整理成表
- 对 Heart（Task02）跑一次 planning，与 Hippocampus 的 plan 做并排对比
- 用 `-gpu_memory_target 8` 重跑 planning，观察同一份数据在 8G 显存目标下 batch size 如何变化
- 用 `nnUNetTrainerBenchmark_5epochs` 实测单轮耗时，再决定训练折数
