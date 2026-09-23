import torch
from torch import nn

class DoubleConv(nn.Module):
    """U-Net 的基本积木：两次 3x3 卷积，每次后面接批归一化和 ReLU。
    padding=1 让卷积前后宽高不变，所以过完这个块，尺寸和进来时一样。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """下采样：先 2x2 最大池化把宽高减半，再过一次双卷积。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    """上采样：宽高翻倍，再和编码器对应层的特征拼接，最后过双卷积。"""
    def __init__(self, in_channels):
        super().__init__()
        # 转置卷积：可以学习的放大操作，通道数同时减半
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, in_channels // 2)
    def forward(self, x, skip):
        x = self.up(x)  # 宽高翻倍，通道减半
        x = torch.cat([skip, x], dim=1)  # 拼接跳连：沿着通道维接起来
        return self.conv(x)  # 通道数减半，尺寸不变

class UNet(nn.Module):
    """输入 (B, 3, H, W) 的 RGB 图像，输出 (B, 1, H, W) 的分割得分。

       输出不经过 sigmoid，直接交给 BCEWithLogitsLoss。
       宽高需要能被 16 整除（经过 4 次池化），256 和 512 都满足。
       """
    def __init__(self, in_channels = 3, out_channels = 1, base_channels = 64):
        super().__init__()
        c1 = base_channels  # 64
        c2 = base_channels * 2  # 128
        c3 = base_channels * 4  # 256
        c4 = base_channels * 8  # 512

        # 编码器：尺寸一路减半，通道一路翻倍
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c4 * 2)

        # 解码器：尺寸一路翻倍，通道一路减半（注意 up1 的输入是瓶颈层的 c4*2）
        self.up1 = Up(c4 * 2)
        self.up2 = Up(c4)
        self.up3 = Up(c3)
        self.up4 = Up(c2)

        # 输出层：1x1 卷积把 c1 个通道压成 out_channels，宽高不变
        self.outc = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)  # 64   H     W
        x2 = self.down1(x1)  # 128  H/2   W/2
        x3 = self.down2(x2)  # 256  H/4   W/4
        x4 = self.down3(x3)  # 512  H/8   W/8
        x5 = self.down4(x4)  # 1024 H/16  W/16

        x = self.up1(x5, x4)  # 512  H/8   W/8
        x = self.up2(x, x3)  # 256  H/4   W/4
        x = self.up3(x, x2)  # 128  H/2   W/2
        x = self.up4(x, x1)  # 64   H     W
        return self.outc(x)  # 1    H     W

def dice_coeff(pred_logits, targets, threshold=0.5, eps=1e-6):
    """按图计算 Dice，再对这批图取平均（评估用）。

    pred_logits: (B, 1, H, W) 模型的原始输出
    targets:     (B, 1, H, W) 取值 0/1 的标签
    返回一个标量，约等于这批图各自的 Dice 平均值。
    """
    probs = torch.sigmoid(pred_logits)          # logits -> 概率
    preds = (probs > threshold).float()         # 阈值化成 0/1，这才是指标用的形式

    preds = preds.flatten(1)                    # (B, 1, H, W) -> (B, H*W)
    targets = targets.flatten(1)

    intersection = (preds * targets).sum(dim=1)              # 每张图的交集
    union = preds.sum(dim=1) + targets.sum(dim=1)            # 每张图的面积之和

    dice = (2 * intersection + eps) / (union + eps)          # 每张图的 Dice
    return dice.mean()                                       # 再对这批图取平均


def dice_loss(pred_logits, targets, eps=1e-6):
    """软 Dice 损失（训练用）：直接用概率算，不做阈值化，保证梯度能回传。"""
    probs = torch.sigmoid(pred_logits).flatten(1)
    targets = targets.flatten(1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()                     # 损失 = 1 - Dice，越小越好


def combined_loss(pred_logits, targets):
    """BCE + Dice：BCE 提供稳定的逐像素梯度，Dice 直接优化重叠率。"""
    bce = nn.BCEWithLogitsLoss()(pred_logits, targets)
    return bce + dice_loss(pred_logits, targets)

if __name__ == "__main__":
    # net = UNet()
    # x = torch.randn(2, 3, 256, 256)
    #
    # x1 = net.inc(x);     print("inc  ", tuple(x1.shape))
    # x2 = net.down1(x1);  print("down1", tuple(x2.shape))
    # x3 = net.down2(x2);  print("down2", tuple(x3.shape))
    # x4 = net.down3(x3);  print("down3", tuple(x3.shape))
    # x5 = net.down4(x4);  print("down4", tuple(x5.shape))
    #
    # y = net.up1(x5, x4); print("up1  ", tuple(y.shape))
    # y = net.up2(y, x3);  print("up2  ", tuple(y.shape))
    # y = net.up3(y, x2);  print("up3  ", tuple(y.shape))
    # y = net.up4(y, x1);  print("up4  ", tuple(y.shape))
    #
    # logits = net.outc(y); print("outc ", tuple(logits.shape))
    # print("参数量", sum(p.numel() for p in net.parameters()))
    # 造 4 张 8x8 的假标签：上半、左两列、全前景、全背景
    targets = torch.zeros(4, 1, 8, 8)
    targets[0, :, :4, :] = 1
    targets[1, :, :, :2] = 1
    targets[2, :, :, :] = 1

    # 构造"预测完全正确"的假 logits：标签是 1 就给 +10，是 0 就给 -10
    fake_logits = (targets * 2 - 1) * 10

    print("完美预测:", dice_coeff(fake_logits, targets).item())  # 应该接近 1.0
    print("完全反着预测:", dice_coeff(-fake_logits, targets).item())  # 应该接近 0.0
    print("完美预测的损失:", dice_loss(fake_logits, targets).item())  # 应该接近 0.0

    # 模型输出 0.0，经过 sigmoid 就是 0.5（完全不确定）
    # 真实标签是 1，此时损失应该是 -log(0.5) = ln2 ≈ 0.693
    logits = torch.tensor([[[[0.0]]]])
    target = torch.tensor([[[[1.0]]]])
    print(nn.BCEWithLogitsLoss()(logits, target).item())  # 期望 ≈ 0.6931

    # 模型很自信地说"是"：logit = 2.197 对应概率 0.9，损失应该约 0.105
    logits = torch.tensor([[[[2.197]]]])
    print(nn.BCEWithLogitsLoss()(logits, target).item())  # 期望 ≈ 0.105