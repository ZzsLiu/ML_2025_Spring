import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        """
            卷积层设计：
                逐步增加通道数（64→128→256）：随着网络加深，提取的特征从低级（边缘）到高级（数字整体结构）。
                统一使用3x3卷积核：小卷积核能减少参数且保留局部特征。
                padding=1：保持特征图空间尺寸不变（如输入28x28，输出仍是28x28）。
            批归一化：
                加速训练收敛，减少对初始化的敏感度。
                放在卷积层之后、激活函数之前（顺序：Conv→BN→ReLU）。
            池化层：
                逐步降低分辨率（28x28 → 14x14 → 7x7 → 3x3），增强平移不变性
            全局平均池化（GAP）：
                替代传统的展平操作，直接对每个通道取平均值。
                优点：减少参数量（避免全连接层的巨大参数），降低过拟合风险。
            全连接层：
                最终通过两层全连接输出分类结果。
            Dropout：
                防止过拟合，增强模型泛化能力。
        """
        # 卷积层部分（更深 + 更宽 + BN）
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 1 -> 64
        self.bn1 = nn.BatchNorm2d(64)   # 对64通道的输出做归一化
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 -> 128
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128 -> 256
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)  # 2x2 最大池化

        # 使用全局平均池化（GAP）替代全连接层前的展平
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 输出固定为 [batch, 256, 1, 1]

        # 全连接层部分（输入维度固定为 256）
        self.fc1 = nn.Linear(256, 128)  # 不再需要 256 * 3 * 3
        self.dropout = nn.Dropout(0.5)# 50%神经元随机失活
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> [b, 64, 14, 14]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> [b, 128, 7, 7]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> [b, 256, 3, 3]

        # 全局平均池化：将 [b, 256, 3, 3] -> [b, 256, 1, 1]
        x = self.gap(x)

        # 展平：[b, 256, 1, 1] -> [b, 256]
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)