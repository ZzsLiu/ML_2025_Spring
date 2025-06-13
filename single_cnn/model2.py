import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        """
            修改说明：
            1. 增加第四层卷积，通道数从256扩展到512
            2. 调整池化层位置，确保特征图尺寸逐步减小
            3. 保持批归一化和全局平均池化结构
        """
        # 卷积层部分（四层卷积）
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 1 -> 64
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 -> 128
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128 -> 256
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 新增第四层 256 -> 512
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)  # 2x2 最大池化

        # 使用全局平均池化（GAP）
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 输出固定为 [batch, 512, 1, 1]

        # 全连接层部分（输入维度调整为512）
        self.fc1 = nn.Linear(512, 128)  # 输入维度改为512
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一层：56*56 -> 28*28
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> [b, 64, 14, 14]

        # 第二层：28*28 -> 14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> [b, 128, 7, 7]

        # 第三层：14x14 -> 7x7 (7//2=3)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> [b, 256, 3, 3]

        # 新增第四层：7x7 -> 3*3
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # -> [b, 512, 1, 1]

        # 全局平均池化：将 [b, 512, 1, 1] -> [b, 512, 1, 1] (尺寸不变，更规范)
        x = self.gap(x)

        # 展平：[b, 512, 1, 1] -> [b, 512]
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)