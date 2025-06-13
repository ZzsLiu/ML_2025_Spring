import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        """
            修改说明：
            1. 增加第五层卷积，通道数从512扩展到1024
            2. 调整网络结构以保持合理的特征图尺寸
            3. 注意：由于多次池化，最终特征图尺寸会变得很小
        """
        # 卷积层部分（五层卷积）
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 1 -> 64
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 -> 128
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128 -> 256
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 256 -> 512
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # 新增第五层 512 -> 1024
        self.bn5 = nn.BatchNorm2d(1024)

        # 修改池化策略，避免特征图过小
        self.pool1 = nn.MaxPool2d(2, 2)  # 前几层使用2x2池化
        self.pool2 = nn.MaxPool2d(2, 2)  # 中间层使用2x2池化
        self.pool3 = nn.MaxPool2d(2, 2)  # 后几层可能考虑其他池化策略
        self.pool4 = nn.MaxPool2d(2, 2)  # 后几层可能考虑其他池化策略

        # 使用全局平均池化（GAP）
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 输出固定为 [batch, 1024, 1, 1]

        # 全连接层部分（输入维度调整为1024）
        self.fc1 = nn.Linear(1024, 256)  # 增加中间层维度
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一层：56*56 -> 28*28
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # -> [b, 64, 14, 14]

        # 第二层：28*28 -> 14x14
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # -> [b, 128, 7, 7]

        # 第三层：14x14 -> 7x7 (7//2=3)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # -> [b, 256, 3, 3]

        # 第四层：7x7 -> 3x3 (3//2=1)
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))  # 注意：这里不再池化 -> [b, 512, 1, 1]

        # 第五层：3x3 -> 3x3 (保持尺寸)
        x = F.relu(self.bn5(self.conv5(x)))  # -> [b, 1024, 1, 1]

        # 全局平均池化：将 [b, 1024, 1, 1] -> [b, 1024, 1, 1]
        x = self.gap(x)

        # 展平：[b, 1024, 1, 1] -> [b, 1024]
        x = x.view(x.size(0), -1)

        # 扩展的全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)