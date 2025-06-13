# dataset.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class DigitDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path (string): CSV文件路径，包含图像文件名和标签。
            image_dir (string): 图像文件夹路径。
            transform (callable, optional): 应用于图像的变换操作。
        """
        self.labels = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform  # 添加transform参数

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # 转换为灰度图像
        label = self.labels.iloc[idx, 1]

        # 应用变换（如果提供了transform）
        if self.transform:
            image = self.transform(image)
        else:
            # 如果没有提供transform，则使用默认转换
            image = torch.tensor([image], dtype=torch.float32) / 255.0

        return image, label