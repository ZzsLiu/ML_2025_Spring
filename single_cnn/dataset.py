import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class DigitDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        # 尝试使用 gbk 编码读取
        try:
            self.data = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.data = pd.read_csv(csv_path, encoding='gbk')

        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((56, 56)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])
        img_path = os.path.join(self.image_dir, img_name)
        #default_logger.info(f"Loading image at: {img_path}")
        image = Image.open(img_path)
        image = self.transform(image)
        return image, label