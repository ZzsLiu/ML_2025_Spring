import torch
from torch.utils.data import DataLoader
from dataset import DigitDataset
from model import DigitCNN
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import time
"""
    batch size：控制每次训练用多少数据，影响效率和稳定性。
    epoch：控制整体训练次数，影响训练充分度。
    learning rate：控制每次参数调整幅度，影响训练速度和效果
"""
batch_size=64
epochs=80
lr=0.002

def main():
    print(torch.__version__)  # 查看 PyTorch 版本
    print(torch.version.cuda)  # 查看 PyTorch 使用的 CUDA 版本
    print(torch.cuda.is_available())  # 检查 CUDA 是否可用


    # 数据加载(启用多线程加载图像)
    info_out(f"训练开始：batch_size={batch_size} epochs={epochs} lr={lr}")
    train_dataset = DigitDataset(r"..\split_images\train\labels.csv", r"..\split_images\train\images")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)


    # 模型与训练配置
    #model = DigitCNN()
    # 模型与训练配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #default_logger.info(f"[batch_size={batch_size} epoches={epoches} lr={lr}]训练开始：")
    # 训练过程
    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        running_loss = 0.0
        for images, labels in  tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        info_out(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), f"digit_model_{time.time()}.pth")
    info_out("训练完成，模型已保存。")

def info_out(str):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("app.log", "a", encoding="utf-8") as f:
        f.write(f"[{now_str}] {str}\n")

if __name__ == "__main__":
    main()