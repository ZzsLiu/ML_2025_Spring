import threading
import torch
from torch.utils.data import DataLoader
from dataset import DigitDataset
from model3 import DigitCNN
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import time


def main():
    args = [
        [64, 200, 0.001],
        [64, 160, 0.001],
        [64, 120, 0.001],
        [64, 80, 0.001],
    ]
    """
        batch size：控制每次训练用多少数据，影响效率和稳定性。
        epoch：控制整体训练次数，影响训练充分度。
        learning rate：控制每次参数调整幅度，影响训练速度和效果
    """
    # 创建线程列表
    threads = []

    # 最大同时运行的线程数（根据你的 GPU 数量和性能调整）
    max_threads = 4

    # 遍历所有超参数组合
    thread_id = 0
    for b, e, l in args:
        # 如果当前运行的线程数达到最大值，等待其中一个线程完成
        while threading.active_count() > max_threads:
            time.sleep(1)  # 暂停 1 秒，等待线程完成

        # 创建一个新的线程来运行训练任务
        thread = threading.Thread(target=func, args=(b, e, l, thread_id))
        thread.start()  # 启动线程
        threads.append(thread)  # 将线程添加到列表中
        thread_id += 1
    # 等待所有线程完成
    for thread in threads:
        thread.join()

    print("All training tasks completed.")


def func(batch_size, epochs, lr, thread_id):
    """
    训练函数，用于在指定超参数下训练模型。

    参数:
    - batch_size (int): 每次训练的批量大小。
    - epochs (int): 训练的总轮数。
    - lr (float): 学习率。
    - thread_id (int): 线程 ID，用于区分不同线程的进度条或其他输出。
    """

    print(torch.__version__)  # 查看 PyTorch 版本
    print(torch.version.cuda)  # 查看 PyTorch 使用的 CUDA 版本
    print(torch.cuda.is_available())  # 检查 CUDA 是否可用

    # 使用自定义的 DigitDataset 加载训练数据
    info_out(f"训练开始：batch_size={batch_size} epochs={epochs} lr={lr}")

    train_dataset = DigitDataset(
        csv_path=r"..\split_images\train\labels.csv",  # 训练集标签文件路径
        image_dir=r"..\split_images\train\images",  # 训练集图像文件夹路径
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # 每次加载的批量大小
        shuffle=True,  # 是否在每个 epoch 开始时打乱数据。 每个epoch打乱数据顺序，避免模型学习到顺序偏差
        num_workers=0  # 数据加载的线程数（当前设置为 0，表示不使用多线程）
    )

    # 初始化模型，并将其移动到指定的设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = DigitCNN().to(device)

    # 定义损失函数，这里使用交叉熵损失（CrossEntropyLoss），适用于分类任务
    criterion = nn.CrossEntropyLoss()
    # 定义优化器，这里使用 Adam 优化器，并设置学习率为 lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(epochs):
        # 将模型设置为训练模式（启用 Dropout 和 BatchNorm 的训练行为）
        model.train()
        total, correct = 0, 0   # 用于计算准确率的变量
        running_loss = 0.0      # 用于累积每个 epoch 的损失值

        # 遍历 DataLoader 中的每个批次
        # 使用 tqdm 显示进度条，desc 参数设置进度条的描述，position 参数设置进度条的位置（用于多线程环境下固定进度条位置）
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", position=thread_id, ncols=20):
            # 将当前批次的数据移动到指定的设备（GPU 或 CPU）
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播：将输入数据传入模型，得到预测结果
            outputs = model(images)

            # 计算损失值：将模型的预测结果与真实标签进行比较，计算交叉熵损失
            loss = criterion(outputs, labels)

            # 反向传播：清空之前的梯度，计算当前批次的梯度
            optimizer.zero_grad()  # 清空模型参数的梯度，避免梯度累积。
            loss.backward()  # 反向传播，计算梯度 自动计算所有参数的梯度

            # 参数更新：根据计算出的梯度，使用优化器更新模型参数
            optimizer.step()

            # 累积损失值，用于计算整个 epoch 的平均损失
            running_loss += loss.item()

            # 计算当前批次的预测准确率
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果（取最大值的索引）
            total += labels.size(0)  # 累积当前批次的样本总数
            correct += (predicted == labels).sum().item()  # 累积当前批次中预测正确的样本数

        # 计算当前 epoch 的平均损失和准确率
        acc = 100 * correct / total  # 准确率 = (正确预测的样本数 / 总样本数) * 100%
        avg_loss = running_loss / len(train_loader)  # 平均损失 = 总损失 / 批次数量

        # 记录当前 epoch 的训练结果（损失和准确率）
        info_out(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

        # 在每个 epoch 结束后调用
        #scheduler.step()

        # 在每个 epoch 结束后评估验证集
        model.eval()



    # 保存模型
    torch.save(model.state_dict(), f"./pth/5/digit_model_{batch_size}_{epochs}_{lr}.pth")
    info_out("训练完成，模型已保存。")


def info_out(str):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("app.log", "a", encoding="utf-8") as f:
        f.write(f"[{now_str}] {str}\n")


if __name__ == "__main__":
    main()