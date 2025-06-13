import threading
import torch
from torch.utils.data import DataLoader
from dataset2_1 import DigitDataset
from model2_1 import DigitCNN
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import time
import os
import pandas as pd
from torchvision import transforms
from PIL import Image
from info import info_out

# 创建日志目录
os.makedirs("logs", exist_ok=True)
# 创建模型保存目录
os.makedirs("pth/best_models", exist_ok=True)


def main():
    args = [
        [64, 200, 0.001],
        [64, 160, 0.001],
        [64, 120, 0.001],
        [64, 80, 0.001],
    ]

    # 创建线程列表
    threads = []
    max_threads = 4  # 最大同时运行的线程数

    # 遍历所有超参数组合
    thread_id = 0
    for b, e, l in args:
        # 等待可用线程槽位
        while threading.active_count() > max_threads:
            time.sleep(1)

        # 创建新线程运行训练任务
        thread = threading.Thread(target=train_model, args=(b, e, l, thread_id))
        thread.start()
        threads.append(thread)
        thread_id += 1

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    print("All training tasks completed.")


def train_model(batch_size, epochs, lr, thread_id):
    """训练函数，包含完整训练流程和优化策略"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info_out(f"线程{thread_id}: 使用设备 {device}", thread_id)
    info_out(f"线程{thread_id}: 开始训练 - batch_size={batch_size}, epochs={epochs}, lr={lr}", thread_id)

    # ================= 数据增强与加载 =================
    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转 ±10度
        transforms.RandomAffine(0, shear=10),  # 随机剪切变换
        transforms.ColorJitter(contrast=0.2),  # 对比度扰动
        #transforms.RandomResizedCrop(56, scale=(0.9, 1.1)),  # 随机缩放裁剪
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 验证集基础转换
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 加载数据集
    train_dataset = DigitDataset(
        csv_path=r"..\split_images\train\labels.csv",
        image_dir=r"..\split_images\train\images",
        transform=train_transform
    )
    val_dataset = DigitDataset(
        csv_path=r"..\split_images\val\labels.csv",
        image_dir=r"..\split_images\val\images",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4  # 增加工作线程提高数据加载速度
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # ================= 模型初始化 =================
    model = DigitCNN().to(device)

    # ================= 训练配置 =================
    # 优化器 (AdamW 比 Adam 更适合CNN)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 检查PyTorch版本并调整调度器
    torch_version = torch.__version__
    info_out(f"线程{thread_id}: PyTorch版本: {torch_version}", thread_id)

    # 兼容旧版本PyTorch的调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    # 损失函数 (带标签平滑)
    # 兼容旧版本PyTorch
    try:
        # 尝试使用新版本支持标签平滑的损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    except TypeError:
        # 旧版本不支持label_smoothing参数
        info_out("线程{thread_id}: 当前PyTorch版本不支持内置标签平滑，使用自定义实现", thread_id)
        criterion = nn.CrossEntropyLoss()

    # ================= 训练变量初始化 =================
    best_val_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    early_stop_patience = 10  # 连续10个epoch验证准确率无提升则停止

    # ================= 训练循环 =================
    for epoch in range(epochs):
        # ----- 训练阶段 -----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        train_iter = tqdm(train_loader,
                          desc=f"线程{thread_id} Epoch {epoch + 1}/{epochs} [训练]",
                          position=thread_id,
                          ncols=80)

        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计指标
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 更新进度条
            train_iter.set_postfix(loss=loss.item())

        # 计算训练指标
        train_loss = train_loss / train_total
        train_acc = 100.0 * train_correct / train_total

        # ----- 验证阶段 -----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        # with torch.no_grad():
        #     for images, labels in val_loader:
        #         images, labels = images.to(device), labels.to(device)
        #
        #         outputs = model(images)
        #         loss = criterion(outputs, labels)
        #
        #         val_loss += loss.item() * images.size(0)
        #         _, predicted = torch.max(outputs, 1)
        #         val_total += labels.size(0)
        #         val_correct += (predicted == labels).sum().item()
        #
        # # 计算验证指标
        # val_loss = val_loss / val_total
        # val_acc = 100.0 * val_correct / val_total

        basic_labels_df = pd.read_csv("../split_images/任务一标签（新）.csv")
        with torch.no_grad():
            for idx, row in basic_labels_df.iterrows():
                basic_img_name = row['filename'].replace('.jpg', '')
                if os.path.isfile(f"../split_images/val/images/{basic_img_name}_meter_1_part_1.jpg"):
                    val_total += 1
                    true_label = int(row['number'] * 10)

                    predict_label = 0
                    for i in range(1, 7):
                        img_name = f"{basic_img_name}_meter_1_part_{i}.jpg"

                        image = Image.open(f"../split_images/val/images/{img_name}")
                        image = val_transform(image).unsqueeze(0).to(device)
                        output = model(image)
                        _, predicted = torch.max(output, 1)

                        pre = predicted.item()
                        predict_label = predict_label * 10 + pre

                    if predict_label == true_label:
                        val_correct += 1

        # 计算验证指标
        val_acc = 100.0 * val_correct / val_total


        # 更新学习率
        scheduler.step(val_acc)

        # 记录日志
        log_msg = (f"线程{thread_id} Epoch {epoch + 1}/{epochs}: "
                   f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}% | "
                   f"验证准确率: {val_correct}/{val_total} {val_acc:.2f}% | "
                   f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        info_out(log_msg, thread_id)

        # ----- 模型保存与早停检查 -----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            early_stop_counter = 0

            # 保存最佳模型
            model_path = f"./pth/best_models/model_thread{thread_id}_epoch{epoch + 1}_acc{val_acc:.2f}.pth"
            torch.save(model.state_dict(), model_path)
            info_out(f"线程{thread_id}: 保存最佳模型 @ Epoch {epoch + 1}, 验证准确率: {val_acc:.2f}%", thread_id)
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                info_out(f"线程{thread_id}: 早停触发 @ Epoch {epoch + 1}, 最佳验证准确率: {best_val_acc:.2f}% @ Epoch {best_epoch}", thread_id)
                break

    # 最终模型保存
    final_model_path = f"./pth/final_model_thread{thread_id}.pth"
    torch.save(model.state_dict(), final_model_path)
    info_out(f"线程{thread_id}: 训练完成，最终模型已保存。最佳验证准确率: {best_val_acc:.2f}%", thread_id)


# def info_out(message, thread_id=None):
#     """带线程ID的日志输出"""
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     log_entry = f"[{timestamp}]"
#
#     if thread_id is not None:
#         log_entry += f" [线程{thread_id}]"
#
#     log_entry += f" {message}"
#
#     print(log_entry)
#     with open(f"logs/training_log_{thread_id or 'main'}.txt", "a", encoding="utf-8") as f:
#         f.write(log_entry + "\n")


if __name__ == "__main__":
    main()