import torch
from PIL import Image
from model2_1 import DigitCNN
import torchvision.transforms as transforms
import pandas as pd
from datetime import datetime
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitCNN()
model = model.to(device)
def model_pth_set(pth_file_path):
    model.load_state_dict(torch.load(pth_file_path))
    model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
def predict(image_path):
    image = Image.open(image_path)
    image = val_transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    #print(f"Predicted digit: {predicted.item()}")
    return predicted.item()

# 示例预测
labels_df = pd.read_csv("../split_images/val/labels.csv")
# for i in range(1,10):
#     print(f"predicting image{i}: ",end="")
#     predict(f"../single_digit/images/00{i}.png")


def pre1():
    total = 13
    correct = 0
    for i in range(1, total+1):

        img_name = f"{i:03}.png"
        # 从CSV里找到对应文件名的标签
        true_label = labels_df.loc[labels_df['file_name'] == img_name, 'label'].values[0]

        pre=predict(f"../single_digit/test_images/{img_name}")
        if pre == true_label:
            correct += 1

        #print(f"predicting image{i} ({img_name}), true label: {true_label},Predicted digit: ", end="")
        #print(pre)

    print(f"Correct: {correct}/{total} {correct / total * 100:.2f}%")

def pre2(pth_file_path):
    correct = 0
    for idx, row in labels_df.iterrows():
        img_name = row['filename']
        true_label = row['label']

        pre = predict(f"../split_images/val/images/{img_name}")
        if pre == true_label:
            correct += 1

        #print(f"predicting image {idx + 1} ({img_name}), true label: {true_label}, Predicted digit: {pre}")

    print(f"Correct: {correct}/{len(labels_df)} {correct / len(labels_df) * 100:.2f}%")

    info_out(f"pre2 {pth_file_path} - Correct: {correct}/{len(labels_df)} {correct / len(labels_df) * 100:.2f}%")


def pre3(pth_file_path):
    correct = 0
    total = 0
    basic_labels_df= pd.read_csv("../split_images/任务一标签（新）.csv")
    for idx,row in basic_labels_df.iterrows():
        basic_img_name = row['filename'].replace('.jpg','')
        if os.path.isfile(f"../split_images/val/images/{basic_img_name}_meter_1_part_1.jpg"):
            total += 1
            true_label = row['number']*10

            predict_label = 0
            for i in range(1,7):
                img_name = f"{basic_img_name}_meter_1_part_{i}.jpg"
                pre = predict(f"../split_images/val/images/{img_name}")
                predict_label = predict_label * 10 + pre

            #print(f"predicting image {idx + 1} ({basic_img_name}), true label: {true_label}, Predicted label: {predict_label}")
            if predict_label == true_label:
                correct += 1

    print(f"Correct: {correct}/{total} {correct / total * 100:.2f}%")
    info_out(f"pre3 {pth_file_path} - Correct: {correct}/{total} {correct / total * 100:.2f}%")


def info_out(str):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("app.log", "a", encoding="utf-8") as f:
        f.write(f"[{now_str}] {str}\n")


if __name__ == "__main__":
    # pth_file_path = "digit_model_32_15_0.001.pth"
    # model_pth_set(pth_file_path)
    # pre2(pth_file_path)

    model_dir = "./pth/"  # 或者你可以改成具体路径如 "./models/"
    pth_files = glob.glob(os.path.join(model_dir, "*.pth"))

    info_out(f"开始读取pth文件进行验证集验证：")
    for pth_file_path in pth_files:
        print(pth_file_path)
        model_pth_set(pth_file_path)
        pre2(pth_file_path)
        pre3(pth_file_path)
    info_out(f"验证结束")