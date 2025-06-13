from predict_PreProcess import *
from CLAHE import make_gray_scale
import torch
from model2_1 import DigitCNN
import torchvision.transforms as transforms
import os
import pandas as pd
#F:\ML2025_CV_NEW\Test2
test_path = "F:\ML2025_CV_NEW\Test2"

pth_dir='./pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitCNN()
model = model.to(device)

transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
])
def model_pth_set(pth_file_path):
    model.load_state_dict(torch.load(pth_file_path))
    model.eval()
def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

if __name__ == '__main__':
    # 前期数据处理
    box_images = "../data/box_images"
    digit_images = "../data/digit_images"
    grayscale_images = "../data/grayscale_images"
    crop_and_save_images(test_path, box_images)
    split_images_parts(box_images, digit_images)
    make_gray_scale(digit_images, grayscale_images)

    results=[]
    for pth in os.listdir(pth_dir):
        if pth.split('.')[-1] == 'pth':
            pth_path=os.path.join(pth_dir, pth)
            model_pth_set(pth_path)
            df=pd.read_csv(os.path.join(test_path, 'labels.csv'))
            for idx,row in df.iterrows():
                predict_label=0
                for i in range(1,7):
                    part_img = str(row['filename'].replace('.jpg',''))+ f"_part_{i}.jpg"
                    pre=predict(os.path.join(grayscale_images,part_img))
                    predict_label = predict_label * 10 + pre
                predict_label = "{:.1f}".format(float(predict_label / 10))
                results.append({'id': row['filename'].replace('.jpg',''), 'number': predict_label})
            result_df = pd.DataFrame(results)
            result_df['number'] = result_df['number'].astype(str)
            result_df.to_csv(os.path.join(test_path,f'results_{pth}.csv'), index=False)




