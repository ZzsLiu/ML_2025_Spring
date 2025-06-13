import cv2
import os
"""
    自适应直方图均衡化
"""
def func1(img_dir,img_name):
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    corrected = clahe.apply(img)
    cv2.imwrite(f'{img_dir}/{img_name}', corrected)


def make_gray_scale(img_dir, output_dir):
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        # 读取图像为灰度图
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # 应用CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        corrected = clahe.apply(img)
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        # 构造输出图像路径（保留原始文件名）
        output_path = os.path.join(output_dir, img_name)
        # 保存处理后的图像
        cv2.imwrite(output_path, corrected)

