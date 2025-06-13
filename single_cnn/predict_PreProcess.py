from PIL import Image
import os
from CLAHE import func1
from info import info_out
import csv

def crop_and_save_images(base_path, output_dir):
    """
        读取csv文件，截取读数区域
        保存读数区域图，输出labels
    """
    os.makedirs(output_dir, exist_ok=True)
    # 定义文件路径
    labels_path = os.path.join(base_path, 'labels.csv')
    images_dir = base_path
    #print(labels_path)
    # 检查文件是否存在
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.csv not found in {base_path}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"images directory not found in {base_path}")

    # 读取CSV文件
    with open(labels_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            filename = row['filename']#+'.jpg'
            try:
                xmin = int(row['xmin'])
                ymin = int(row['ymin'])
                xmax = int(row['xmax'])
                ymax = int(row['ymax'])
            except ValueError as e:
                info_out(f"[crop_and_save_images ]Error parsing coordinates for {filename}: {e}")
                continue

            # 构建图像路径
            image_path = os.path.join(images_dir, filename)
            #print(image_path)
            if not os.path.exists(image_path):
                info_out(f"[crop_and_save_images] Image {filename} not found in images directory")
                continue

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    xmin = max(0, min(xmin, width))
                    xmax = max(0, min(xmax, width))
                    ymin = max(0, min(ymin, height))
                    ymax = max(0, min(ymax, height))

                    if xmin >= xmax or ymin >= ymax:
                        info_out(f"[crop_and_save_images] Invalid coordinates for {filename}: ({xmin},{ymin})-({xmax},{ymax})")
                        continue

                    cropped_img = img.crop((xmin, ymin, xmax, ymax))

                    # 保存截取的图像
                    output_path = os.path.join(output_dir, f"{filename}")
                    cropped_img.save(output_path)
                    #print(f"Saved cropped image: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def split_images_parts(input_folder, output_base_folder):
    """
        将读数区域均分为六份
    """
    # 确保输出基础目录存在
    os.makedirs(output_base_folder, exist_ok=True)
    # 获取输入文件夹中的所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(image_extensions)]
    if not image_files:
        info_out(f"[split_images_parts] 在文件夹 {input_folder} 中没有找到图片文件")
        return
    info_out(f"[split_images_parts] 找到 {len(image_files)} 张图片需要处理")

    for image_file in image_files:
        original_path = os.path.join(input_folder, image_file)
        try:
            with Image.open(original_path) as img:
                width, height = img.size
                part_width = width // 6
                for i in range(6):
                    left = i * part_width
                    right = (i + 1) * part_width if i < 5 else width

                    # 切割图片
                    cropped_img = img.crop((left, 0, right, height))

                    # 保存切割后的图片
                    output_path = os.path.join(output_base_folder, f"{image_file.split('.')[0]}_part_{i + 1}{os.path.splitext(image_file)[1]}")
                    cropped_img.save(output_path)

            #info_out(f"[split_images_parts] 成功处理: {image_file} → 保存到 {output_base_folder}")
        except Exception as e:
            info_out(f"[split_images_parts] 处理图片 {image_file} 时出错: {str(e)}")

def image_process():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # os.path.join(os.path.dirname(current_dir), "images")  # 假设上一级目录中有images文件夹
    input_folder = r'../meter_regions1'
    # 设置输出基础文件夹（上一级目录中的split_images）
    # os.path.join(os.path.dirname(current_dir), "split_images")
    output_base_folder = r'../split_images/images'
    split_images_parts(input_folder, output_base_folder)
    print("所有图片处理完成！")

def pre_process(img_dir):
    for img in os.listdir(img_dir):
        func1(img_dir, img)