from PIL import Image
import os
import pandas as pd
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

    # 检查文件是否存在
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.csv not found in {base_path}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"images directory not found in {base_path}")

    # 读取CSV文件
    with open(labels_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            filename = row['filename']
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

def get_labels(input_csv_path,output_csv_path):
    """
        处理CSV文件：
        1. 读取file_name列，添加'_meter_1'后缀
        2. 读取number列，每个值乘以10
        3. 将处理后的数据保存到新CSV文件
            每行数据存储六条，分别处理其label
            xxxx_meter_1_part_i.jpg
        参数:
            input_csv_path (str): 输入CSV文件路径
            output_csv_path (str): 输出CSV文件路径
    """
    try:
        # 读取 CSV 文件
        df = pd.read_csv(input_csv_path)

        # 检查必需的列是否存在
        if 'filename' not in df.columns or 'number' not in df.columns:
            raise ValueError("CSV 文件中必须包含 'filename' 和 'number' 列")
        # 处理数据：每行扩展为 6 行
        new_rows = []
        for _, row in df.iterrows():
            filename = row['filename'].replace('.jpg', '')  # 去除 .jpg 后缀
            label = row['number']*10
            # 生成 6 个新文件名和对应的 label
            for i in range(1, 7):
                new_filename = f"{filename}_meter_1_part_{i}.jpg"

                # 计算新 label
                if i == 1:
                    new_label = label // 100000
                elif i == 2:
                    new_label = (label // 10000) % 10
                elif i == 3:
                    new_label = (label // 1000) % 10
                elif i == 4:
                    new_label = (label // 100) % 10
                elif i == 5:
                    new_label = (label // 10) % 10
                elif i == 6:
                    new_label = label % 10

                new_rows.append({
                    'filename': new_filename,
                    'label': int(new_label)
                })
        # 创建新的 DataFrame
        result_df = pd.DataFrame(new_rows, columns=['filename', 'label'])

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        # 保存到新 CSV 文件
        result_df.to_csv(output_csv_path, index=False)
        print(f"处理完成，结果已保存到: {output_csv_path}")

    except Exception as e:
        print(f"处理 CSV 文件时出错: {str(e)}")
    pass
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

if __name__ == "__main__":
    #pre_process(r'../meter_regions1')
    image_process()
    #get_labels(r'../任务一标签（新）.csv',r'../split_images/labels.csv')