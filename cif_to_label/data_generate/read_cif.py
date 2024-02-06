import os
import shutil
import cv2
import json


def convert_image(source_folder, target_folder, json_file_path):
    name_mapping = {}  # 用来存储文件名映射的字典

    try:
        # 确保目标文件夹存在，如果不存在则创建
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 遍历源文件夹中的文件
        tiff_count = 92
        for filename in os.listdir(source_folder):
            source_path = os.path.join(source_folder, filename)

            # 检查是否为tiff文件
            if os.path.isfile(source_path) and filename.lower().endswith('.png'):
                # 重命名为 molecule_i.jpg
                target_filename = f"molecule_{tiff_count}.png"
                tiff_count += 1

                # 转换tiff为jpg
                im = cv2.imread(source_path)
                target_path = os.path.join(target_folder, target_filename)
                cv2.imwrite(target_path, im)

                # 更新文件名映射字典
                name_mapping[filename] = target_filename

                print(f"Converted and copied file: {filename} to {target_filename}")

        print(f"Converted and copied {tiff_count} tiff files to jpg format")

        # 将映射字典保存为json文件
        with open(json_file_path, 'a+') as f:
            json.dump(name_mapping, f, indent=4)

        print(f"File name mapping saved to {json_file_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


# 指定源文件夹和目标文件夹
source_folder = "D:\\project\\phase_structure\\phase_structure_recognition\\cif_to_label\\new images\\2023"  # 替换为源文件夹的实际路径
target_folder = "F:\\MOLECULE dataset3\\"  # 替换为目标文件夹的实际路径
json_file_path = "F:\\MOLECULE dataset3\\name_mapping.json"  # 替换为要保存json文件的实际路径
os.makedirs(target_folder, exist_ok=True)
# 调用函数进行tiff格式转换和复制操作
convert_image(source_folder, target_folder, json_file_path)
