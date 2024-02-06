import os
import shutil


def copy_images(source_folder, target_folder):
    try:
        target_folder_filename = os.listdir(target_folder)

        # 遍历源文件夹中的文件
        for filename in os.listdir(source_folder):
            source_path = os.path.join(source_folder, filename)
            flag = filename.split('.png')[0] + '_dose'

            # 检查是否为图片文件
            if filename.endswith('.png'):
                for target_filename in target_folder_filename:
                    if target_filename.endswith('.tiff') and flag in target_filename:
                        output_filename = target_filename.split('.')[0] + '.png'
                        target_path = os.path.join(target_folder, output_filename)
                        shutil.copy2(source_path, target_path)
                        print(f"复制文件：{filename} 到 {target_folder}")
        print("复制完成")
    except Exception as e:
        print(f"发生错误：{str(e)}")


# 指定源文件夹和目标文件夹
source_folder = "D:\\project\\phase_structure\\phase_structure_recognition\\cif_to_label\\new images\\xinzeng"  # 替换为源文件夹的实际路径
target_folder = "D:\\project\\phase_structure\\phase_structure_recognition\\cif_to_label\\new images\\2023"  # 替换为目标文件夹的实际路径

# 调用函数进行复制操作
copy_images(source_folder, target_folder)
