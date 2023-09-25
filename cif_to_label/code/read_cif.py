import os
import shutil
import cv2


def convert_image(source_folder, target_folder):
    try:
        # 确保目标文件夹存在，如果不存在则创建
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 遍历源文件夹中的文件
        tiff_count = 85
        for i, filename in enumerate(os.listdir(source_folder)):
            source_path = os.path.join(source_folder, filename)

            # 检查是否为tiff文件
            if os.path.isfile(source_path) and filename.lower().endswith('.tiff'):
                # 重命名为 molecule_i.jpg
                source_path_png = os.path.join(source_folder, filename.split('.')[0]+'.png')
                target_filename = f"molecule_{tiff_count}.jpg"
                target_filename_png = f"molecule_{tiff_count}.png"
                tiff_count += 1

                # 转换tiff为jpg
                im = cv2.imread(source_path)
                target_path = os.path.join(target_folder+'training', target_filename)
                target_path_png = os.path.join(target_folder+'label_annotation', target_filename_png)
                cv2.imwrite(target_path, im)
                shutil.copy2(source_path_png, target_path_png)
                print(f"转换并复制文件：{filename} 到 {target_folder} 为 {target_filename}")
        print(f"转换并复制 {tiff_count} 个tiff文件为jpg格式")
    except Exception as e:
        print(f"发生错误：{str(e)}")


# 指定源文件夹和目标文件夹
source_folder = "D:\\project\\deep_learning_recovery\\Xiong_cifs and images\\cifs and images\\0616-saifen-simulation-25.14pm"  # 替换为源文件夹的实际路径
target_folder = "F:\\MOLECULE dataset\\"  # 替换为目标文件夹的实际路径

# 调用函数进行tiff格式转换和复制操作
convert_image(source_folder, target_folder)

