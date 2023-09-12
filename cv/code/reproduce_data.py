import os
import shutil


def copy_images(source_folder, target_folder):
    try:
        target_folder_filename = os.listdir(target_folder)

        # 遍历源文件夹中的文件
        for filename in os.listdir(source_folder):
            source_path = os.path.join(source_folder, filename)
            flag = filename.split('_')[0] + '_dose'

            # 检查是否为图片文件
            if filename.endswith('catalyst.png'):
                for target_filename in target_folder_filename:
                    if target_filename.endswith('.tiff') and flag in target_filename:
                        output_filename = target_filename.split('.')[0] + '_catalyst.png'
                        target_path = os.path.join(target_folder, output_filename)
                        shutil.copy2(source_path, target_path)
                        print(f"复制文件：{filename} 到 {target_folder}")
        print("复制完成")
    except Exception as e:
        print(f"发生错误：{str(e)}")


# 指定源文件夹和目标文件夹
source_folder = "D:\\project\\deep_learning_recovery\\cv\\new images\\saifen"  # 替换为源文件夹的实际路径
target_folder = "D:\\project\\deep_learning_recovery\\Xiong_cifs and images\\cifs and images\\3"  # 替换为目标文件夹的实际路径

# 调用函数进行复制操作
copy_images(source_folder, target_folder)
