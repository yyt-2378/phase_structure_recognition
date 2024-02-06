# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from PIL import Image
#
#
# def compute_pixel_density_histogram(image_files, dataset_path):
#     """计算数据集中所有图片的像素值密度直方图"""
#     histograms = []
#     for file in image_files:
#         img = Image.open(os.path.join(dataset_path, file)).convert('L')
#         hist, _ = np.histogram(np.array(img).flatten(), bins=256, range=(0, 256))
#         histograms.append(hist / hist.sum())  # 计算密度
#     avg_histogram = np.mean(histograms, axis=0)
#     return avg_histogram
#
#
# def compare_image_datasets(small_dataset_path, large_dataset_path, output_csv_path):
#     """比较两个灰度图片数据集的亮度分布差异，并保存结果到 CSV 文件"""
#     # 获取文件列表
#     small_image_files = [f for f in os.listdir(small_dataset_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
#     large_image_files = [f for f in os.listdir(large_dataset_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
#
#     # 计算平均像素值密度直方图
#     small_histogram = compute_pixel_density_histogram(small_image_files, small_dataset_path)
#     large_histogram = compute_pixel_density_histogram(large_image_files, large_dataset_path)
#
#     # 可视化亮度分布差异
#     plt.figure(figsize=(10, 5))
#     sns.lineplot(data=small_histogram, label='Real STEM Dataset', color='blue')
#     sns.lineplot(data=large_histogram, label='Simulation STEM Dataset', color='orange')
#     plt.title('Pixel Value Density Distribution Comparison')
#     plt.xlabel('Pixel Intensity')
#     plt.ylabel('Density')
#     plt.legend()
#     plt.show()
#
#     # 将结果保存到 CSV 文件
#     df = pd.DataFrame({
#         'Pixel Intensity': range(256),
#         'Density - Real STEM': small_histogram,
#         'Density - Simulation STEM': large_histogram
#     })
#     df.to_csv(output_csv_path, index=False)
#
#
# # 定义数据集路径和输出 CSV 文件路径
# small_dataset_path = 'D:\\project\\phase_structure\\real_stem_data'
# large_dataset_path = 'F:\\MOLECULE dataset2\\training'
# output_csv_path = 'D:\\project\\phase_structure\\phase_structure_recognition\\histogram_comparison.csv'
#
# # 比较数据集并保存结果
# compare_image_datasets(small_dataset_path, large_dataset_path, output_csv_path)

import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_noise_features(image_files, dataset_path):
    """计算数据集中所有图片的噪声特征"""
    noise_features = {'mean': [], 'std': []}
    for file in image_files:
        img = Image.open(os.path.join(dataset_path, file)).convert('L')
        img_array = np.array(img).astype(np.float32)
        noise = img_array - img_array.mean()
        noise_features['mean'].append(noise.mean())
        noise_features['std'].append(noise.std())
    return pd.DataFrame(noise_features)


def analyze_datasets(small_dataset_path, large_dataset_path):
    """分析两个数据集的噪声特征"""
    small_image_files = [f for f in os.listdir(small_dataset_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    large_image_files = [f for f in os.listdir(large_dataset_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    small_noise_features = calculate_noise_features(small_image_files, small_dataset_path)
    large_noise_features = calculate_noise_features(large_image_files, large_dataset_path)

    return small_noise_features, large_noise_features


# 定义数据集路径
small_dataset_path = 'D:\\project\\phase_structure\\real_stem_data'
large_dataset_path = 'F:\\MOLECULE dataset2\\training'

# 分析数据集
small_noise_features, large_noise_features = analyze_datasets(small_dataset_path, large_dataset_path)

# 保存结果到CSV
small_noise_features.to_csv('real_stem_noise_features.csv', index=False)
large_noise_features.to_csv('simulation_stem_noise_features.csv', index=False)

# 分析结果
print("Real STEM Dataset Noise Features:\n", small_noise_features.describe())
print("\nSimulation STEM Dataset Noise Features:\n", large_noise_features.describe())


def plot_comparison(real_features, sim_features, feature_name):
    """绘制两个数据集特征的比较图"""
    plt.figure(figsize=(10, 6))

    # 绘制箱线图进行比较
    sns.boxplot(data=[real_features[feature_name], sim_features[feature_name]])
    plt.xticks([0, 1], ['Real STEM Dataset', 'Simulation STEM Dataset'])
    plt.ylabel(feature_name)
    plt.title(f'Comparison of {feature_name} Between Real and Simulated Datasets')
    # plt.show()
    plt.savefig(f'Comparison of {feature_name} Between Real and Simulated Datasets.jpg')


# 读取之前保存的CSV文件
real_stem_noise = pd.read_csv('D:\\project\\phase_structure\\phase_structure_recognition\\get_molecule_information\\real_stem_noise_features.csv')
sim_stem_noise = pd.read_csv('D:\\project\\phase_structure\\phase_structure_recognition\\get_molecule_information\\simulation_stem_noise_features.csv')

# 绘制平均值比较图
plot_comparison(real_stem_noise, sim_stem_noise, 'mean')

# 绘制标准差比较图
plot_comparison(real_stem_noise, sim_stem_noise, 'std')
