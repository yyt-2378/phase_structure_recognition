import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim
import pandas as pd


def estimate_noise_type(img_original, img_denoised):
    # Calculate the noise image
    noise_img = img_original - img_denoised

    # Calculate the mean and standard deviation
    mean = np.mean(noise_img)
    std = np.std(noise_img)

    # Calculate the mean and standard deviation for the original image
    mean_original = np.mean(img_original)
    std_original = np.std(img_original)

    # Estimate noise type based on the characteristics
    if abs(mean) < 0.05 * mean_original and abs(std - std_original) > 0.05 * std_original:
        return "Gaussian"
    elif abs(mean) > 0.05 * mean_original:
        return "Poisson"
    else:
        return "Unknown"


def save_and_visualize_SSIM_difference(original, denoised, output_dir):
    # 计算差异和SSIM
    diff = original - denoised
    s, diff_map = ssim(original, denoised, full=True)

    # 创建一个文件夹来保存图像
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # # 保存并显示去噪后的图像
    # plt.figure(figsize=(5, 5))
    # plt.imshow(denoised, cmap='gray')
    # plt.title("DIVAE Reconstructed Image")
    # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, 'denoised_image.png'), transparent=True)
    # plt.close()

    # 保存并显示差异图像
    plt.figure(figsize=(5, 5))
    plt.imshow(np.abs(diff)*10, cmap='magma')
    plt.title("magma Amplified Difference")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'magma_difference_image.png'), transparent=True)
    plt.close()

    # plt.figure(figsize=(10, 5))
    # # 绘制原始图像的直方图
    # plt.hist(original.ravel()*255, bins=255, color='blue', alpha=0.7, label="Original")
    # # 绘制去噪后图像的直方图
    # plt.hist(denoised.ravel()*255, bins=255, color='green', alpha=0.7, label="DIVAE Reconstructed")
    # plt.legend(loc='upper right')
    # plt.title("Histogram Comparison")
    # plt.xlabel("Pixel Intensity")
    # plt.ylabel("Frequency")
    # # 保存合并后的直方图到文件
    # save_path = os.path.join(output_dir, 'combined_histogram.png')
    # plt.savefig(save_path, transparent=True)
    # plt.close()  # 关闭图像以节省内存

    # 打印噪声类型和程度
    noise_type = estimate_noise_type(original, denoised)
    print(f"Noise Type: {noise_type}")
    print(f"Noise Level: {np.std(diff):.4f}")


def mse(imageA, imageB):
    """计算均方误差"""
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def compute_psnr(original, denoised):
    """计算PSNR值"""
    max_pixel = 255.0
    mean_square_error = mse(original, denoised)
    if mean_square_error == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mean_square_error))


def visualize_difference(images, titles, scores):
    plt.figure(figsize=(20, 10))
    for idx, (img, title, score) in enumerate(zip(images, titles, scores)):
        plt.subplot(1, len(images), idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"{title}\nSSIM: {score:.4f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def compare_with_label(original, denoised, label_img):
    original_score, _ = ssim(original, label_img, full=True)
    denoised_score, _ = ssim(denoised, label_img, full=True)

    print(f"SSIM between Original and Label: {original_score:.4f}")
    print(f"SSIM between Denoised and Label: {denoised_score:.4f}")

    closer_image = "Original" if original_score > denoised_score else "Denoised"
    print(f"{closer_image} image is closer to the label image.")

    # Pass the SSIM scores along with images and titles to the visualization function
    visualize_difference([original, denoised, label_img],
                         ["Original", "DIVAESR Reconstructed", "Label"],
                         [original_score, denoised_score, 1])


def read_image(file_path, scale=False):
    """读取图像并根据需要进行缩放"""
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if scale:
        image = image / 255.0
    return image


def compute_metrics(original, denoised, label):
    """计算和打印所有评价指标"""
    original_ssim, _ = ssim(original, label, full=True)
    denoised_ssim, _ = ssim(denoised, label, full=True)
    psnr_value = compute_psnr(original*255, denoised*255)
    noise_type = estimate_noise_type(original, denoised)

    print(f"SSIM Original vs. Label: {original_ssim:.4f}")
    print(f"SSIM Denoised vs. Label: {denoised_ssim:.4f}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"Estimated Noise Type: {noise_type}")


def visualize_and_save_images(original, denoised, label, original_ssim, denoised_ssim, output_dir):
    """可视化图像和它们的SSIM分数，并将每张子图保存到指定的目录"""
    titles = ['Original', 'DIVAESR Reconstructed', 'Label']
    ssim_scores = [original_ssim, denoised_ssim, None]
    images = [original, denoised, label]

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (img, title, score) in enumerate(zip(images, titles, ssim_scores)):
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        if score is not None:
            plt.title(f"{title}\nSSIM: {score:.4f}")
        else:
            plt.title(title)
        plt.axis('off')

        # 保存子图
        save_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
        plt.savefig(save_path, transparent=True)
        plt.close()  # 关闭图像以节省内存


def histogram_visual_as_boxplot(original_images, denoised_images, output_dir):
    plt.figure(figsize=(15, 10))

    # 准备数据
    data_original = [img.ravel() * 255 for img in original_images]
    data_denoised = [img.ravel() * 255 for img in denoised_images]
    data = [val for pair in zip(data_original, data_denoised) for val in pair]

    # 准备标签
    labels = ['Ori 1', 'Denoise 1', 'Ori 2', 'Denoise 2', 'Ori 3', 'Denoise 3',
              'Ori 4', 'Denoise 4', 'Ori 5', 'Denoise 5', 'Ori 6', 'Denoise 6']

    # 定义颜色，使用两种颜色来区分原始和去噪图像
    color_original = 'skyblue'
    color_denoised = 'lightgreen'
    box_colors = [color_original if i % 2 == 0 else color_denoised for i in range(len(data))]

    # 绘制盒须图，并不显示异常值
    bp = plt.boxplot(data, labels=labels, notch=True, patch_artist=True)

    # 设置颜色
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)

    # 移除标题
    # plt.title("Box Plot Comparison of Six Groups")  # Removed as per instruction

    plt.xlabel("Group")
    plt.ylabel("Pixel Intensity")
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'box_plot_histograms_summary.png')
    plt.savefig(save_path, transparent=True)
    plt.close()


def histogram_visual_as_lineplot(original_images, denoised_images, output_dir):
    plt.figure(figsize=(15, 10))

    # 定义蓝色和绿色系列颜色用于原始图像和去噪图像
    colors_original = ['blue', 'deepskyblue', 'dodgerblue', 'steelblue', 'lightblue', 'skyblue']
    colors_denoised = ['green', 'limegreen', 'forestgreen', 'darkgreen', 'seagreen', 'mediumseagreen']

    # 用于图例的标签
    labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5', 'Group 6']

    for i in range(6):
        # 计算直方图的值
        hist_original, bins = np.histogram(original_images[i].ravel() * 255, bins=255, range=(0, 255))
        hist_denoised, _ = np.histogram(denoised_images[i].ravel() * 255, bins=255, range=(0, 255))

        # 中心位置
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # 绘制线图，使用不同的颜色和线型
        plt.plot(bin_centers, hist_original, color=colors_original[i], label=f'Original {labels[i]}')
        plt.plot(bin_centers, hist_denoised, color=colors_denoised[i], linestyle='--', label=f'Reconstructed {labels[i]}')

    # 显示图例
    plt.legend()

    # 移除标题
    # plt.title("Line Plot Comparison of Six Groups")  # Title removed as per instruction

    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'line_plot_histograms_summary.png')
    plt.savefig(save_path, transparent=True)
    plt.close()


def histogram_visual(original_images, denoised_images, label, output_dir):
    plt.figure(figsize=(15, 10))

    # 定义蓝色和绿色用于原始图像和去噪图像
    color_original = 'darkgreen'
    color_denoised = 'darkblue'
    color_label = 'blue'

    # 初始化数据存储列表
    original_data = []
    denoised_data = []
    label_data = []

    # 聚合所有原始图像和去噪图像的数据
    for i in range(len(original_images)):
        original_data.extend(original_images[i].ravel() * 255)
        denoised_data.extend(denoised_images[i].ravel() * 255)
        label_data.extend(label[i].ravel() * 255)

    # 计算原始图像的直方图
    hist_original, bins_original = np.histogram(original_data, bins=255, range=(0, 255))
    # 计算去噪图像的直方图
    hist_denoised, bins_denoised = np.histogram(denoised_data, bins=255, range=(0, 255))
    # 计算去噪图像的直方图
    hist_label, bins_label = np.histogram(label_data, bins=255, range=(0, 255))

    # 设置条形的中心位置
    bin_centers_original = (bins_original[:-1] + bins_original[1:]) / 2
    bin_centers_denoised = (bins_denoised[:-1] + bins_denoised[1:]) / 2
    bin_centers_label = (bins_label[:-1] + bins_label[1:]) / 2

    # 绘制原始图像的直方图
    plt.bar(bin_centers_original, hist_original, width=1, color=color_original, alpha=0.6, label='Original')
    # 绘制去噪图像的直方图
    plt.bar(bin_centers_denoised, hist_denoised, width=1, color=color_denoised, alpha=0.6, label='Reconstructed')
    # 绘制去噪图像的直方图
    plt.bar(bin_centers_label, hist_label, width=1, color=color_label, alpha=0.6, label='Label')

    # 显示图例
    plt.legend(fontsize=22)  # 或者指定具体数值，如 fontsize=12

    plt.xlabel("Pixel Intensity", fontsize=20)  # 设置x轴标签的字号
    plt.ylabel("Frequency", fontsize=20)  # 设置y轴标签的字号
    plt.tight_layout()

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, 'histograms_summary.png')
    plt.savefig(save_path, transparent=True)
    plt.close()


def calculate_histogram_data(image):
    """Calculate histogram data for a single image."""
    hist, bins = np.histogram(image.ravel() * 255, bins=255, range=(0, 255))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return hist, bin_centers


def histogram_visual_to_csv(original_images, denoised_images, label_images, output_dir):
    """Calculate histogram data and save to CSV."""
    histograms_data = {
        'bin_centers': [],
        'original_hist': [],
        'denoised_hist': [],
        'label_hist': []
    }

    # Calculate histogram data for each image group
    for original, denoised, label in zip(original_images, denoised_images, label_images):
        hist_original, bin_centers = calculate_histogram_data(original)
        hist_denoised, _ = calculate_histogram_data(denoised)
        hist_label, _ = calculate_histogram_data(label)

        histograms_data['bin_centers'].extend(bin_centers)
        histograms_data['original_hist'].extend(hist_original)
        histograms_data['denoised_hist'].extend(hist_denoised)
        histograms_data['label_hist'].extend(hist_label)

    # Convert to DataFrame
    df_histograms = pd.DataFrame(histograms_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'histograms_data.csv')
    df_histograms.to_csv(csv_path, index=False)
    print(f"Histogram data saved to {csv_path}")

    return df_histograms


# # 文件路径
output_directory = 'D:\\project\\phase_structure\\phase_structure_recognition\\get_molecule_information\\histo_result'  # 定义输出目录
# original_file_path = 'F:\\test_128_different_dose\\500dose\\molecule_1.jpg\\original_crop_9.png'
# denoised_file_path = 'F:\\test_128_different_dose\\500dose\\molecule_1.jpg\\SR_reconstructed_img_9.png'
# label_file_path = 'F:\\test_128_different_dose\\500dose\\molecule_1.jpg\\crop_9.png'
#
# # 读取图像
# original = read_image(original_file_path, scale=True)
# denoised = read_image(denoised_file_path, scale=True)
# label = read_image(label_file_path, scale=True)

# # 计算指标
# compute_metrics(original, denoised, label)

# 可视化图像
# original_ssim, _ = ssim(original, label, full=True)
# denoised_ssim, _ = ssim(denoised, label, full=True)
# visualize_and_save_images(original, denoised, label, original_ssim, denoised_ssim, output_directory)
# save_and_visualize_SSIM_difference(original, denoised, output_directory)

# 假设您有六组原始和去噪后的图像数组
molecule_filename = [9]
list_path = os.listdir('F:\\test_128_different_dose')
os.makedirs(output_directory, exist_ok=True)
original_images = []
denoised_images = []
label_images = []
for dif_dose_i in range(len(list_path)):
    dif_dose = list_path[dif_dose_i]
    dose_path = os.path.join('F:\\test_128_different_dose', dif_dose)
    for j in os.listdir(dose_path):
        orginal_img = os.path.join(dose_path, j, f"original_crop_{molecule_filename[dif_dose_i]}.png")
        denoise_img = os.path.join(dose_path, j, f"SR_reconstructed_img_{molecule_filename[dif_dose_i]}.png")
        label_img = os.path.join(dose_path, j, f"crop_{molecule_filename[dif_dose_i]}.png")
        original = read_image(orginal_img, scale=True)
        denoised = read_image(denoise_img, scale=True)
        label = read_image(label_img, scale=True)
        original_images.append(original)
        denoised_images.append(denoised)
        label_images.append(label)

histogram_visual(original_images, denoised_images, label_images, output_directory)
# Generate and save histogram data to CSV
df_histograms = histogram_visual_to_csv(original_images, denoised_images, label_images, output_directory)