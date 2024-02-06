import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sewar.full_ref import vifp

# 用来存储结果的数据结构
results = {
    'image_name': [],
    'psnr_ori': [],
    'ssim_ori': [],
    'vif_ori': [],
    'psnr_sr': [],
    'ssim_sr': [],
    'vif_sr': [],
    'noise_type': []
}

# 用于保存结果的目录
output_directory = 'F:\\dose_result'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def calculate_vif(original, distorted):
    vif = vifp(original, distorted)
    return vif


def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)


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


# 更新此函数以符合您的文件命名规范和文件夹结构
def get_image_paths(folder_path, image_name):
    # 此处按照文件名命名规则调整路径
    ori_name = image_name
    label_image_path = os.path.join(folder_path, 'label', image_name)
    noisy_image_path = os.path.join(folder_path, 'original', ori_name)
    sr_image_path = os.path.join(folder_path, 'denoise', image_name)
    return label_image_path, noisy_image_path, sr_image_path


def boxplot_comparison(values, metric_name):
    fig, ax = plt.subplots(figsize=(7, 5))

    # 设置标签名
    labels = ['Original', 'DIVAESR Reconstructed']

    # 绘制箱形图
    ax.boxplot(values, vert=True, patch_artist=True, labels=labels)

    # 设置标题和标签
    ax.set_title('{} Comparison Between Original and DIVAESR Reconstructed'.format(metric_name))
    ax.set_ylabel(metric_name)

    plt.savefig(os.path.join(output_directory, '{}_comparison.png'.format(metric_name.lower())), dpi=300)  # 保存为高分辨率图像
    plt.show()


def visualize_and_save_results(results):
    # 创建图表
    plt.figure(figsize=(15, 8))

    # PSNR去噪分布图
    plt.subplot(2, 2, 1)
    plt.bar(range(len(results['image_name'])), results['psnr_denoised'], color='royalblue')
    plt.xlabel('Image Index')
    plt.ylabel('PSNR (Original)')
    plt.title('PSNR for Original Images')

    # SSIM去噪分布图
    plt.subplot(2, 2, 2)
    plt.bar(range(len(results['image_name'])), results['ssim_denoised'], color='royalblue')
    plt.xlabel('Image Index')
    plt.ylabel('SSIM (Original)')
    plt.title('SSIM for Original Images')

    # PSNR超分辨率分布图
    plt.subplot(2, 2, 3)
    plt.bar(range(len(results['image_name'])), results['psnr_sr'], color='seagreen')
    plt.xlabel('Image Index')
    plt.ylabel('PSNR (DIVAESR)')
    plt.title('PSNR for DIVAESR-Reconstructed Images')

    # SSIM超分辨率分布图
    plt.subplot(2, 2, 4)
    plt.bar(range(len(results['image_name'])), results['ssim_sr'], color='seagreen')
    plt.xlabel('Image Index')
    plt.ylabel('SSIM (DIVAESR)')
    plt.title('SSIM for DIVAESR-Reconstructed Images')

    # 自动调整subplot间距
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'metrics_distribution_high_res.png'), dpi=300)  # 保存为高分辨率图像
    plt.show()

    # 保存结果到文件
    with open(os.path.join(output_directory, 'analysis_results.txt'), 'w') as file:
        for i, image_name in enumerate(results['image_name']):
            file.write(
                f"{image_name}, {results['psnr_denoised'][i]}, {results['ssim_denoised'][i]}, {results['psnr_sr'][i]}, {results['ssim_sr'][i]}, {results['noise_type'][i]}\n")


def analyze_folder(folder_path):
    for image_name in os.listdir(os.path.join(folder_path, 'label')):
        label_image_path, noisy_image_path, sr_image_path = get_image_paths(folder_path, image_name)
        # 读取图像
        label = cv2.imread(label_image_path, cv2.IMREAD_GRAYSCALE)

        img_with_noise = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
        sr_reconstructed = cv2.imread(sr_image_path, cv2.IMREAD_GRAYSCALE)

        # 检查图像是否存在
        if img_with_noise is None or sr_reconstructed is None:
            print(f"Skipping image {image_name} due to missing files.")
            continue

        # 计算PSNR和SSIM, MSE
        psnr_ori = compare_psnr(label, img_with_noise)
        ssim_ori = compare_ssim(label, img_with_noise)
        vif_ori = calculate_vif(label, img_with_noise)

        # label_resized_for_ssim = cv2.resize(label, (128, 128), interpolation=cv2.INTER_LINEAR)
        psnr_sr = compare_psnr(label, sr_reconstructed)
        ssim_sr = compare_ssim(label, sr_reconstructed)
        vif_sr = calculate_vif(label, sr_reconstructed)
        vif_sr = calculate_vif(label, label)

        # 估计噪声类型
        noise_type = estimate_noise_type(img_with_noise, label)

        # 将结果存入数据结构中
        results['image_name'].append(image_name)
        results['psnr_ori'].append(psnr_ori)
        results['ssim_ori'].append(ssim_ori)
        results['vif_ori'].append(vif_ori)
        results['psnr_sr'].append(psnr_sr)
        results['ssim_sr'].append(ssim_sr)
        results['vif_sr'].append(vif_sr)
        results['noise_type'].append(noise_type)

    # 计算平均值
    average_psnr_noised = np.mean(results['psnr_ori'])
    std_psnr_noised = np.std(results['psnr_ori'])
    average_ssim_noised = np.mean(results['ssim_ori'])
    std_ssim_noised = np.std(results['ssim_ori'])
    average_vif_noised = np.mean(results['vif_ori'])
    std_vif_noised = np.std(results['vif_ori'])
    average_psnr_sr = np.mean(results['psnr_sr'])
    std_psnr_sr = np.std(results['psnr_sr'])
    average_ssim_sr = np.mean(results['ssim_sr'])
    std_ssim_sr = np.std(results['ssim_sr'])
    average_vif_sr = np.mean(results['vif_sr'])
    std_vif_sr = np.std(results['vif_sr'])

    print(f"Average PSNR for noised Images: {average_psnr_noised}")
    print(f"Average SSIM for noised Images: {average_ssim_noised}")
    print(f"Average PSNR for Super-Resolution Images: {average_psnr_sr}")
    print(f"Average SSIM for Super-Resolution Images: {average_ssim_sr}")
    with open(os.path.join(output_directory, 'noise_analysis_results.txt'), 'a+') as file:
        file.write(f"1000000dose\n, "
                   f"'average_psnr_noised': {average_psnr_noised}, 'std_psnr_noised': {std_psnr_noised}\n, "
                   f"'average_ssim_noised': {average_ssim_noised}, 'std_ssim_noised': {std_ssim_noised}\n, "
                   f"'average_vif_noised': {average_vif_noised}, 'std_vif_noised': {std_vif_noised}\n, "
                   f"'average_psnr_sr': {average_psnr_sr}, 'std_psnr_sr': {std_psnr_sr}\n"
                   f"'average_ssim_sr':{average_ssim_sr}, 'std_ssim_noised':{std_ssim_sr}\n"
                   f"'average_vif_sr':{average_vif_sr}, 'std_vif_sr':{std_vif_sr}\n")


if __name__ == '__main__':
    # 指定文件夹路径
    folder_path = 'F:\\test_dose_final\\1000000dose'
    analyze_folder(folder_path)
