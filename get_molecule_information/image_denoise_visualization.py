import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import PIL


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


def visualize_SSIM_difference(original, denoised):
    # 计算差异和SSIM
    diff = original - denoised
    s, _ = ssim(original, denoised, full=True)

    plt.figure(figsize=(15, 10))

    # 显示原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # 显示去噪后的图像
    plt.subplot(2, 3, 2)
    plt.imshow(denoised, cmap='gray')
    plt.title("Denoised")
    plt.axis('off')

    # 显示差异
    plt.subplot(2, 3, 3)
    plt.imshow(np.abs(diff)*10, cmap='hot')
    plt.title("Difference (Amplified)")
    plt.axis('off')

    # 显示原始图像的直方图
    plt.subplot(2, 3, 4)
    plt.hist(original.ravel(), bins=255, color='blue', alpha=0.7, label="Original")
    plt.legend(loc='upper right')
    plt.title("Histogram")

    # 显示去噪后的图像的直方图
    plt.subplot(2, 3, 5)
    plt.hist(denoised.ravel(), bins=255, color='green', alpha=0.7, label="Denoised")
    plt.legend(loc='upper right')
    plt.title("Histogram")

    plt.tight_layout()
    plt.show()

    # 打印噪声类型和程度
    print(f"Noise Type: {estimate_noise_type(original, denoised)}")
    print(f"Noise Level: {np.std(diff):.4f}")
    print(f"SSIM: {s:.4f}")


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


def visualize_PSNR_difference(original, denoised):
    diff = original - denoised

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(denoised, cmap='gray')
    plt.title("Denoised")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(np.abs(diff)*10, cmap='hot')
    plt.title("Difference (Amplified)")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.hist(original.ravel(), bins=255, color='blue', alpha=0.7, label="Original")
    plt.legend(loc='upper right')
    plt.title("Histogram")

    plt.subplot(2, 3, 5)
    plt.hist(denoised.ravel(), bins=255, color='green', alpha=0.7, label="Denoised")
    plt.legend(loc='upper right')
    plt.title("Histogram")

    plt.tight_layout()
    plt.show()

    psnr_value = compute_psnr(original*255, denoised*255)
    print(f"PSNR: {psnr_value:.2f} dB")


def visualize_difference(images, titles):
    plt.figure(figsize=(20, 10))

    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def compare_with_label(original, denoised, label_img):
    original_score, _ = ssim(original, label_img, full=True)
    denoised_score, _ = ssim(denoised, label_img, full=True)

    print(f"SSIM between Original and Label: {original_score:.4f}")
    print(f"SSIM between Denoised and Label: {denoised_score:.4f}")

    if original_score > denoised_score:
        print("Original image is closer to the label image.")
    else:
        print("Denoised image is closer to the label image.")

    visualize_difference([original, denoised, label_img], ["Original", "Denoised", "Label"])


# 读取图像
original = cv2.imread('F:\\STEM_DCVAESR_Data\\test\\label\\molecule_0_r90_darker.png', cv2.IMREAD_GRAYSCALE)
denoised = cv2.imread('F:\\rcnn_test\\molecule_0_r90_darker.png', cv2.IMREAD_GRAYSCALE)

visualize_SSIM_difference(original, denoised)


# 读取图像
original = cv2.imread('F:\\STEM_DCVAESR_Data\\test\\label\\molecule_0_r90_darker.png', cv2.IMREAD_GRAYSCALE) / 255.0
denoised = cv2.imread('F:\\rcnn_test\\molecule_0_r90_darker.png', cv2.IMREAD_GRAYSCALE) / 255.0

visualize_PSNR_difference(original, denoised)

# estimated noise
img_original = cv2.imread('F:\\STEM_DCVAESR_Data\\test\\label\\molecule_0_r90_darker.png', cv2.IMREAD_GRAYSCALE)
img_denoised = cv2.imread('F:\\rcnn_test\\molecule_0_r90_darker.png', cv2.IMREAD_GRAYSCALE)

noise_type = estimate_noise_type(img_original, img_denoised)
print(f"The estimated noise type is: {noise_type}")

# 使用图像
original_img = cv2.imread('F:\\rcnn_test\\molecule_0_r90_darker.png', cv2.IMREAD_GRAYSCALE)
denoised_img = cv2.imread('F:\\test_model\\version_0_0.05SRloss\\SR_reconstructed\\SR_reconstructed_molecule_0_r90_darker.png', cv2.IMREAD_GRAYSCALE)
label_img = cv2.imread('F:\\STEM_DCVAESR_Data\\test\\label\\molecule_0_r90_darker.png', cv2.IMREAD_GRAYSCALE)

compare_with_label(original_img, denoised_img, label_img)
