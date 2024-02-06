from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from skimage import filters
from skimage.metrics import structural_similarity as ssim
import os


# Function to load images and convert to binary
def load_and_convert_images(image_paths):
    images = []
    for path in image_paths:
        with Image.open(path) as img:
            images.append(img.convert('L'))  # Convert to grayscale
    return images


# Function to find the largest image size among all images
def find_largest_image_size(images):
    max_width = max(img.size[0] for img in images)
    max_height = max(img.size[1] for img in images)
    return (max_width, max_height)


# Function to resize images to match the largest image size
def resize_images_to_largest(images, largest_size):
    resized_imgs = [img.resize(largest_size, Image.ANTIALIAS) for img in images]
    return resized_imgs


# Function to convert images to binary using Otsu's thresholding
def convert_to_binary(images, last_image_thresh_factor=0.72):
    binary_images = []
    for i, img in enumerate(images):
        img_array = np.array(img)
        thresh = filters.threshold_otsu(img_array)

        # 对于最后一张图像，降低阈值
        if i == len(images) - 1:
            thresh *= last_image_thresh_factor

        binary = img_array > thresh
        binary_images.append(binary)
    return binary_images


# Function to extract skeletons
def extract_skeletons(binary_img):
    return [skeletonize(binary) for binary in binary_img]


# Function to calculate SSIM
def calculate_ssim(reference_image, other_images):
    reference_array = np.array(reference_image, dtype=np.float32)
    ssim_scores = []
    for img in other_images:
        img_array = np.array(img, dtype=np.float32)
        score = ssim(reference_array, img_array, data_range=img_array.max() - img_array.min())
        ssim_scores.append(score)
    return ssim_scores


# Function to plot all images as subplots with SSIM scores
def plot_all_images_as_subplots(images, scores, image_paths, highlight_index, output_dir):
    # 计算子图的布局
    n = len(images)
    cols = 3  # 你可以根据需要调整列数
    rows = n // cols + (n % cols > 0)

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axs = axs.ravel() if rows > 1 else [axs]

    for i, (image, score, path) in enumerate(zip(images, scores, image_paths)):
        ax = axs[i]
        ax.imshow(image, cmap='gray')
        title = f"{os.path.basename(path)} - SSIM: {score:.5f}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')

        # 如果是得分最高的图像，将标题和SSIM得分设置为红色
        if i == highlight_index:
            ax.set_title(title, fontsize=10, color='red')

    # 删除多余的子图
    for i in range(n, rows * cols):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_conformation_with_ssim.png'))
    plt.show()


if __name__ == '__main__':
    # Load the images
    image_paths = [os.path.join('C:\\Users\\yyt70\\Desktop\\sample\\saifen', img) for img in
                   os.listdir('C:\\Users\\yyt70\\Desktop\\sample\\saifen')]
    image_paths.append('C:\\Users\\yyt70\\Desktop\\sample_2\\saifen\\17\\LLL.tif')
    output = 'C:\\Users\\yyt70\\Desktop\\'

    # Load images
    images = load_and_convert_images(image_paths)

    # Find the largest image size
    largest_size = find_largest_image_size(images)

    # Resize all images to match the largest image size
    resized_images = resize_images_to_largest(images, largest_size)

    # Convert resized images to binary
    binary_images = convert_to_binary(resized_images)

    # Extract skeletons
    skeletons = binary_images

    # Calculate SSIM scores using the skeleton of the last image (PNG) as the reference
    reference_skeleton = skeletons[-1]
    plt.figure()
    plt.imshow(reference_skeleton, cmap='gray')
    plt.show()
    ssim_scores = calculate_ssim(reference_skeleton, skeletons[:-1])  # Exclude the reference skeleton

    # 计算得分最高的TIFF图像的索引
    highest_score_index = np.argmax(ssim_scores)

    # 绘制所有图像并突出显示得分最高的图像
    plot_all_images_as_subplots(resized_images[:-1], ssim_scores, image_paths[:-1], highest_score_index, output)
