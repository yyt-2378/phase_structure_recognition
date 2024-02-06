import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.ndimage import generic_filter
import itertools


def plot_contour_and_save(image_array, levels=None, file_name='contour.png'):
    """
    Plot a 2D contour map from a 2D image array and save the plot to a file.

    Parameters:
    - image_array: 2D array of the image.
    - levels: Number of levels to draw in the contour plot.
    - file_name: Name of the file to save the plot.
    """
    # Create the plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(image_array, levels=levels, cmap='hot')
    plt.colorbar(contour)
    plt.title('2D Contour Map')
    plt.axis('off')  # Hide the axis

    # Save the plot without any extra space around it
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Save the figure
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure without displaying


def get_scatter_size(image_shape, base_size=100, scale_factor=0.0001):
    """
    Calculate an appropriate scatter plot size (s) based on the size of the image.

    Parameters:
    - image_shape: The shape of the image (height, width).
    - base_size: Base size for scatter plot point.
    - scale_factor: Factor to scale the base_size based on the image size.

    Returns:
    - s: Calculated size for scatter plot points.
    """
    # Calculate the size of the image (number of pixels)
    image_size = image_shape[0] * image_shape[1]

    # Adjust the scatter plot size based on the image size
    s = base_size / (1 + scale_factor * image_size)
    s = s - 1

    return s


def filter_valid_clusters(points, labels, valid_points_mask, min_cluster_size):
    """
    过滤出有效的簇（聚类）。

    参数:
    points - 包含点的数组或列表。
    labels - 每个点对应的簇标签。
    valid_points_mask - 一个布尔数组，表示哪些点是有效的。
    min_cluster_size - 簇的最小大小。

    返回:
    valid_clusters - 满足条件的有效簇列表。
    """
    unique_labels = set(labels)
    valid_clusters = []

    for k in unique_labels:
        if k == -1:
            # 跳过噪声
            continue
        class_member_mask = (labels == k)
        # 过滤出属于簇的点
        valid_points_mask_k = valid_points_mask & class_member_mask
        xy = points[valid_points_mask_k]
        # 过滤出小型簇
        if len(xy) >= min_cluster_size:
            valid_clusters.append(xy)

        flattened_arr = np.array(list(itertools.chain(*valid_clusters)))

    return flattened_arr


def apply_filter_and_dbscan(image_array, low_thresh, high_thresh, eps, min_samples, min_cluster_size=None,
                            apply_filter=True, check_threshold=200, radius=5):
    """
    Apply a custom filter and DBSCAN clustering to the image array and filter out small clusters.

    Parameters:
    - image_array: 2D numpy array of the image.
    - low_thresh: Lower bound for pixel intensity to be considered for clustering.
    - high_thresh: Upper bound for pixel intensity.
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    - min_cluster_size: The minimum number of pixels in a cluster for it to be considered valid.
    - apply_filter: Whether to apply a custom filter based on check_neighbors.
    - check_threshold: The threshold used by check_neighbors to filter neighboring pixels.
    - radius: The radius used to define the neighborhood in check_neighbors.

    Returns:
    - valid_clusters: A list of valid clusters (numpy arrays of points).
    """
    # Create a mask for pixels within the desired grayscale range
    mask = (image_array >= low_thresh) & (image_array <= high_thresh)
    y_indices, x_indices = np.where(mask)
    points = np.column_stack((x_indices, y_indices))

    # Define a function to check if there are any pixels above a certain threshold within a radius
    def check_neighbors(window):
        center = (window.size - 1) // 2
        window = window.reshape((radius*2+1, radius*2+1))
        window[window <= check_threshold] = 0
        window[window > check_threshold] = 1
        return window.sum() - window[radius, radius]  # Exclude the center pixel

    # Apply the filter to each point if required
    if apply_filter:
        neighborhood_size = (radius*2) + 1
        high_value_filter = generic_filter(image_array, check_neighbors, size=neighborhood_size, mode='constant', cval=0)
        valid_points_mask = (high_value_filter[y_indices, x_indices] == 0)
    else:
        valid_points_mask = np.full_like(y_indices, True, dtype=bool)

    # Use DBSCAN to find clusters
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    # Group points by cluster label
    labels = db.labels_

    if min_cluster_size:
        filter_arr = filter_valid_clusters(points, labels, valid_points_mask, min_cluster_size)
    else:
        # Filter points that are part of a cluster
        valid_points_mask &= (labels != -1)
        filter_arr = points[valid_points_mask]

    return filter_arr


if __name__ == '__main__':
    # 图片路径
    img_folder = 'C:\\Users\\yyt70\\Desktop\\test_cluster'
    output_folder = 'C:\\Users\\yyt70\\Desktop\\paper_data_cluster'
    os.makedirs(output_folder, exist_ok=True)
    output_folder_sub1 = 'C:\\Users\\yyt70\\Desktop\\paper_data_cluster\\cluster'
    output_folder_sub2 = 'C:\\Users\\yyt70\\Desktop\\paper_data_cluster\\heatmap'
    os.makedirs(output_folder_sub1, exist_ok=True)
    os.makedirs(output_folder_sub2, exist_ok=True)
    for im_path in os.listdir(img_folder):
        im_filename = im_path
        image_path = os.path.join(img_folder, im_path)
        # 加载图片并转换为灰度
        image = Image.open(image_path).convert('L')
        # # 获取原始图像的尺寸
        # original_size = image.size
        # # 计算新的尺寸（原始尺寸的四倍）
        # new_size = (original_size[0] * 2, original_size[1] * 2)
        # # 调整图像尺寸
        # image = image.resize(new_size, Image.ANTIALIAS)
        image_array = np.array(image)  # 将图片转换为numpy数组
        img_shape = image_array.shape
        # Call the function with different parameters
        clustered1 = apply_filter_and_dbscan(image_array, low_thresh=28, high_thresh=165, eps=2, min_samples=4,
                                             apply_filter=True)
        clustered2 = apply_filter_and_dbscan(image_array, low_thresh=160, high_thresh=190, eps=2, min_samples=3,
                                             apply_filter=False)
        clustered3 = apply_filter_and_dbscan(image_array, low_thresh=185, high_thresh=255, eps=2, min_samples=2,
                                             apply_filter=False)

        # Plotting
        plt.figure(figsize=(8, 8), frameon=False)  # frame on = False to remove the frame
        plt.imshow(image_array, cmap='gray', aspect='auto')
        s = get_scatter_size(img_shape)  # Calculate scatter size
        plt.scatter(clustered1[:, 0], clustered1[:, 1], color='burlywood', s=s)
        plt.scatter(clustered2[:, 0], clustered2[:, 1], color='red', s=s)
        plt.scatter(clustered3[:, 0], clustered3[:, 1], color='blue', s=s)
        plt.axis('off')  # Turn off the axis

        # Save the plot without any extra space around it
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # Save the figure
        plt.savefig(os.path.join(output_folder_sub1, im_filename), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()  # Close the figure to prevent it from displaying

        # 调用函数以绘制图形
        plot_contour_and_save(image_array, levels=50, file_name=os.path.join(output_folder_sub2, im_filename))
        print(f'-------------------success cluster img {im_filename}---------------------------')





