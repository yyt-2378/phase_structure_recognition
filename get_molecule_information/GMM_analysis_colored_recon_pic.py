import os
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
import numpy as np


def normalize_element_weights(selected_elements, element_pixel_values):
    """
    Normalize the weights of the selected elements based on their pixel values.

    Parameters:
    - selected_elements: A list of element keys indicating the selected elements.
    - element_pixel_values: A dictionary with element keys, and their pixel values and raw weights.

    Returns:
    - normalized_weights: A dictionary with element keys and their normalized weights, sorted by pixel value.
    """
    # Extract the weights and pixel values for the selected elements
    weights = np.array([element_pixel_values[element][1] for element in selected_elements], dtype=np.float)
    pixel_values = np.array([element_pixel_values[element][0] for element in selected_elements], dtype=np.float)

    # Normalize the weights
    normalized_weights_raw = weights / np.sum(weights)

    # Sort the elements by pixel value, and sort the normalized weights accordingly
    sorted_indices = np.argsort(pixel_values)
    sorted_elements = np.array(selected_elements)[sorted_indices]
    sorted_normalized_weights = normalized_weights_raw[sorted_indices]

    # Create a dictionary of normalized weights
    normalized_weights = {element: weight for element, weight in zip(sorted_elements, sorted_normalized_weights)}
    normalized_weights_arr = np.array(list(normalized_weights.values()))

    return normalized_weights_arr


def determine_best_cluster_number(data, original_data, element_pixel_values, min_clusters=3, max_clusters=5):
    best_error = np.infty
    best_cluster = min_clusters
    best_elements = None

    for n_clusters in range(min_clusters, max_clusters + 1):
        # 根据权重提取前n_cluster类
        top_elements = sorted(element_pixel_values.items(), key=lambda item: item[1][1], reverse=False)[:n_clusters]
        # 将结果转换回字典格式
        top_elements_dict = {element: values for element, values in top_elements}
        top_elements_list = list(top_elements_dict.keys())
        top_elements_pixel_values = np.array([top_elements_dict[element][0] for element in top_elements_list])  # 提取像素值
        gmm = GaussianMixture(n_components=n_clusters)
        gmm.fit(data)
        centers = gmm.means_.flatten()

        # 计算每个元素的像素值与每个聚类中心的差的绝对值
        errors = np.abs(centers[:, np.newaxis] - top_elements_pixel_values)

        # 加权误差
        weights = np.array([1 / n_clusters for i in range(len(top_elements_dict.keys()))])  # 提取权重
        weighted_errors = errors * weights

        # 使用匈牙利算法找到最优匹配
        row_ind, col_ind = linear_sum_assignment(weighted_errors)
        total_error = weighted_errors[row_ind, col_ind].sum()  # 计算匹配的总误差

        # 如果当前聚类数的总误差小于之前的最小误差，则更新最佳聚类数、误差和元素列表
        if total_error < best_error:
            best_error = total_error
            best_cluster = n_clusters
            best_elements = [top_elements_list[index] for index in col_ind]

    return best_cluster, best_elements, top_elements_dict


def position_encoding(data):
    """
    Encode the position of pixels using sine and cosine functions.

    Parameters:
    - data: 2D numpy array representing an image.

    Returns:
    - encoded_data: Position encoded data.
    """
    x_coords, y_coords = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
    encoded_x = np.sin(x_coords / data.shape[1] * np.pi)
    encoded_y = np.cos(y_coords / data.shape[0] * np.pi)

    return encoded_x, encoded_y


def map_color_to_clusters(cluster_centers, best_elements, best_element_dict):
    """
    Map the cluster centers to their corresponding elements and the associated colors.

    Parameters:
    - cluster_centers: The centers of the clusters determined by the GMM.
    - best_elements: The list of elements corresponding to the best clustering.
    - element_pixel_values: Dictionary of element pixel values and their associated colors.

    Returns:
    - cluster_colors: The list of colors corresponding to the cluster centers.
    """
    # Extract the colors for the best elements
    color_mapping = {'bg': 'black', 'Si': 'navy', 'O': 'skyblue', 'C': 'gray', 'N': 'yellow', 'S': 'red'}
    element_colors = {element: color_mapping[element] for element in best_elements}

    # Map each cluster center to the color of the closest element by pixel value
    cluster_colors = []
    for center in cluster_centers:
        # Find the element with the closest pixel value to the cluster center
        closest_element = min(best_elements, key=lambda element: abs(best_element_dict[element][0] - center))
        # Get the color for this element
        cluster_colors.append(element_colors[closest_element])

    return cluster_colors


def gmm_clustering(img_path, output_dir, background_threshold=100, min_area=3):
    # Load the image and convert to grayscale
    img = Image.open(img_path).convert('L')
    img_basename = os.path.basename(img_path)
    img_output = os.path.join(output_dir, img_basename)
    data = np.array(img)

    # Set background threshold
    data = np.where(data < background_threshold, 0, data)

    # Reshape data for clustering
    data_reshaped = data.reshape(-1, 1)
    encoded_x, encoded_y = position_encoding(data)

    element_pixel_values = {'bg': (0, 0.3), 'Si': (235, 0.4), 'Al': (225, 3), 'O': (177, 0.4), 'H': (109, 10),
                            'C': (158, 0.8), 'S': (255, 2), 'N': (167, 2)}

    # Determine the best number of clusters
    best_n_clusters, best_elements, best_elements_dict = determine_best_cluster_number(data_reshaped, data, element_pixel_values)
    print(best_n_clusters)
    # Combining the encoded positions with the original data
    # data_reshaped = data.flatten()
    # encoded_x_reshaped = encoded_x.flatten()
    # encoded_y_reshaped = encoded_y.flatten()
    # features = np.column_stack((data_reshaped, encoded_x_reshaped, encoded_y_reshaped))
    # Apply Gaussian Mixture Model with the best number of clusters
    gmm = GaussianMixture(n_components=5)
    gmm.fit(data_reshaped)
    labels = gmm.predict(data_reshaped)
    cluster_centers = gmm.means_.flatten()

    # Map colors to clusters based on cluster centers
    # colors = map_color_to_clusters(cluster_centers, best_elements, best_elements_dict)
    # cmap = ListedColormap(colors)

    # Apply the clustering result back to the image shape
    clustered_image = labels.reshape(data.shape)

    # Perform connected component analysis
    label_image = label(clustered_image)

    # Get properties of the regions
    regions = regionprops(label_image)

    # # Revise small areas
    # for region in regions:
    #     if region.area < min_area:
    #         # Create a slightly larger bounding box
    #         minr, minc, maxr, maxc = region.bbox
    #         minr = max(minr - 1, 0)
    #         minc = max(minc - 1, 0)
    #         maxr = min(maxr + 1, label_image.shape[0])
    #         maxc = min(maxc + 1, label_image.shape[1])
    #
    #         # Get the coordinates of all points in the bounding box
    #         local_coords = np.array([[i, j] for i in range(minr, maxr)
    #                                  for j in range(minc, maxc)])
    #
    #         # Extract the coordinates of the region
    #         coords = region.coords
    #         neighbor_labels = clustered_image[coords[:, 0], coords[:, 1]]
    #
    #         # Exclude the points that belong to the current region
    #         local_cluster_labels = clustered_image[local_coords[:, 0], local_coords[:, 1]]
    #         local_cluster_labels = local_cluster_labels[local_cluster_labels != neighbor_labels[0]]
    #
    #         # If there are neighboring labels, calculate the most common label
    #         if local_cluster_labels.size > 0:
    #             most_common_label = np.bincount(local_cluster_labels).argmax()
    #
    #             # Re-label the current region
    #             clustered_image[region.coords[:, 0], region.coords[:, 1]] = most_common_label

    # 使用排序后的颜色映射显示图像
    plt.imshow(clustered_image)
    plt.axis('off')  # 关闭坐标轴显示

    # 保存图像（不包含颜色条）
    plt.show()
    # plt.savefig(img_output, bbox_inches='tight', pad_inches=0)
    # plt.close()


if __name__ == '__main__':
    # 加载图像
    image_path = 'F:\\large_scale'
    output_dir = 'F:\\test_sr_colored'
    os.makedirs(output_dir, exist_ok=True)
    for im in os.listdir(image_path):
        im_path = os.path.join(image_path, im)
        gmm_clustering(im_path, output_dir)
        print(f'---------------------colored reconstruction picture {im}---------------------------')


