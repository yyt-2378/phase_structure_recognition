import numpy as np
import pandas as pd
import cv2
import easygraphics.dialog as dlg
from matplotlib.colors import rgb2hex
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import os

os.environ['OMP_NUM_THREADS'] = '1'


def elbow_method(data, min_clusters, max_clusters):
    """
    使用 elbow method 确定最佳的聚类数目

    Parameters:
        - data: numpy array, shape=(n_samples, n_features)
            待聚类的数据
        - max_clusters: int, optional (default=320)
            最大的聚类数目

    Returns:
        - best_k: int
            最佳的聚类数目
        - elbow_plot: matplotlib.figure.Figure
            elbow plot
    """
    distortion = []
    k_list = []
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=4).fit(data)
        distortion.append(kmeans.inertia_)
        k_list.append(k)
    elbow_plot = plt.figure(figsize=(10, 5))
    plt.plot(range(min_clusters, max_clusters + 1), distortion, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.savefig('elbow_k.png')
    # 寻找拐点
    deltas = [distortion[i] - distortion[i - 1] for i in range(1, len(distortion))]
    best_k = k_list[deltas.index(max(deltas)) + 1 if deltas else 1]
    return best_k, elbow_plot


def find_nearest_neighbors(coord_arr, point):
    """
    计算一个点与周围最近的2个点的距离
    参数：
    coord_arr：一个包含所有点坐标的numpy数组
    point：一个表示查询点的坐标的numpy数组
    返回值：
    一个包含最近的2个点的数组，以及它们与查询点的距离
    """
    # 创建kd-tree对象
    tree = cKDTree(coord_arr)

    # 查找最近的3个点（包括查询点本身）
    dists, indices = tree.query(point, k=3)

    # 返回最近的三个点和它们与查询点的距离
    return indices[1:], dists[1:]


def kmeans_visualize(bond_length_arr: np.ndarray, n_clusters: int, filename: str):
    kmeans = KMeans(n_clusters=n_clusters, random_state=4).fit(bond_length_arr)
    print(kmeans.cluster_centers_)

    fig, ax = plt.subplots()

    # 随机生成使用16进制表示的颜色
    colors = tuple([(np.random.random(), np.random.random(), np.random.random()) for i in range(n_clusters)])
    colors = [rgb2hex(x) for x in colors]  # from  matplotlib.colors import  rgb2hex

    for i, color in enumerate(colors):
        need_idx = np.where(kmeans.labels_ == i)[0]
        ax.scatter(bond_length_arr[need_idx], np.zeros(len(need_idx)), c=color, label=i)
    # 标出聚类的中心位置
    for i, center in enumerate(kmeans.cluster_centers_):
        ax.scatter(center, 0, c=colors[i], marker='x', s=150, linewidth=3, zorder=10)
        labels = [f'Cluster {i}: {kmeans.cluster_centers_[i]}' for i in range(n_clusters)]
        legend = ax.legend(loc='upper right', labels=labels)

    plt.savefig(filename)


if __name__ == '__main__':
    # 读取STEM电镜图片
    filepath = dlg.get_open_file_name("Please select the path where the carbon matrix is located",
                                       dlg.FileFilter.AllFiles)
    if filepath == '':
        dlg.show_message(f"Unselected file")
        exit(-1)
    img = cv2.imread(filepath)
    filename_1 = 'atom_coord_cal_bond.png'
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将gray_img_separate的像素值归一化处理
    gray_img_normalized = gray_img / np.max(gray_img)
    # todo:指定像素值的阈值，并获取符合条件的点位坐标
    threshold_background = 0.3
    # todo: 根据dose值的不同需要调整
    threshold_atom_detect = 0.7
    gray_img_separate = np.where(gray_img_normalized > threshold_background, gray_img_normalized, 0)

    points = np.transpose(np.where(gray_img_separate >= 0))  # 提取所有点位
    points_list = points.tolist()

    for p_list in points_list:
        if gray_img_separate[p_list[0], p_list[1]] == 0:
            p_list.append(0)  # 将像素值为0的点位z坐标赋为0
        else:
            p_list.append(gray_img_separate[p_list[0], p_list[1]])  # 将点位的像素值作为z坐标

    # points_arr保存所有点位信息，并绘制和保存三维图片
    points_arr = np.array(points_list)

    # 创建图像并设置数据源
    # Triangulation对象用于创建三角形面片
    tri = Triangulation(points_arr[:, 0], points_arr[:, 1])

    # 创建3D坐标系
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面
    ax.plot_trisurf(points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], triangles=tri.triangles, cmap=plt.cm.Spectral)

    # 设置坐标轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 显示图像
    plt.savefig('3d_points_surface.png')

    # todo: 输入电镜实际的尺寸大小以及原子数的范围
    y_real, x_real = input("please input the size of stem: Use commas to separate, first y second x ").split(',')
    y_real, x_real = float(y_real), float(x_real)
    min_clusters, max_clusters = input("Please enter the range of atomic numbers in the STEM picture: "
                                       "Use commas to separate, first min second max ").split(',')
    min_clusters, max_clusters = int(min_clusters), int(max_clusters)

    # 根据z轴的值对原子簇进行聚类，聚类的数目可以手动指定或者使用聚类算法自动确定
    z = points_arr[:, 2]
    best_k, elbow_plot = elbow_method(points_arr, min_clusters, max_clusters)
    print(f"The best k is {best_k}")
    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(points_arr)
    labels = kmeans.labels_

    # 对每个原子簇拟合一个高斯分布
    centers = kmeans.cluster_centers_
    centers = centers.tolist()
    gmm_models_center = []
    for center_index in range(len(centers)):
        center = centers[center_index]
        data = points_arr[labels == center_index]
        gmm = GaussianMixture(n_components=1, covariance_type='full')
        gmm.fit(data)
        gmm_models_center.append(gmm.means_.tolist())
    gmm_models_center_arr = np.array(gmm_models_center)

    # 之前通过阈值来检测原子的操作
    coords = gmm_models_center_arr.squeeze()

    # 计算点位坐标在图片中的相对位置，保存在 DataFrame中
    coords_df = pd.DataFrame(coords, columns=['row', 'col', 'pixel_value'])
    coords_df['x'] = coords_df['col'] / gray_img.shape[1]
    coords_df['y'] = coords_df['row'] / gray_img.shape[0]
    atom_nearest_indices_list = []
    atom_nearest_distances_list = []
    for point_index in range(coords.shape[0]):
        point_x = coords[point_index][0]
        point_y = coords[point_index][1]
        point = np.array([point_x, point_y])
        coords_pos = coords[:, 0:2]
        nearest_indices, nearest_dists = find_nearest_neighbors(coords_pos, point)
        atom_nearest_indices_list.append(nearest_indices)
        atom_nearest_distances_list.append(nearest_dists)

    atom_nearest_distances_arr = np.array(atom_nearest_distances_list)
    atom_nearest_indices_arr = np.array(atom_nearest_indices_list) + 1
    # 对atom_indices和atom_distances进行筛选
    atom_nearest_distances_arr = np.where(atom_nearest_distances_arr > 5, atom_nearest_distances_arr, 0)
    atom_nearest_mask_arr = np.where(atom_nearest_distances_arr > 0, 1, 0)
    atom_nearest_indices_arr = atom_nearest_indices_arr * atom_nearest_mask_arr
    atom_indices_df = pd.DataFrame(atom_nearest_indices_arr,
                                   columns=['atom_nearest_indices_1', 'atom_nearest_indices_2'])
    atom_distances_df = pd.DataFrame(atom_nearest_distances_arr,
                                     columns=['atom_nearest_distance_1', 'atom_nearest_distance_2'])
    coords_df = pd.concat([coords_df, atom_indices_df, atom_distances_df], axis=1)
    # 保存点位的像素值和坐标到 CSV 文件
    coords_df['index'] = np.arange(1, coords_df.shape[0] + 1, dtype=int)
    # 将其转化为真实图片尺寸的大小
    for i in range(1, 3):
        coords_df[f'atom_nearest_distance_{i}'] = (coords_df[f'atom_nearest_distance_{i}'] / gray_img.shape[0] * y_real
                                                   + coords_df[f'atom_nearest_distance_{i}'] / gray_img.shape[1] * x_real) / 2
    coords_df.to_csv('STEM_atom_coordination_information.csv', index=False)

    # 拼接三列数据成一维数组
    bond_length_arr = pd.concat([coords_df[f'atom_nearest_distance_1'], coords_df[f'atom_nearest_distance_2']]).to_numpy()
    bond_length_arr = bond_length_arr[bond_length_arr != 0]
    bond_length_arr = bond_length_arr.reshape(-1, 1)
    # 将不同点的像素值进行聚类分析
    n_clusters = 1  # 假设类别数为 1
    kmeans_visualize(bond_length_arr, n_clusters, filename_1)

