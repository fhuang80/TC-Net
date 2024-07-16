from visualization_tree_my import save_load_view_points, save_load_view_point
import open3d as o3d
import glob
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import PointLoss
from utils import calc_distance, pc_add_dis2file




def exponential_normalization(distances, distance_min, distance_max):
    # 计算归一化的指数部分
    exponent = (distances - distance_min) / (distance_max - distance_min)

    # 计算指数归一化的结果
    normalized_values = torch.exp(exponent) - 1

    # 将结果缩放到[0, 1]区间
    normalized_values = normalized_values / (torch.exp(torch.tensor(1.0)) - 1)

    return normalized_values


def main():
    incomplete_folder = 'E:\Postgraduate\Department\sychen\PCSS\Code\TC-Net-v1\\register\\register_incomplete_pc\\0.pts'
    reconstruction_folder = 'E:\Postgraduate\Department\sychen\PCSS\Code\TC-Net-v1\\register\\register_reconstruction_pc\\0.pts'

    incomplete_np = np.loadtxt(incomplete_folder).astype(np.float64)
    reconstruction_np = np.loadtxt(reconstruction_folder).astype(np.float64)

    # 计算CD值
    incomplete_torch = torch.tensor(incomplete_np[:, 0:3]).unsqueeze(0)
    reconstruction_torch = torch.tensor(reconstruction_np)[:, 0:3].unsqueeze(0)

    criterion_PointLoss = PointLoss()

    CD_incomplete = criterion_PointLoss(reconstruction_torch, incomplete_torch)
    print(incomplete_folder[-4:-1], ' incomplete CD:', str(CD_incomplete))

    # 归一化
    min_pc = np.min(incomplete_np, axis=0)
    max_pc = np.max(incomplete_np, axis=0)
    max_val = np.max(max_pc - min_pc)
    for i in range(3):
        incomplete_np[:, i] = (incomplete_np[:, i] - min_pc[i]) / max_val
        reconstruction_np[:, i] = (reconstruction_np[:, i] - min_pc[i]) / max_val

    # predict_np，incomplete_pc
    # 补全前和补全后的distances；必须放一起做距离归一化!
    distances1 = calc_distance(torch.tensor(incomplete_np[:, 0:3]), torch.tensor(reconstruction_np[:, 0:3]))
    distances = distances1
    distance_max = torch.max(distances)
    distance_min = torch.min(distances)

    # 创建一个颜色映射
    color_map = plt.get_cmap("hot")  # 你可以选择其他的 colormap 如 "hot", "cool", "magma" 等

    save_names = ["incomplete", "predict", "ground_truth"]
    pcds = [incomplete_np, reconstruction_np[:, 0:3]]
    for i in range(0, 2):
        # 将标准化的数值转换为颜色
        normalized_values = (distances - distance_min) / distance_max
        # normalized_values = (torch.log(1 + (distances[i] - distance_min))
        #                      / torch.log(1 + (distance_max - distance_min)))
        # normalized_values = exponential_normalization(distances[i], distance_min, distance_max)
        colors = color_map(normalized_values)[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(reconstruction_np[:, 0:3])
        pcd.colors = o3d.utility.Vector3dVector(colors)

        save_pts = np.concatenate((reconstruction_np[:, 0:3], colors), axis=1)

        # save_load_view_points(pcd=pcd, filename="../o3d.json",
        #                       png_name=pure_name + '_' + save_names[i] + '.png')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # point_range = range(0, points.shape[0], skip) # skip points to prevent crash
        point_range = range(0, reconstruction_np[:, 0:3].shape[0])
        ax.scatter(reconstruction_np[:, 0:3][point_range, 0],  # x
                   reconstruction_np[:, 0:3][point_range, 1],  # y
                   reconstruction_np[:, 0:3][point_range, 2],  # z
                   # c=normalized_values[point_range],  # height data for color
                   c=colors[point_range],  # height data for color
                   # cmap='summer',
                   # cmap='hot',
                   # cmap='Wistia',
                   marker="o",
                   s=12)

        # 设置坐标轴的范围
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

        # 设置坐标轴刻度的间隔
        ax.set_xticks(np.arange(0, 1, 0.1))  # x轴刻度间隔
        ax.set_yticks(np.arange(0, 1, 0.1))  # y轴刻度间隔
        ax.set_zticks(np.arange(0, 1, 0.1))  # z轴刻度间隔

        # 设置坐标轴字体大小
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='z', which='major', labelsize=7, pad=15)

        ax.axis('scaled')  # {'equal', 'scaled'}

        plt.show()

        print("done")


if __name__ == '__main__':
    main()