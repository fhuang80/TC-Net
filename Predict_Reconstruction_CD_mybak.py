from visualization_tree_my import save_load_view_points, save_load_view_point
import open3d as o3d
import glob
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import PointLoss
from utils import calc_distance, pc_add_dis2file

predict_folder = 'predict_CD/predict/'
reconstruction_folder = 'predict_CD/reconstruction/'
incomplete_folder = 'predict_CD/incomplete/'


def exponential_normalization(distances, distance_min, distance_max):
    # 计算归一化的指数部分
    exponent = (distances - distance_min) / (distance_max - distance_min)

    # 计算指数归一化的结果
    normalized_values = torch.exp(exponent) - 1

    # 将结果缩放到[0, 1]区间
    normalized_values = normalized_values / (torch.exp(torch.tensor(1.0)) - 1)

    return normalized_values


def main():
    total_predict_CD = 0
    total_incomplete_CD = 0
    # 可视化效果好的。注释后是CD值
    vis_good_example = {"0",   # 23.88 -> 13.07
                        "1",  # 19 -> 15
                        "2",
                        "20",  # CD值变大了，0.6
                        "4",  # CD值变小了，1.2
                        "5",  # CD值没咋变
                        "7",  # CD值变小了，1
                        "9"  # CD值小了，2；可视化整体来说颜色变暗了；但还是有部分没有补全出来
                        }

    for pcd_name in glob.glob(os.path.join(predict_folder, '*.pts')):
        # 读取预测、不完整和重建点云
        pure_name = os.path.split(os.path.splitext(pcd_name)[0])[1]

        # 控制只显示效果好的 or 不好的
        # if pure_name not in vis_good_example:
        #     continue

        predict_np = np.loadtxt(pcd_name).astype(np.float64)
        incomplete_np = np.loadtxt(incomplete_folder + pure_name + '.pts').astype(np.float64)[:, 0:3]
        reconstruction_np = np.loadtxt(reconstruction_folder + pure_name + '.pts').astype(np.float64)

        # 随机采样不完整和重建
        crop_point_num = 2048
        idx = np.random.choice(reconstruction_np.shape[0], crop_point_num, replace=False)
        reconstruction_np = reconstruction_np[idx]
        idx = np.random.choice(incomplete_np.shape[0], crop_point_num, replace=False)
        incomplete_np = incomplete_np[idx]

        # 平移配准
        x_translation = 0.2
        y_translation = 0.0
        z_translation = 42.05
        reconstruction_np[:, 0] = reconstruction_np[:, 0] + x_translation
        reconstruction_np[:, 1] = reconstruction_np[:, 1] + y_translation
        reconstruction_np[:, 2] = reconstruction_np[:, 2] + z_translation

        # 计算CD值
        predict_torch = torch.tensor(predict_np).unsqueeze(0)
        incomplete_torch = torch.tensor(incomplete_np).unsqueeze(0)
        reconstruction_torch = torch.tensor(reconstruction_np)[:, 0:3].unsqueeze(0)

        criterion_PointLoss = PointLoss()

        CD_predict = criterion_PointLoss(reconstruction_torch, predict_torch)
        print(pure_name, ' predict CD:', str(CD_predict))
        total_predict_CD = total_predict_CD + CD_predict

        CD_incomplete = criterion_PointLoss(reconstruction_torch, incomplete_torch)
        print(pure_name, ' incomplete CD:', str(CD_incomplete))
        total_incomplete_CD = total_incomplete_CD + CD_incomplete

        # 保存配准后的文件
        save_root = '.'
        if not os.path.exists(save_root + '/register/register_reconstruction_pc'):
            os.makedirs(save_root + '/register/register_reconstruction_pc')
        np.savetxt(save_root + '/register/register_reconstruction_pc/' + pure_name + '.pts',
                   reconstruction_np, fmt="%f %f %f %f %f %f")

        # 归一化
        min_pc = np.min(predict_np, axis=0)
        max_pc = np.max(predict_np, axis=0)
        max_val = np.max(max_pc - min_pc)
        for i in range(3):
            predict_np[:, i] = (predict_np[:, i] - min_pc[i]) / max_val
            incomplete_np[:, i] = (incomplete_np[:, i] - min_pc[i]) / max_val
            reconstruction_np[:, i] = (reconstruction_np[:, i] - min_pc[i]) / max_val

        # predict_np，incomplete_pc
        # 补全前和补全后的distances；必须放一起做距离归一化!
        distances1 = calc_distance(torch.tensor(incomplete_np), torch.tensor(reconstruction_np[:, 0:3]))
        distances2 = calc_distance(torch.tensor(predict_np), torch.tensor(reconstruction_np[:, 0:3]))
        distances = [distances1, distances2, torch.zeros_like(distances1)]
        distances_comp = torch.cat([distances1, distances2], dim=0)
        distance_max = torch.max(distances_comp)
        distance_min = torch.min(distances_comp)

        # 创建一个颜色映射
        color_map = plt.get_cmap("hot")  # 你可以选择其他的 colormap 如 "hot", "cool", "magma" 等

        save_names = ["incomplete", "predict", "ground_truth"]
        pcds = [incomplete_np, predict_np, reconstruction_np[:, 0:3]]
        for i in range(0, 2):
            # 将标准化的数值转换为颜色
            normalized_values = (distances[i] - distance_min) / distance_max
            # normalized_values = (torch.log(1 + (distances[i] - distance_min))
            #                      / torch.log(1 + (distance_max - distance_min)))
            # normalized_values = exponential_normalization(distances[i], distance_min, distance_max)
            colors = color_map(normalized_values)[:, :3]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(reconstruction_np[:, 0:3])
            pcd.colors = o3d.utility.Vector3dVector(colors)

            save_pts = np.concatenate((reconstruction_np[:, 0:3], colors), axis=1)
            np.savetxt(f"similarity_pre/{pure_name}_{save_names[i]}.pts", save_pts, fmt="%f %f %f %f %f %f")

            save_load_view_points(pcd=pcd, filename="../o3d.json",
                                  png_name=pure_name + '_' + save_names[i] + '.png')

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