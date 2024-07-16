from visualization_tree import save_load_view_points, save_load_view_point
import open3d as o3d
import glob
import os
import numpy as np
import torch
from utils import PointLoss

predict_folder = 'predict_CD/predict/'
reconstruction_folder = 'predict_CD/reconstruction/'
incomplete_folder = 'predict_CD/incomplete/'

i = 21
total_predict_CD = 0
total_incomplete_CD = 0

for pcd_name in glob.glob(os.path.join(predict_folder, '*.pts')):
    # 读取预测、不完整和重建点云
    pure_name = os.path.split(os.path.splitext(pcd_name)[0])[1]
    predict_pc = np.loadtxt(pcd_name).astype(np.float64)
    incomplete_pc = np.loadtxt(incomplete_folder + pure_name + '.pts').astype(np.float64)[:, 0:3]
    reconstruction_pc = np.loadtxt(reconstruction_folder + pure_name + '.pts').astype(np.float64)

    # 随机采样不完整和重建
    crop_point_num = 2048
    idx = np.random.choice(reconstruction_pc.shape[0], crop_point_num, replace=False)
    reconstruction_pc = reconstruction_pc[idx]
    idx = np.random.choice(incomplete_pc.shape[0], crop_point_num, replace=False)
    incomplete_pc = incomplete_pc[idx]

    # 平移配准
    x_translation = 0.2
    y_translation = 0.0
    z_translation = 42.05
    reconstruction_pc[:, 0] = reconstruction_pc[:, 0] + x_translation
    reconstruction_pc[:, 1] = reconstruction_pc[:, 1] + y_translation
    reconstruction_pc[:, 2] = reconstruction_pc[:, 2] + z_translation

    # 计算CD值
    predict_torch = torch.tensor(predict_pc).unsqueeze(0)
    incomplete_torch = torch.tensor(incomplete_pc).unsqueeze(0)
    reconstruction_torch = torch.tensor(reconstruction_pc)[:, 0:3].unsqueeze(0)

    criterion_PointLoss = PointLoss()

    CD_predict = criterion_PointLoss(reconstruction_torch, predict_torch)
    print(pure_name, ' predict CD:', str(CD_predict))
    total_predict_CD = total_predict_CD + CD_predict

    CD_incomplete = criterion_PointLoss(reconstruction_torch, incomplete_torch)
    print(pure_name, ' incomplete CD:', str(CD_incomplete))
    total_incomplete_CD = total_incomplete_CD + CD_incomplete

    # 保存配准后的文件
    saveroot = '.'
    if not os.path.exists(saveroot + '/register/register_reconstruction_pc'):
        os.makedirs(saveroot + '/register/register_reconstruction_pc')
    np.savetxt(saveroot + '/register/register_reconstruction_pc/' + pure_name + '.pts', reconstruction_pc, fmt="%f %f %f %f %f %f")

    # 可视化
    min_pc = np.min(predict_pc, axis=0)
    max_pc = np.max(predict_pc, axis=0)
    max_val = np.max(max_pc - min_pc)
    for i in range(3):
        predict_pc[:, i] = (predict_pc[:, i] - min_pc[i]) / max_val
        incomplete_pc[:, i] = (incomplete_pc[:, i] - min_pc[i]) / max_val
        reconstruction_pc[:, i] = (reconstruction_pc[:, i] - min_pc[i]) / max_val

    win_size = [800, 800]
    pc_size = 5.0

    pcd_predict = o3d.geometry.PointCloud()
    pcd_predict.points = o3d.utility.Vector3dVector(predict_pc)
    pcd_incomplete = o3d.geometry.PointCloud()
    pcd_incomplete.points = o3d.utility.Vector3dVector(incomplete_pc)
    pcd_reconstruction = o3d.geometry.PointCloud()
    pcd_reconstruction.points = o3d.utility.Vector3dVector(reconstruction_pc[:, 0:3])
    pcd_reconstruction.colors = o3d.utility.Vector3dVector(reconstruction_pc[:, 3:6]/256.0)

    print("pcd_predict + pcd_reconstruction")
    save_load_view_points(pcd1=pcd_predict, pcd2=pcd_reconstruction, filename="../o3d.json", win_size=win_size, pc_size=pc_size,
                          png_name=pure_name + '_' + 'register_predict.png')
    print("pcd_incomplete + pcd_reconstruction")
    save_load_view_points(pcd1=pcd_incomplete, pcd2=pcd_reconstruction, filename="../o3d.json", win_size=win_size,
                          pc_size=pc_size,
                          png_name=pure_name + '_' + 'register_incomplete.png')
    print("pcd_predict")
    save_load_view_point(pcd=pcd_predict, filename="../o3d.json", win_size=win_size,
                         pc_size=pc_size,
                         png_name=pure_name + '_' + 'predict.png')
    print("pcd_incomplete")
    save_load_view_point(pcd=pcd_incomplete, filename="../o3d.json", win_size=win_size,
                         pc_size=pc_size,
                         png_name=pure_name + '_' + 'incomplete.png')
    print("pcd_reconstruction")
    save_load_view_point(pcd=pcd_reconstruction, filename="../o3d.json", win_size=win_size,
                         pc_size=pc_size,
                         png_name=pure_name + '_' + 'reconstruction.png')

    print("done!")

print('mean predict CD:', str(total_predict_CD/i))
print('mean incomplete CD:', str(total_incomplete_CD/i))

