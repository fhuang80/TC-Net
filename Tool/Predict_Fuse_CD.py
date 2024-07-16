from visualization_tree_my import save_load_view_points, save_load_view_point
import open3d as o3d
import glob
import os
import numpy as np
import torch
from utils import PointLoss


def pc_normalize(pc):
    """ pc: NxC, return NxC """
    min_pc = np.min(pc, axis=0)
    max_pc = np.max(pc, axis=0)
    pc = (pc - min_pc) / (max_pc - min_pc)
    return pc


# 0.初始化
visual = False
predict_folder = '../../../Predict/OwnTree_Predict_v2/230314_v7_routetype1_missingtype3_Tree_pretrain_0/G200/predict/'
car_fps_folder = '../../../Data/OwnTree_Predict_v2/Car_FPS/'
fuse_fps_folder = '../../../Data/OwnTree_Predict_v2/Fuse_FPS/'
total_car_CD = 0
total_predict_CD = 0
total_norm_car_CD = 0
total_norm_predict_CD = 0
N_CD = 0
f=open('CD.txt','w')

for car_fps_name in glob.glob(os.path.join(car_fps_folder, '*.pts')):
    # 读取数据
    N_CD = N_CD + 1
    pure_name = os.path.split(os.path.splitext(car_fps_name)[0])[1]
    car_fps_pc = np.loadtxt(car_fps_name).astype(np.float64)
    fuse_fps_pc = np.loadtxt(fuse_fps_folder + pure_name + '.pts').astype(np.float64)
    predict_pc = np.loadtxt(predict_folder + pure_name + '.txt').astype(np.float64)
    criterion_PointLoss = PointLoss()

    # 计算绝对CD值
    predict_torch = torch.tensor(predict_pc).unsqueeze(0)
    car_fps_torch = torch.tensor(car_fps_pc).unsqueeze(0)
    fuse_fps_torch = torch.tensor(fuse_fps_pc).unsqueeze(0)

    CD_car = criterion_PointLoss(car_fps_torch, fuse_fps_torch)
    print('Car CD of',  pure_name, 'is', CD_car.item())
    f.write('Car CD of ' + pure_name + ' is ' + str(CD_car.item()) + '\n')
    total_car_CD = total_car_CD + CD_car

    CD_predict = criterion_PointLoss(predict_torch, fuse_fps_torch)
    print('Predict CD of', pure_name, 'is', CD_predict.item())
    f.write('Predict CD of ' + pure_name + ' is ' + str(CD_predict.item()) + '\n')
    total_predict_CD = total_car_CD + CD_predict

    # 归一化后计算相对CD值
    car_fps_norm_pc = pc_normalize(car_fps_pc)
    fuse_fps_norm_pc = pc_normalize(fuse_fps_pc)
    predict_norm_pc = pc_normalize(predict_pc)

    predict_norm_torch = torch.tensor(predict_norm_pc).unsqueeze(0)
    car_fps_norm_torch = torch.tensor(car_fps_norm_pc).unsqueeze(0)
    fuse_fps_norm_torch = torch.tensor(fuse_fps_norm_pc).unsqueeze(0)

    CD_norm_car = criterion_PointLoss(car_fps_norm_torch, fuse_fps_norm_torch)
    print('Car norm CD of', pure_name, 'is', CD_norm_car.item())
    f.write('Car norm CD of ' + pure_name + ' is ' + str(CD_norm_car.item()) + '\n')
    total_norm_car_CD = total_norm_car_CD + CD_norm_car

    CD_norm_predict = criterion_PointLoss(predict_norm_torch, fuse_fps_norm_torch)
    print('Predict norm CD of', pure_name, 'is', CD_norm_predict.item())
    f.write('Predict norm CD of ' + pure_name + ' is ' + str(CD_norm_predict.item()) + '\n')
    total_norm_predict_CD = total_norm_predict_CD + CD_norm_predict

    # 使用归一化点云完成可视化
    if visual:
        min_pc = np.min(predict_pc, axis=0)
        max_pc = np.max(predict_pc, axis=0)
        max_val = np.max(max_pc - min_pc)
        for i in range(3):
            predict_pc[:, i] = (predict_pc[:, i] - min_pc[i]) / max_val
            car_fps_pc[:, i] = (car_fps_pc[:, i] - min_pc[i]) / max_val
            fuse_fps_pc[:, i] = (fuse_fps_pc[:, i] - min_pc[i]) / max_val

        win_size = [800, 800]
        pc_size = 8.0

        pcd_predict = o3d.geometry.PointCloud()
        pcd_predict.points = o3d.utility.Vector3dVector(predict_pc)
        pcd_car_fps = o3d.geometry.PointCloud()
        pcd_car_fps.points = o3d.utility.Vector3dVector(car_fps_pc)
        pcd_fuse_fps = o3d.geometry.PointCloud()
        pcd_fuse_fps.points = o3d.utility.Vector3dVector(fuse_fps_pc)

        save_load_view_point(pcd=pcd_fuse_fps, filename="../o3d.json", win_size=win_size,
                             pc_size=pc_size,
                             png_name=pure_name + '_' + 'fuse_fps.png')
        save_load_view_point(pcd=pcd_car_fps, filename="../o3d.json", win_size=win_size,
                             pc_size=pc_size,
                             png_name=pure_name + '_' + 'car_fps.png')
        save_load_view_point(pcd=pcd_predict, filename="../o3d.json", win_size=win_size,
                             pc_size=pc_size,
                             png_name=pure_name + '_' + 'predict.png')

        pcd_car_fps.colors = o3d.utility.Vector3dVector(np.zeros_like(car_fps_pc))
        pcd_predict.colors = o3d.utility.Vector3dVector(np.zeros_like(predict_pc))
        save_load_view_points(pcd1=pcd_car_fps, pcd2=pcd_fuse_fps, filename="../o3d.json", win_size=win_size,
                              pc_size=pc_size,
                              png_name=pure_name + '_' + 'register_car_fps.png')
        save_load_view_points(pcd1=pcd_predict, pcd2=pcd_fuse_fps, filename="../o3d.json", win_size=win_size,
                              pc_size=pc_size,
                              png_name=pure_name + '_' + 'register_predict.png')

print('mean car CD is', total_car_CD.item() / N_CD)
print('mean predict CD is', total_predict_CD.item() / N_CD)
print('mean norm car CD is', total_norm_car_CD.item() / N_CD)
print('mean norm predict CD is', total_norm_predict_CD.item() / N_CD)
f.write('mean car CD is ' + str(total_car_CD.item() / N_CD) + '\n')
f.write('mean predict CD is ' + str(total_predict_CD.item() / N_CD) + '\n')
f.write('mean norm car CD is ' + str(total_norm_car_CD.item() / N_CD) + '\n')
f.write('mean norm predict CD is ' + str(total_norm_predict_CD.item() / N_CD) + '\n')

pass
