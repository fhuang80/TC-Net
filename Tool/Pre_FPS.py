import glob
import os
import numpy as np
import torch
from utils import farthest_point_sample

fuse_folder = '../../../Data/OwnTree_Predict_v2/Fuse/'
fuse_fps_folder = '../../../Data/OwnTree_Predict_v2/Fuse_FPS/'
car_fps_folder = '../../../Data/OwnTree_Predict_v2/Car_FPS/'
car_folder = '../../../Data/OwnTree_Predict_v2/00000000/points/'

for pcd_name in glob.glob(os.path.join(fuse_folder, '*.pts')):
    # 0.读取融合点云
    pure_name = os.path.split(os.path.splitext(pcd_name)[0])[1]
    fuse_pc = np.loadtxt(pcd_name).astype(np.float64)

    # 1.保存完整车载
    car_pc = fuse_pc[fuse_pc[:, 7] == 1]
    np.savetxt(car_folder + pure_name + '.pts', car_pc[:, 0:3], fmt="%f %f %f")

    # 2.保存2048点融合-fuse_fps
    fuse_pc_torch = torch.tensor(fuse_pc[:, 0:3])
    fuse_pc_torch = torch.unsqueeze(fuse_pc_torch, 0)
    fuse_fps_index = farthest_point_sample(fuse_pc_torch, 2048, Float64=True)
    fuse_fps = fuse_pc_torch[0, fuse_fps_index, :]
    fuse_fps = torch.squeeze(fuse_fps)
    fuse_fps = fuse_fps.numpy()
    np.savetxt(fuse_fps_folder + pure_name + '.pts', fuse_fps[:, 0:3], fmt="%f %f %f")

    # 3.保存2048点车载-car_fps
    car_pc_torch = torch.tensor(car_pc[:, 0:3])
    car_pc_torch = torch.unsqueeze(car_pc_torch, 0)
    car_fps_index = farthest_point_sample(car_pc_torch, 2048, Float64=True)
    car_fps = car_pc_torch[0, car_fps_index, :]
    car_fps = torch.squeeze(car_fps)
    car_fps = car_fps.numpy()
    np.savetxt(car_fps_folder + pure_name + '.pts', car_fps[:, 0:3], fmt="%f %f %f")