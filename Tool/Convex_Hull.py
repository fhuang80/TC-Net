import open3d as o3d
import numpy as np
from visualization_tree_my import save_load_view_point
import time
import os
import glob

# 验证我们补全的工作是有效的
pcd_2_folder = "Covex_Hull/Predict - rename - covex"
pcd_folder = "Covex_Hull/Predict - rename - covex"

for pcd_name in glob.glob(os.path.join(pcd_folder, '*.pcd')):
    pure_name = os.path.split(os.path.splitext(pcd_name)[0])[1]

    pcd = o3d.io.read_point_cloud(pcd_name)
    pc = np.array(pcd.points)
    pcd.points = o3d.utility.Vector3dVector(pc)

    pcd_2_name = os.path.join(pcd_2_folder, pure_name+'.pcd')
    pcd2 = o3d.io.read_point_cloud(pcd_2_name)
    pc2 = np.array(pcd2.points)
    pcd2.points = o3d.utility.Vector3dVector(pc2)


    win_size = [800, 800]
    pc_size = 8.0
    t = time.localtime()
    current_time = time.strftime("%H%M%S", t)

    save_load_view_point(pcd=pcd, filename="../o3d.json", win_size=win_size, pc_size=pc_size,
                         png_name=current_time + '_' + str(pure_name) + '_Incomplete_' + 'pc.png')
    # 计算点云的凸包
    hull, _ = pcd.compute_convex_hull(joggle_inputs=False)
    # 计算凸包的体积
    volume = hull.get_volume()
    # 可视化结果
    print(pure_name, ' Incomplete: ', str(volume))
    save_load_view_point(pcd=pcd, hull=hull, filename="../o3d.json", win_size=win_size, pc_size=pc_size,
                         png_name=current_time + '_' + str(pure_name) + '_Incomplete_' + 'hull.png')

    save_load_view_point(pcd=pcd2, filename="../o3d.json", win_size=win_size, pc_size=pc_size,
                         png_name=current_time + '_' + str(pure_name) + '_Predict_' + 'pc.png')
    # 计算点云的凸包
    hull, _ = pcd2.compute_convex_hull(joggle_inputs=False)
    # 计算凸包的体积
    volume = hull.get_volume()
    # 可视化结果
    print(pure_name, ' Predict: ', str(volume))
    save_load_view_point(pcd=pcd2, hull=hull, filename="../o3d.json", win_size=win_size, pc_size=pc_size,
                         png_name=current_time + '_' + str(pure_name) + '_Predict_' + 'hull.png')

