import os
import glob
import open3d as o3d
import json

Data_DIR = '../../../Data/OwnTree_Predict_v2/00000000/points'
Save_DIR = '../../../Data/OwnTree_Predict_v2/train_test_split'

pts_list = []

for pcd_name in glob.glob(os.path.join(Data_DIR, '*.pts')):
    # 存json文件
    pure_name = os.path.split(os.path.splitext(pcd_name)[0])[1]
    pts_list.append("shape_data/11111111/" + pure_name)

    # 转存pts
    # pcd = o3d.io.read_point_cloud(pcd_name)
    # o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud(os.path.join(Save_DIR, pure_name + '.pts'), pcd)

with open(os.path.join(Save_DIR, "shuffled_train_file_list_11111111.json"), "w") as f:
    json.dump(pts_list, f)
    print("生成成功")

