import os
import glob
import open3d as o3d
import numpy as np
import time

from open3d.cpu.pybind.visualization import Visualizer


def save_load_view_point(pcd, filename, win_size, pc_size, png_name, hull=None):
    vis: Visualizer = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=win_size[0], height=win_size[1])
    render_option: o3d.visualization.RenderOption = vis.get_render_option()
    render_option.point_size = pc_size
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    if hull != None:
        vis.add_geometry(hull)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.capture_screen_image(png_name)
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def save_load_view_points(pcd, filename, png_name):
    # 初始化参数
    pc_size = 8.0
    win_size = [800, 800]

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=win_size[0], height=win_size[1])

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = pc_size

    # 读取并应用摄像头参数
    param = o3d.io.read_pinhole_camera_parameters(filename)

    # 修改摄像头参数以改变视图角度
    # 例如，设置一个从侧面看的视角
    ctr = vis.get_view_control()
    ctr.set_front([0.0, -1.0, 0.0])  # 方向
    ctr.set_lookat([0.0, 0.0, 0.0])  # 注视点
    ctr.set_up([0.0, 0.0, 1.0])  # 上方向
    ctr.set_zoom(0.5)  # 缩放

    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)

    # 运行可视化并捕获屏幕图像
    vis.run()
    vis.capture_screen_image(png_name)

    # 保存当前视图的摄像头参数
    param = ctr.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)

    # 销毁窗口
    vis.destroy_window()