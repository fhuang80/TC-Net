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

def save_load_view_points(pcd1, pcd2, filename, win_size, pc_size, png_name, hull=None):
    vis: Visualizer = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=win_size[0], height=win_size[1])
    render_option: o3d.visualization.RenderOption = vis.get_render_option()
    render_option.point_size = pc_size
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.capture_screen_image(png_name)
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()