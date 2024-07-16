from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import numpy as np

# 做数据增强的代码
"""
用于计算绕指定轴旋转一定角度的旋转矩阵。它接受一个角度(angle)和一个轴向量(axis)作为输入，并返回一个3x3的旋转矩阵(torch.Tensor)。
"""
def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


"""
用于对点云进行缩放操作。它接受一个缩放因子范围(lo和hi)作为初始化参数，并在调用时将点云的前三个维度按照随机缩放因子进行缩放。
"""
class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


"""
用于对点云进行旋转操作。它接受一个轴向量(axis)作为初始化参数，并在调用时随机生成旋转角度，然后根据旋转角度和轴向量计算旋转矩阵，将点云进行旋转。
"""
class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


"""
用于对点云进行扰动旋转操作。它接受扰动角度的标准差(angle_sigma)和角度的截断范围(angle_clip)作为初始化参数，并在调用时随机生成扰动角度，然后根据扰动角度分别绕x、y、z轴进行旋转，最后将点云进行旋转。
"""
class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


"""
用于对点云进行抖动操作。它接受抖动的标准差(std)和抖动的截断范围(clip)作为初始化参数，并在调用时为点云生成随机的抖动数据，并将其添加到点云的前三个维度上。
"""
class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
            .normal_(mean=0.0, std=self.std)
            .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


"""
用于对点云进行平移操作。它接受平移范围(translate_range)作为初始化参数，并在调用时随机生成平移向量，将点云进行平移。
"""
class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation
        return points


"""
用于将点云数据转换为torch.Tensor类型
"""
class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


"""
用于对点云数据进行随机输入丢失操作。它接受最大丢失比例(max_dropout_ratio)作为初始化参数，并在调用时根据丢失比例随机选择一些点进行丢失。
"""
class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        pc = points.numpy()

        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return torch.from_numpy(pc).float()
