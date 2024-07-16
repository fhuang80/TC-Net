import math

import torch
import random
from utils import distance_squre


def random_crop(real_point, input_cropped1, crop_point_num):
    # 初始化，并去除多余维度->变为B*N*1/3
    batch_size, _, pnum, _ = real_point.size()
    real_point = torch.squeeze(real_point, 1)
    input_cropped1 = torch.squeeze(input_cropped1, 1)

    b = crop_point_num
    distance_list = torch.rand(batch_size, b)
    distance_list, distance_order = torch.sort(distance_list, dim=1, descending=True)
    batch_index = torch.arange(batch_size).unsqueeze(1)
    batch_index = batch_index + torch.zeros(1, b)
    batch_index = batch_index.reshape(-1).to(torch.long)
    distance_index = distance_order[:, 0:b]
    distance_index = distance_index.reshape(-1).to(torch.long)
    sequence_index = torch.arange(b).unsqueeze(0)
    sequence_index = torch.zeros(batch_size, 1) + sequence_index
    sequence_index = sequence_index.reshape(-1).to(torch.long)
    input_cropped1[batch_index, distance_index] = torch.FloatTensor([0, 0, 0])

    # 恢复维度
    input_cropped1 = torch.unsqueeze(input_cropped1, 1)

    return input_cropped1



def random_center(real_point, input_cropped1, real_center):

    batch_size, _, pnum, _ = real_point.size()
    crop_point_num = real_center.size()[2]
    choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]), torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]

    for m in range(batch_size):
        index = random.sample(choice, 1)
        distance_list = []
        p_center = index[0]
        for n in range(pnum):
            distance_list.append(distance_squre(real_point[m, 0, n], p_center))
        distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])

        for sp in range(crop_point_num):
            input_cropped1.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])
            real_center.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]]

    return input_cropped1, real_center


def distance_route_vectorization(Ps, A, B):
    # P - B*N*3, A/B - 3
    batch_size, pnum, _ = Ps.size()
    Ps = torch.reshape(Ps, (-1, 3))
    # 计算向量AB和AP，利用广播机制
    AB = B - A
    AP = Ps.sub(AB)
    # P在AB上的投影点Q
    t = torch.mv(AP, AB) / AB.dot(AB)
    t = torch.unsqueeze(t, 1)
    t = AB * t
    Q = A + t
    PQ = Ps - Q
    # 计算距离
    distance = torch.linalg.norm(PQ, axis=1)
    distance = torch.reshape(distance, (batch_size, pnum))

    return distance


def random_vehicle_route(real_point, input_cropped1, real_center, route_type, missing_type):
    # 简化版，使用到车道线的距离而非冠层距离

    # 初始化，并去除多余维度->变为B*N*1/3
    batch_size, _, pnum, _ = real_point.size()
    crop_point_num = real_center.size()[2]
    real_point = torch.squeeze(real_point, 1)
    input_cropped1 = torch.squeeze(input_cropped1, 1)
    real_center = torch.squeeze(real_center, 1)

    # 给定随机的车道线上的两点A，B（对应12条边+12条中线）
    if route_type == 1:
        rand_n = random.randint(0, 3)
        if rand_n == 0:
            A = torch.Tensor([0, 0, 0])
            B = torch.Tensor([1, 0, 0])
        elif rand_n == 1:
            A = torch.Tensor([0, 0, 0])
            B = torch.Tensor([0, 1, 0])
        elif rand_n == 2:
            A = torch.Tensor([0, 1, 0])
            B = torch.Tensor([1, 1, 0])
        elif rand_n == 3:
            A = torch.Tensor([1, 0, 0])
            B = torch.Tensor([1, 1, 0])

    elif route_type == 2:
        rand_n = random.randint(0, 3)
        offset = random.randint(0, 3)
        if rand_n == 0:
            A = torch.Tensor([0, 0 - offset, 0])
            B = torch.Tensor([1, 0 - offset, 0])
        elif rand_n == 1:
            A = torch.Tensor([0 - offset, 0, 0])
            B = torch.Tensor([0 - offset, 1, 0])
        elif rand_n == 2:
            A = torch.Tensor([0, 1 + offset, 0])
            B = torch.Tensor([1, 1 + offset, 0])
        elif rand_n == 3:
            A = torch.Tensor([1 + offset, 0, 0])
            B = torch.Tensor([1 + offset, 1, 0])

    elif route_type == 3:
        choice_w1_w2 = [torch.Tensor([0, 0]), torch.Tensor([0, 1]), torch.Tensor([1, 0]), torch.Tensor([1, 1]),
                        torch.Tensor([0, 0.5]), torch.Tensor([0.5, 0]), torch.Tensor([1, 0.5]), torch.Tensor([0.5, 1])]
        w1_w2 = random.sample(choice_w1_w2, 1)[0]
        rand_n = random.randint(0, 2)
        if rand_n == 0:
            A = torch.Tensor([w1_w2[0], w1_w2[1], 0])
            B = torch.Tensor([w1_w2[0], w1_w2[1], 1])
        elif rand_n == 1:
            A = torch.Tensor([w1_w2[0], 0, w1_w2[1]])
            B = torch.Tensor([w1_w2[0], 1, w1_w2[1]])
        elif rand_n == 2:
            A = torch.Tensor([0, w1_w2[0], w1_w2[1]])
            B = torch.Tensor([1, w1_w2[0], w1_w2[1]])

    # 为了作图
    elif route_type == 4:
        A = torch.Tensor([0, 0, 0])
        B = torch.Tensor([1, 0, 0])

    # 完全向量化版本
    # 1.计算车道线距离distance_list
    distance_list = distance_route_vectorization(real_point, A, B)
    distance_list, distance_order = torch.sort(distance_list, dim=1, descending=True)

    if missing_type == 1:
        b = 512
        batch_index = torch.arange(batch_size).unsqueeze(1)
        batch_index = batch_index + torch.zeros(1, b)
        batch_index = batch_index.reshape(-1).to(torch.long)
        distance_index = distance_order[:, 0:b]
        distance_index = distance_index.reshape(-1).to(torch.long)
        sequence_index = torch.arange(b).unsqueeze(0)
        sequence_index = torch.zeros(batch_size, 1) + sequence_index
        sequence_index = sequence_index.reshape(-1).to(torch.long)
        input_cropped1[batch_index, distance_index] = torch.FloatTensor([0, 0, 0])
        real_center[batch_index, sequence_index] = real_point[batch_index, distance_index]

    elif missing_type == 2 or missing_type == 3:
        # 2.选取前a个直接保留，后b个直接缺失
        if missing_type == 2:
            a = 1024
            b = 256
            theta = 1
        if missing_type == 3:
            a = random.randint(768, 1280)
            b = random.randint(128, 384)
            theta = random.randint(1, 4)
        c = pnum - a - b
        batch_index = torch.arange(batch_size).unsqueeze(1)
        batch_index = batch_index + torch.zeros(1, b)
        batch_index = batch_index.reshape(-1).to(torch.long)
        distance_index = distance_order[:, 0:b]
        distance_index = distance_index.reshape(-1).to(torch.long)
        sequence_index = torch.arange(b).unsqueeze(0)
        sequence_index = torch.zeros(batch_size, 1) + sequence_index
        sequence_index = sequence_index.reshape(-1).to(torch.long)
        input_cropped1[batch_index, distance_index] = torch.FloatTensor([0, 0, 0])
        real_center[batch_index, sequence_index] = real_point[batch_index, distance_index]
        # 3.从剩下的中凑齐缺失
        distance_list_new = distance_list[:, b:b + c]
        distance_order_new = distance_order[:, b:b + c]
        distance_min = torch.min(distance_list_new, dim=1).values.unsqueeze(1)
        distance_max = torch.max(distance_list_new, dim=1).values.unsqueeze(1)
        distance_min = distance_min + torch.zeros(1, c)
        distance_max = distance_max + torch.zeros(1, c)
        distance_list_new = (distance_list_new - distance_min) / (distance_max - distance_min)
        # 4.转换为缺失概率
        P_missing = -1 * torch.exp(-1 * theta * distance_list_new) + 1
        # 5.按照概率消失，即和均匀分布概率差值最小的消失
        P_missing = torch.rand(batch_size, c) - P_missing
        P_missing, order_missing = torch.sort(P_missing, dim=1)
        res_crop_num = crop_point_num - b
        batch_index = torch.arange(batch_size).unsqueeze(1)
        batch_index = batch_index + torch.zeros(1, res_crop_num)
        batch_index = batch_index.reshape(-1).to(torch.long)
        distance_index = order_missing[:, 0:res_crop_num].reshape(-1).to(torch.long)
        distance_index = distance_order_new[batch_index, distance_index]
        sequence_index = torch.arange(b, b+res_crop_num).unsqueeze(0)
        sequence_index = torch.zeros(batch_size, 1) + sequence_index
        sequence_index = sequence_index.reshape(-1).to(torch.long)
        input_cropped1.data[batch_index, distance_index] = torch.FloatTensor([0, 0, 0])
        real_center.data[batch_index, sequence_index] = real_point[batch_index, distance_index]

    # 6.恢复维度
    input_cropped1 = torch.unsqueeze(input_cropped1, 1)
    real_center = torch.unsqueeze(real_center, 1)

    return input_cropped1, real_center