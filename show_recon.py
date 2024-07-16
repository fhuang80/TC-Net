#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import argparse
import torch.utils.data
import torchvision.transforms as transforms
import utils
from utils import PointLoss
import data_utils as d_utils
import shapenet_part_loader
from model_TCNet import _netG
from crop_method import random_center, random_vehicle_route
import open3d as o3d
from visualization_tree_my import save_load_view_point
from collections import OrderedDict
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',  default='dataset/train', help='path to dataset')
    parser.add_argument('--saveroot',  default='../../Predict/PF-Net/221202_2/G200_Campus', help='path to save result')
    parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
    parser.add_argument('--crop_point_num',type=int,default=512,help='0 means do not use else use with this weight')
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--netG', default='./Trained_Model/Tree_221202/point_netG200.pth', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--drop',type=float,default=0.2)
    parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
    parser.add_argument('--point_scales_list',type=list,default=[2048,1024,512],help='number of points in each scales')
    parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
    parser.add_argument('--wtl2',type=float,default=0.9,help='0 means do not use else use with this weight')
    parser.add_argument('--cropmethod', default = 'random_vehicle_route', help = 'random_vehicle_route|random_center')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--save_txt', action='store_true', default=False, help='save result [default: False]')
    parser.add_argument('--distributed', action='store_true', default=False, help='distributed [default: False]')
    parser.add_argument('--route_type', type=int, default=1, help='route_type')
    parser.add_argument('--missing_type', type=int, default=1, help='missing_type')
    parser.add_argument('--class_choice', type=str, default=None, help='class_choice')
    opt = parser.parse_args()
    print(opt)

    def distance_squre1(p1,p2):
        return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
        ]
    )

    # test_dset = shapenet_part_loader.PartDataset( root=opt.data_root,classification=True, class_choice='Car', npoints=opt.pnum, split='test')
    test_dset = shapenet_part_loader.PartDataset(root=opt.data_root,classification=True, npoints=opt.pnum, split='test', class_choice=opt.class_choice)
    test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                             shuffle=False,num_workers = int(opt.workers))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
    point_netG.to(device)
    if opt.distributed:
        state_dict = torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
        point_netG.load_state_dict(new_state_dict)
    else:
        point_netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])

    point_netG.eval()
    input_cropped1 = torch.FloatTensor(opt.batchSize, 1, opt.pnum, 3)
    criterion_PointLoss = PointLoss().to(device)
    errG_min = 100
    n = 0

    if not os.path.exists(opt.saveroot + '/crop'):
        os.makedirs(opt.saveroot + '/crop')
    if not os.path.exists(opt.saveroot + '/fake'):
        os.makedirs(opt.saveroot + '/fake')
    if not os.path.exists(opt.saveroot + '/real'):
        os.makedirs(opt.saveroot + '/real')

    num_iter = 0
    errG_all_missing = 0
    errG_all = 0
    for i, data in enumerate(test_dataloader, 0):
        real_point, target, length_ratio = data
        length_ratio = length_ratio.numpy()
        real_point = torch.unsqueeze(real_point, 1)
        batch_size = real_point.size()[0]
        real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
        input_cropped1.resize_(real_point.size()).copy_(real_point)
        input_cropped1 = input_cropped1.cpu()
        p_origin = [0,0,0]

        if opt.cropmethod == 'random_center':
            input_cropped1, real_center = random_center(real_point, input_cropped1, real_center)
        elif opt.cropmethod == 'random_vehicle_route':
            input_cropped1, real_center = random_vehicle_route(real_point, input_cropped1, real_center, opt.route_type, opt.missing_type)

        real_center = torch.squeeze(real_center,1)
        input_cropped1 = torch.squeeze(input_cropped1,1)
        input_cropped1 = input_cropped1.to(device)
        input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
        input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
        input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
        input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
        input_cropped2 = input_cropped2.to(device)
        input_cropped3 = input_cropped3.to(device)
        input_cropped  = [input_cropped1,input_cropped2,input_cropped3]
        fake_center1, fake_center2, fake=point_netG(input_cropped)
        fake = fake.cuda()
        real_center = real_center.cuda()
        # 输入格式B*N*3
        num_iter = num_iter + 1
        errG = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1))
        errG_all_missing = errG_all_missing + errG.cpu().detach().numpy()
        # print('CDs of batchs of missing part: ', errG.cpu().detach().numpy(), ' * 0.01')

        # 计算整体CD值
        # 从input_cropped1中提取非0部分得到np_crop，B*N*3
        b = opt.pnum - opt.crop_point_num
        output_crop = torch.zeros(batch_size, b, 3).cuda()
        batch_index = torch.arange(batch_size).unsqueeze(1).cuda()
        batch_index = batch_index + torch.zeros(1, b).cuda()
        batch_index = batch_index.reshape(-1).to(torch.long)
        distance_index = input_cropped1 != torch.tensor([0, 0, 0]).cuda()
        distance_index = distance_index[:, :, 0] + distance_index[:, :, 1] + distance_index[:, :, 2]
        sequence_index = torch.arange(b).unsqueeze(0).cuda()
        sequence_index = torch.zeros(batch_size, 1).cuda() + sequence_index
        sequence_index = sequence_index.reshape(-1).to(torch.long)
        output_crop[batch_index, sequence_index] = input_cropped1[distance_index]
        crop_fake = torch.cat((output_crop, fake), dim=1)
        # 计算CD值
        real_point = real_point.cuda()
        errG = criterion_PointLoss(torch.squeeze(crop_fake, 1), torch.squeeze(real_point, 1))
        errG_all = errG_all + errG.cpu().detach().numpy()
        # print('CDs of batchs of all parts: ', errG.cpu().detach().numpy(), ' * 0.01')

        if opt.visual or opt.save_txt:
            # 选择batch中第一个可视化和保存，如果需要开启该模式，建议batch_size设置为1
            np_crop = output_crop[0].cpu().detach().numpy()
            np_fake = fake[0].cpu().detach().numpy()  # 512
            np_real = real_center[0].cpu().detach().numpy()  # 512

            np_crop = np.transpose(np_crop, (1, 0))
            np_fake = np.transpose(np_fake, (1, 0))
            np_real = np.transpose(np_real, (1, 0))
            for j in range(0, 3):
                np_crop[j, :] = np_crop[j, :] * length_ratio[0, j]
                np_fake[j, :] = np_fake[j, :] * length_ratio[0, j]
                np_real[j, :] = np_real[j, :] * length_ratio[0, j]
            np_crop = np.transpose(np_crop, (1, 0))
            np_fake = np.transpose(np_fake, (1, 0))
            np_real = np.transpose(np_real, (1, 0))

        # 展示点云
        if opt.visual:
            win_size = [800, 800]
            pc_size = 8.0

            t = time.localtime()
            current_time = time.strftime("%H%M%S", t)

            pc = np.append(np_crop, np_real, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            save_load_view_point(pcd=pcd, filename="o3d.json", win_size=win_size, pc_size=pc_size,
                                 png_name=current_time + str(i) + '_' + 'complete.png')

            pc = np_crop
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            save_load_view_point(pcd=pcd, filename="o3d.json", win_size=win_size, pc_size=pc_size, png_name=current_time + str(i) + '_' + 'incomplete.png')

            # pc = np_fake
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pc)
            # save_load_view_point(pcd=pcd, filename="o3d.json", win_size=win_size, pc_size=pc_size, png_name= current_time + str(i)+'_'+'fake.png')

            pc = np.append(np_crop, np_fake, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            save_load_view_point(pcd=pcd, filename="o3d.json", win_size=win_size, pc_size=pc_size, png_name=current_time + str(i) + '_' + 'predict.png')



        if opt.save_txt:
            np.savetxt(opt.saveroot + '/crop/'+str(n)+'.txt', np_crop, fmt = "%f %f %f")
            np.savetxt(opt.saveroot + '/fake/'+str(n)+'.txt', np_fake, fmt = "%f %f %f")
            np.savetxt(opt.saveroot + '/real/'+str(n)+'.txt', np_real, fmt = "%f %f %f")

    errG_all_missing = errG_all_missing / num_iter
    errG_all = errG_all / num_iter
    print('mean CDs of all missing parts: ', errG_all_missing, ' *0.01')
    print('mean CDs of all: ', errG_all, ' *0.01')
