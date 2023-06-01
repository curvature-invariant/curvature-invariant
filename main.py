# -*- coding: utf-8 -*-

import os
import argparse
import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ShapeNetDataLoader import PartNormalDataset
from torch.utils.data import DataLoader, TensorDataset


from utils.logging import Logging_str
from utils.utils import set_seed

from attacks import PointCloudAttack
from utils.set_distance import ChamferDistance, HausdorffDistance, CurvatureDistance



def load_data(args):
    print('Start Loading Dataset...')
    if args.dataset == 'ModelNet40':
        TEST_DATASET = ModelNetDataLoader(
            root=args.data_path,
            npoint=args.input_point_nums,
            split='test',
            normal_channel=True
        )
    elif args.dataset == 'ShapeNetPart':
        TEST_DATASET = PartNormalDataset(
            root=args.data_path,
            npoints=args.input_point_nums,
            split='test',
            normal_channel=True
        )
    else:
        raise NotImplementedError

    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print('Finish Loading Dataset...')
    return testDataLoader



def data_preprocess(data):
    points, target = data

    points = points
    target = target[:, 0]

    points = points.cuda()
    target = target.cuda()

    return points, target


def save_tensor_as_txt(points, filename):
    points = points.squeeze(0).detach().cpu().numpy()
    with open(filename, "a") as file_object:
        for i in range(points.shape[0]):
            msg = str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2]) + \
                ' ' + str(points[i][3].item()) +' ' + str(points[i][3].item()) + ' '+ str(1-points[i][3].item())
            file_object.write(msg+'\n')
        file_object.close()
    print('Have saved the tensor into {}'.format(filename))


def main():
    test_loader = load_data(args)

    num_class = 0
    if args.dataset == 'ModelNet40':
        num_class = 40
    elif args.dataset == 'ShapeNetPart':
        num_class = 16
    assert num_class != 0
    args.num_class = num_class

    attack = PointCloudAttack(args)

    atk_success = 0
    avg_query_costs = 0.
    avg_mse_dist = 0.
    avg_chamfer_dist = 0.
    avg_hausdorff_dist = 0.
    avg_curvature_dist = 0.
    avg_time_cost = 0.
    chamfer_loss = ChamferDistance()
    hausdorff_loss = HausdorffDistance()
    curvature_loss = CurvatureDistance()
    for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        points, target = data_preprocess(data)
        target = target.long()
        t0 = time.clock()
        adv_points, adv_target, query_costs = attack.run(points, target)
        t1 = time.clock()
        avg_time_cost += t1 - t0
        atk_success += 1 if adv_target != target else 0

        points = points[:,:,:3].data
        pert_pos = torch.where(abs(adv_points-points).sum(2))
        count_map = torch.zeros_like(points.sum(2))
        count_map[pert_pos] = 1.

        avg_mse_dist += np.sqrt(F.mse_loss(adv_points, points).detach().cpu().numpy() * 3072)
        avg_chamfer_dist += chamfer_loss(adv_points, points)
        avg_hausdorff_dist += hausdorff_loss(adv_points, points)
        avg_curvature_dist += curvature_loss(adv_points, points)
        
    print(atk_success)
    atk_success /= batch_id + 1
    print('Attack success rate: ', atk_success)
    avg_time_cost /= batch_id + 1
    print('Average time cost: ', avg_time_cost)
    avg_mse_dist /= batch_id + 1
    print('Average MSE Dist:', avg_mse_dist)
    avg_chamfer_dist /= batch_id + 1
    print('Average Chamfer Dist:', avg_chamfer_dist.item())
    avg_hausdorff_dist /= batch_id + 1
    print('Average Hausdorff Dist:', avg_hausdorff_dist.item())
    avg_curvature_dist /= batch_id + 1
    print('Average Curvature Dist:', avg_curvature_dist.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Curvature-invariant Method')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', 
                        help='input batch size for training (default: 1)')
    parser.add_argument('--input_point_nums', type=int, default=1024,
                        help='Point nums of each point cloud')
    parser.add_argument('--seed', type=int, default=2022, metavar='S',
                        help='random seed (default: 2022)')
    parser.add_argument('--dataset', type=str, default='ModelNet40',
                        choices=['ModelNet40', 'ShapeNetPart'])
    parser.add_argument('--data_path', type=str, 
                        default='../data/modelnet40_normal_resampled/')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Worker nums of data loading.')

    parser.add_argument('--attack_method', type=str, default=None,
                        choices=['curvature'])
    parser.add_argument('--surrogate_model', type=str, default='pointnet_cls',
                        choices=['pointnet_cls', 'pointnet2_cls_msg', 'dgcnn', 'pointconv'])
    parser.add_argument('--target_model', type=str, default='pointnet_cls',
                        choices=['pointnet_cls', 'pointnet2_cls_msg', 'dgcnn', 'pointconv'])
    parser.add_argument('--defense_method', type=str, default=None,
                        choices=['sor', 'srs', 'dupnet'])
    parser.add_argument('--top5_attack', action='store_true', default=False,
                        help='Whether to attack the top-5 prediction [default: False]')

    parser.add_argument('--max_steps', default=5, type=int,
                        help='max iterations for black-box attack')
    parser.add_argument('--eps', default=0.16, type=float,
                        help='epsilon of perturbation')
    parser.add_argument('--step_size', default=0.07, type=float,
                        help='step-size of perturbation')
    args = parser.parse_args()

    set_seed(args.seed)
    args.device = torch.device("cuda")

    print("start")
    main()