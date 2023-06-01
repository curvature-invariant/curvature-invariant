import os
from pickle import FALSE
import sys
import numpy as np
from collections import Iterable
import importlib
import open3d as o3d

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from utils.set_distance import ChamferDistance, HausdorffDistance
from pytorch3d.ops.points_normals import estimate_pointcloud_local_coord_frames
from pytorch3d.ops import knn_points, knn_gather

from baselines import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model/classifier'))


class PointCloudAttack(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device

        self.eps = args.eps
        self.normal = args.normal
        self.step_size = args.step_size
        self.num_class = args.num_class
        self.max_steps = args.max_steps
        self.top5_attack = args.top5_attack
        self.attack_method = args.attack_method

        self.build_models()
        self.defense_method = args.defense_method
        if not args.defense_method is None:
            self.pre_head = self.get_defense_head(args.defense_method)


    def build_models(self):
        MODEL = importlib.import_module(self.args.surrogate_model)
        wb_classifier = MODEL.get_model(
            self.num_class,
            normal_channel=self.normal
        )
        wb_classifier = wb_classifier.to(self.device)
        MODEL = importlib.import_module(self.args.target_model)
        classifier = MODEL.get_model(
            self.num_class,
            normal_channel=self.normal
        )
        classifier = classifier.to(self.args.device)
        wb_classifier = self.load_models(wb_classifier, self.args.surrogate_model)
        classifier = self.load_models(classifier, self.args.target_model)
        self.wb_classifier = wb_classifier.eval()
        self.classifier = classifier.eval()


    def load_models(self, classifier, model_name):
        model_path = os.path.join('../checkpoint/' + self.args.dataset, model_name)
        if os.path.exists(model_path + '.pth'):
            checkpoint = torch.load(model_path + '.pth')
        elif os.path.exists(model_path + '.t7'):
            checkpoint = torch.load(model_path + '.t7')
        elif os.path.exists(model_path + '.tar'):
            checkpoint = torch.load(model_path + '.tar')
        else:
            raise NotImplementedError

        try:
            if 'model_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state_dict'])
            elif 'model_state' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state'])
            else:
                classifier.load_state_dict(checkpoint)
        except:
            classifier = nn.DataParallel(classifier)
            classifier.load_state_dict(checkpoint)
        return classifier

    def CWLoss(self, logits, target, kappa=0, tar=False, num_classes=40):
        target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
        target_one_hot = Variable(torch.eye(num_classes).type(torch.cuda.FloatTensor)[target.long()].cuda())

        real = torch.sum(target_one_hot*logits, 1)
        if not self.top5_attack:
            other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
        else:
            other = torch.topk((1-target_one_hot)*logits - (target_one_hot*10000), 5)[0][:, 4]
        kappa = torch.zeros_like(other).fill_(kappa)

        if tar:
            return torch.sum(torch.max(other-real, kappa))
        else :
            return torch.sum(torch.max(real-other, kappa))


    def run(self, points, target):
        if self.attack_method == 'curvature':
            return self.curvature_invariant_ifgm(points, target)
        else:
            NotImplementedError


    def get_defense_head(self, method):
        if method == 'sor':
            pre_head = SORDefense(k=2, alpha=1.1)
        elif method == 'srs':
            pre_head = SRSDefense(drop_num=500)
        elif method == 'dupnet':
            pre_head = DUPNet(sor_k=2, sor_alpha=1.1, npoint=1024, up_ratio=4)
        else:
            raise NotImplementedError
        return pre_head

    
    def get_curvature_vector(self, points):
        curvatures, local_coord_frames = estimate_pointcloud_local_coord_frames(points, neighborhood_size=20)
        normal_vec = local_coord_frames[:, :, :, 0]
        normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True))
        
        return normal_vec, curvatures, local_coord_frames[:, :, :, 1:]


    def get_spin_axis_matrix(self, normal_vec):
        _, N, _ = normal_vec.shape
        x = normal_vec[:,:,0]
        y = normal_vec[:,:,1]
        z = normal_vec[:,:,2]
        u = torch.zeros(1, N, 3, 3).cuda()
        denominator = torch.sqrt(1-z**2)
        u[:,:,0,0] = y / denominator
        u[:,:,0,1] = - x / denominator
        u[:,:,0,2] = 0.
        u[:,:,1,0] = x * z / denominator
        u[:,:,1,1] = y * z / denominator
        u[:,:,1,2] = - denominator
        u[:,:,2] = normal_vec
        pos = torch.where(abs(z ** 2 - 1) < 1e-4)[1]
        u[:,pos,0,0] = 1 / np.sqrt(2)
        u[:,pos,0,1] = - 1 / np.sqrt(2)
        u[:,pos,0,2] = 0.
        u[:,pos,1,0] = z[:,pos] / np.sqrt(2)
        u[:,pos,1,1] = z[:,pos] / np.sqrt(2)
        u[:,pos,1,2] = 0.
        u[:,pos,2,0] = 0.
        u[:,pos,2,1] = 0.
        u[:,pos,2,2] = z[:,pos]
        return u.data


    def get_transformed_point_cloud(self, points, normal_vec):
        intercept = torch.mul(points, normal_vec).sum(-1, keepdim=True)
        spin_axis_matrix = self.get_spin_axis_matrix(normal_vec)
        translation_matrix = torch.mul(intercept, normal_vec).data
        new_points = points + translation_matrix
        new_points = new_points.unsqueeze(-1)
        new_points = torch.matmul(spin_axis_matrix, new_points)
        new_points = new_points.squeeze(-1).data
        return new_points, spin_axis_matrix, translation_matrix


    def get_original_point_cloud(self, new_points, spin_axis_matrix, translation_matrix):
        inputs = torch.matmul(spin_axis_matrix.transpose(-1, -2), new_points.unsqueeze(-1))
        inputs = inputs - translation_matrix.unsqueeze(-1)
        inputs = inputs.squeeze(-1)
        return inputs


    
    def curvature_invariant_ifgm(self, points, target):
        normal_vec = points[:,:,-3:].data
        normal_vec = normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True)) # N, [1, N, 3]
        points = points[:,:,:3].data
        ori_points = points.data
        clip_func = ClipPointsLinf(budget=self.eps)

        for i in range(self.max_steps):
            new_points, spin_axis_matrix, translation_matrix = self.get_transformed_point_cloud(points, normal_vec)
            new_points = new_points.detach()
            new_points.requires_grad = True
            points = self.get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
            points = points.transpose(1, 2)
            
            if not self.defense_method is None:
                logits = self.wb_classifier(self.pre_head(points))
            else:
                logits = self.wb_classifier(points)
            loss = self.CWLoss(logits, target, kappa=0., tar=False, num_classes=self.num_class)
            self.wb_classifier.zero_grad()
            loss.backward()
            grad = new_points.grad.data
            if i != 0:
                grad_label = torch.cat([update_label, update_label, update_label], axis = -1)
                grad_2 = torch.mul(grad, directions) * directions
                grad_ = torch.mul(grad, directions1) * directions1
                grad = torch.where(grad_label == 1.0 , grad_, grad_2)
            grad[:,:,2] = torch.where(grad[:,:,2] > 0.0, grad[:,:,2], torch.zeros_like(grad[:,:,2]))
            

            norm = torch.sum(grad ** 2, dim=[1, 2]) ** 0.5
            new_points = new_points - self.step_size * np.sqrt(3*1024) * grad / (norm[:, None, None] + 1e-9)
            points = self.get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
            points = clip_func(points, ori_points)
            
            normal_vec, curvatures, directions = self.get_curvature_vector(points)

            directions1,_,_ = self.get_transformed_point_cloud(directions[:,:,:,0], normal_vec)
            directions2,_,_ = self.get_transformed_point_cloud(directions[:,:,:,1], normal_vec)
            
            curvature_norm = torch.sqrt(torch.sum(curvatures[:,:,1:] ** 2, dim=-1, keepdim=True))
            curvature_ratio = torch.abs(torch.unsqueeze(curvatures[:,:,1], dim=-1))/curvature_norm
            
            update_label = torch.where(torch.abs((curvature_ratio-0.5)) < 0.2, torch.ones_like(curvature_norm), torch.zeros_like(curvature_norm))
            curvatures_ratio = torch.unsqueeze(curvatures[:,:,1]**2 + curvatures[:,:,2]**2, axis=-1)
            directions = directions1 * torch.unsqueeze(curvatures[:,:,2]**2, axis=-1)/curvatures_ratio + directions2 * torch.unsqueeze(curvatures[:,:,1]**2, axis=-1)/curvatures_ratio
            directions = directions/torch.sqrt(torch.sum(directions ** 2, dim=-1, keepdim=True))          

        with torch.no_grad():
            adv_points = points.data
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.transpose(1, 2).detach()))
            else:
                adv_logits = self.classifier(points.transpose(1, 2).detach())
            adv_target = adv_logits.data.max(1)[1]

        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1

        del normal_vec, grad, new_points, spin_axis_matrix, translation_matrix
        return adv_points, adv_target, (adv_logits.data.max(1)[1] != target).sum().item()