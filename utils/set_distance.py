"""Adopted from https://github.com/XuyangBai/FoldingNet/blob/master/loss.py"""
import torch
import torch.nn as nn
from pytorch3d.ops.points_normals import estimate_pointcloud_local_coord_frames


class _Distance(nn.Module):

    def __init__(self):
        super(_Distance, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        pass

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))  # [B, K, K]
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(
            1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P


class ChamferDistance(_Distance):

    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, preds, gts):
        """
        preds: [B, N1, 3]
        gts: [B, N2, 3]
        """
        P = self.batch_pairwise_dist(gts, preds)  # [B, N2, N1]
        mins, _ = torch.min(P, 1)  # [B, N1], find preds' nearest points in gts
        loss1 = torch.mean(mins, dim=1)  # [B]
        mins, _ = torch.min(P, 2)  # [B, N2], find gts' nearest points in preds
        loss2 = torch.mean(mins, dim=1)  # [B]
        # return loss1, loss2
        # return torch.max(loss1, loss2)
        return (loss1 + loss2) / 2


class HausdorffDistance(_Distance):

    def __init__(self):
        super(HausdorffDistance, self).__init__()

    def forward(self, preds, gts):
        """
        preds: [B, N1, 3]
        gts: [B, N2, 3]
        """
        P = self.batch_pairwise_dist(gts, preds)  # [B, N2, N1]
        # max_{y \in pred} min_{x \in gt}
        mins, _ = torch.min(P, 1)  # [B, N1]
        loss1 = torch.max(mins, dim=1)[0]  # [B]
        # max_{y \in gt} min_{x \in pred}
        mins, _ = torch.min(P, 2)  # [B, N2]
        loss2 = torch.max(mins, dim=1)[0]  # [B]
        # return loss1, loss2
        # return torch.max(loss1, loss2)
        return (loss1 + loss2) / 2


class CurvatureDistance(_Distance):
    def get_curvature_vector(self, points):
        """Calculate the normal vector.
        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 3].
        """
        curvatures, local_coord_frames = estimate_pointcloud_local_coord_frames(points, neighborhood_size=20)
        
        return curvatures[:,:,1:]

    def __init__(self):
        super(CurvatureDistance, self).__init__()

    def forward(self, preds, gts):
        """
        preds: [B, N1, 3]
        gts: [B, N2, 3]
        """
        curvatures_preds = self.get_curvature_vector(preds)
        KG_preds = curvatures_preds[:,:,0] * curvatures_preds[:,:,1]
        curvatures_gts = self.get_curvature_vector(gts)
        KG_gts = curvatures_gts[:,:,0] * curvatures_gts[:,:,1]
        
        return torch.sqrt(torch.sum((KG_preds-KG_gts)**2))
        #torch.mean(torch.sqrt((KG_preds-KG_gts)))


chamfer = ChamferDistance()
hausdorff = HausdorffDistance()
curvature = CurvatureDistance()
