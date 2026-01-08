import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

def mutual_nn_filter(correlation_matrix):
    r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
    '''
    correlation_matrix: B x N x N
    '''
    corr_src_max = torch.max(correlation_matrix, dim=-1, keepdim=True)[0]
    corr_trg_max = torch.max(correlation_matrix, dim=-2, keepdim=True)[0]
    corr_src_max[corr_src_max == 0] += 1e-30
    corr_trg_max[corr_trg_max == 0] += 1e-30
    corr_src = correlation_matrix / corr_src_max
    corr_trg = correlation_matrix / corr_trg_max
    return correlation_matrix * (corr_src * corr_trg)


class SCTask:
    def __init__(self, img_size, down_factor) -> None:
        self.img_size = img_size
        self.down_factor = down_factor
        self.feat_size = self.img_size // self.down_factor

    def compute_loss(self):
        pass

    def compute_trg_kps(self):
        pass


class FlowTask(SCTask):
    def __init__(self, img_size, down_factor, receptive_field_size) -> None:
        super().__init__(img_size, down_factor)

        self.box, self.feat_ids = self.receptive_fields(receptive_field_size, self.down_factor, self.feat_size)
        self.x_normal = torch.linspace(-1, 1, self.feat_size)
        self.y_normal = torch.linspace(-1, 1, self.feat_size)
        self.grid = torch.stack(list(reversed(torch.meshgrid(self.x_normal, self.y_normal, indexing='ij')))).permute(1, 2, 0).cuda()

    def receptive_fields(self, receptive_field_size, jsz, feat_size):
        r"""Returns a set of receptive fields (N, 4)"""
        '''
        receptive_field_size: 35
        jsz: down_factor: 14
        feat_size: 32 
        '''
        width = feat_size
        height = feat_size

        feat_ids = torch.tensor(list(range(width))).repeat(1, height).t().repeat(1, 2)
        feat_ids[:, 0] = torch.tensor(list(range(height))).unsqueeze(1).repeat(1, width).view(-1)

        box = torch.zeros(feat_ids.size()[0], 4)
        box[:, 0] = feat_ids[:, 1] * jsz - receptive_field_size // 2 + jsz // 2
        box[:, 1] = feat_ids[:, 0] * jsz - receptive_field_size // 2 + jsz // 2
        box[:, 2] = feat_ids[:, 1] * jsz + receptive_field_size // 2 + jsz // 2
        box[:, 3] = feat_ids[:, 0] * jsz + receptive_field_size // 2 + jsz // 2

        return box.cuda(), feat_ids.cuda()

    def neighbours(self, box, kps):
        r"""Returns boxes in one-hot format that covers given keypoints"""
        box_duplicate = box.unsqueeze(2).repeat(1, 1, len(kps.t())).transpose(0, 1)
        kps_duplicate = kps.unsqueeze(1).repeat(1, len(box), 1)

        xmin = kps_duplicate[0].ge(box_duplicate[0])
        ymin = kps_duplicate[1].ge(box_duplicate[1])
        xmax = kps_duplicate[0].le(box_duplicate[2])
        ymax = kps_duplicate[1].le(box_duplicate[3])

        nbr_onehot = torch.mul(torch.mul(xmin, ymin), torch.mul(xmax, ymax)).t()  # k x 1024
        n_neighbours = nbr_onehot.sum(dim=1)
        n_points = nbr_onehot.sum(dim=0)

        return nbr_onehot, n_neighbours, n_points

    def kp_to_flow(self, src_kps, trg_kps, n_pts):
        src_kp = src_kps.narrow_copy(0, 0, n_pts)
        trg_kp = trg_kps.narrow_copy(0, 0, n_pts)

        src_nbr_onehot, n_neighbours, n_points = self.neighbours(self.box, src_kp.t())  # src_nbr_onehot: 17x256, n_points=256

        center = torch.stack(((self.box[:, 0] + self.box[:, 2])/2, (self.box[:, 1] + self.box[:, 3])/2), dim=1)
        center = center.unsqueeze(0).repeat(len(src_kp), 1, 1)

        src_idx = src_nbr_onehot.nonzero()

        src_nn = center[src_idx[:,0],src_idx[:,1]]
        kp_selected = src_kp[src_idx[:,0],:]

        vector_summator = torch.zeros_like(center)
        vector_summator[src_idx[:, 0], src_idx[:, 1]] = kp_selected

        n_points_expanded = n_points.unsqueeze(1).repeat(1,2).float()
        n_points_expanded[n_points_expanded == 0] = 1.

        source_averaged = (vector_summator.sum(dim=0) / n_points_expanded)[src_idx[:,1]]

        flow = trg_kp[src_idx[:,0],:] - source_averaged

        flow_index = self.feat_ids.index_select(dim=0, index=src_idx[:,1])

        flow_map = torch.zeros(self.feat_size, self.feat_size, 2).cuda()
        flow_map[flow_index[:,0],flow_index[:,1]] = flow / (self.img_size // self.feat_size)

        flow_map = flow_map.permute(2, 0, 1)
        
        return flow_map
    
    def kps_to_flow(self, batch):
        flows = []
        for src_kps, trg_kps, n_pts in zip(batch['src_kps'], batch['trg_kps'], batch['n_pts']):
            flow = self.kp_to_flow(src_kps.t(), trg_kps.t(), n_pts)
            flows.append(flow)
        return torch.stack(flows, dim=0)

    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum
    
    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        '''
        corr: B x N x H x W  trg x src
        '''
        device = corr.device
        b,_,h,w = corr.size()
        
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b, w).to(device)
        x_normal = x_normal.view(b, w, 1, 1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,h).to(device)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y
    
    def unnormalise_and_convert_mapping_to_flow(self, map):
        # here map is normalised to -1;1
        # we put it back to 0,W-1, then convert it to flow
        B, C, H, W = map.size()
        mapping = torch.zeros_like(map)
        # mesh grid
        mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
        mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise

        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if mapping.is_cuda:
            grid = grid.cuda()
        flow = mapping - grid
        return flow
    
    def flow2kps(self, flow, src_kps, n_pts, upsample_size):
        _, _, h, w = flow.size()
        flow = F.interpolate(flow, upsample_size, mode='bilinear') * (upsample_size[0] / h)
        
        trg_kps = []
        for src_kps, flow, n_pts in zip(src_kps.long(), flow, n_pts):
            size = src_kps.size(1)

            kp = torch.clamp(src_kps.narrow_copy(1, 0, n_pts), 0, upsample_size[0] - 1)
            estimated_kps = kp + flow[:, kp[1, :], kp[0, :]]
            estimated_kps = torch.cat((estimated_kps, torch.ones(2, size - n_pts).cuda() * -1), dim=1)
            trg_kps.append(estimated_kps)

        return torch.stack(trg_kps)

    def EPE(self, input_flow, target_flow, sparse=True, mean=True, sum=False, mask=None):
        EPE_map = torch.norm(target_flow-input_flow, 2, 1)
        if sparse:
            # invalid flow is defined with both flow coordinates to be exactly 0
            if mask is not None:
                EPE_map = EPE_map[mask]
            else:
                mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
                EPE_map = EPE_map[~mask]
        if mean:
            loss = EPE_map.mean()
            if torch.isnan(loss):
                return torch.tensor(0.).to(loss.device) 
            else:
                return loss
        elif sum:
            return EPE_map.sum()
        else:
            return EPE_map.sum()/torch.sum(~mask)

    def compute_pseudo_label_mask(self, pred_flow, kernel=3, threshold=0.1):
        '''
        flow: B x 2 x 32 x 32
        '''
        B, _, h, w = pred_flow.shape
        patches = F.unfold(pred_flow, kernel_size=kernel)
        patches = patches.reshape(B, 2, (kernel * kernel), -1)

        std = (patches - patches.mean(dim=2, keepdim=True)).pow(2).mean(dim=2).mean(dim=1)
        std = std.reshape(B, h-kernel+1, w-kernel+1)
        mask = std < threshold
        nonzero_indices = torch.nonzero(mask)
        arr = []
        for i in range(kernel):
            for j in range(kernel):
                indices = nonzero_indices.clone()
                indices[:, 1] = indices[:, 1] + i 
                indices[:, 2] = indices[:, 2] + j
                arr.append(indices)
        nonzero_indices = torch.cat(arr, dim=0)
        new_mask = torch.ones(size=[B, h, w]).to(pred_flow.device) > 99  # to Boolean
        new_mask[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]] = True
        return new_mask

    def compute_flow(self, corr):
        B = corr.shape[0]
        corr = mutual_nn_filter(corr).transpose(-2, -1)
        grid_x, grid_y = self.soft_argmax(corr.view(B, -1, self.feat_size, self.feat_size))  # B x N x 16 x 16  src x trg
        flow = torch.cat((grid_x, grid_y), dim=1)
        flow = self.unnormalise_and_convert_mapping_to_flow(flow)
        return flow

    def compute_loss(self, corr, batch):
        flow = self.compute_flow(corr)
        gt_flow = self.kps_to_flow(batch)
        loss = self.EPE(flow, gt_flow)
        return loss
    
    def compute_trg_kps(self, corr, batch):
        flow = self.compute_flow(corr)
        pred_trg_kps = self.flow2kps(flow, batch['src_kps'], batch['n_pts'], upsample_size=(self.img_size, self.img_size))
        return pred_trg_kps




