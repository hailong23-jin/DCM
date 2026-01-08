import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import loralib as lora
from PIL import Image
import numpy as np 
from collections import OrderedDict
from src.utils import cosine_similarity_BNC
import math

from ..backbones.resnet import BasicBlock, Bottleneck
from mmengine.registry import MODELS
from .task import *


@MODELS.register_module()
class CorrespondenceModel(nn.Module):
    def __init__(self, pair_image_encoder_cfg, img_size=448, down_factor=14, num_prototypes=20, tau=0.1, t=0.96, gamma=0.99, using_crop=True) -> None:
        super().__init__()
        '''
        tau: threshold for neighborhood shift consistency mask
        t: threshold for semantic consistency mask
        gamma: class prototype update hyperparameter
        '''
        self.down_factor = down_factor
        self.using_crop = using_crop
        self.tau = tau
        self.t = t
        self.gamma = gamma
        
        self.pair_image_encoder = MODELS.build(pair_image_encoder_cfg)
        self.backbone_type = pair_image_encoder_cfg['backbone_cfg']['type']
        lora.mark_only_lora_as_trainable(self.pair_image_encoder)
        num_features = self.pair_image_encoder.image_encoder.num_features
        self.res_refine = nn.Sequential(
            BasicBlock(num_features, num_features, stride=1),
            BasicBlock(num_features, num_features, stride=1),
        )

        self.projections = nn.Linear(num_features, num_features)
        self.task = FlowTask(img_size=img_size, down_factor=down_factor, receptive_field_size=35)
        self.cls_prototypes = nn.Parameter(torch.zeros(size=[num_prototypes, num_features]))
        self.idx = nn.Parameter(torch.zeros(size=[num_prototypes]), requires_grad=False)

    def state_dict(self):
        ckpt1 = lora.lora_state_dict(self.pair_image_encoder)
        ckpt2 = self.res_refine.state_dict()
        ckpt3 = self.projections.state_dict()
        ckpt = dict()
        for k, v in ckpt1.items():
            ckpt[f'pair_image_encoder.{k}'] = v

        for k, v in ckpt2.items():
            ckpt[f'res_refine.{k}'] = v

        for k, v in ckpt3.items():
            ckpt[f'projections.{k}'] = v

        ckpt['cls_prototypes'] = self.cls_prototypes.data
        ckpt['idx'] = self.idx.data

        return ckpt

    def update_prototype(self, src_feat, src_kps, n_pts, category_ids):
        B, N, C = src_feat.shape
        W = H = int(math.sqrt(N))
        src_kps = src_kps.clone() / self.down_factor
        for feat, kps, num, cls_id in zip(src_feat, src_kps, n_pts, category_ids):
            kps = kps[:, :num].long()  # k x 2
            kp_index = kps[0, :] + kps[1, :] * W
            prototype = feat[kp_index].mean(dim=0)
            self.cls_prototypes[cls_id] = self.cls_prototypes[cls_id] * self.gamma + prototype * (1 - self.gamma)
            self.idx[cls_id] += 1

    def forward(self, src_img, trg_img, src_kps=None, n_pts=None, category_ids=None):
        # extract features
        src_feats, trg_feats = self.pair_image_encoder(src_img, trg_img)
        src_feat, trg_feat = src_feats[-1], trg_feats[-1]

        B, N, C = src_feat.shape
        W = H = int(math.sqrt(N))
    
        src_feat = src_feat.transpose(-2, -1).reshape(B, C, H, W)
        trg_feat = trg_feat.transpose(-2, -1).reshape(B, C, H, W)
        src_feat = self.res_refine(src_feat).flatten(-2).transpose(-2, -1)
        trg_feat = self.res_refine(trg_feat).flatten(-2).transpose(-2, -1)

        # update class prototype
        if src_kps is not None and self.training:
            with torch.no_grad():
                self.update_prototype(src_feat.clone().detach(), src_kps, n_pts, category_ids)

        # generate object map for filtering out background
        object_mask = torch.ones(size=[B, W, H]).to(src_feat.device) > 0
        if src_kps is None and category_ids is not None:
            with torch.no_grad():
                prototypes = self.cls_prototypes[category_ids].unsqueeze(1)
                object_map = cosine_similarity_BNC(src_feat, prototypes)  # B x N x 1
                object_map = object_map.reshape(B, W, H)
                object_mask = object_map > 0.5

        src_feat = self.projections(src_feat)
        trg_feat = self.projections(trg_feat)

        corr = cosine_similarity_BNC(src_feat, trg_feat)

        with torch.no_grad():
            semantic_mask = corr.max(dim=-1)[0].reshape(B, H, W) > self.t
            coor_mask = corr.max(dim=-1)[1].reshape(B, H, W)

        return corr, object_mask, semantic_mask, coor_mask
    
    def forward_only_flow(self, src_img, trg_img, category_ids=None):
        corr, object_mask, semantic_mask, coor_mask = self.forward(src_img, trg_img, category_ids=category_ids)
        flow = self.task.compute_flow(corr)
        mask = self.task.compute_pseudo_label_mask(flow, kernel=3, threshold=self.tau)
        mask = mask & semantic_mask & object_mask
        return OrderedDict(flow=flow, mask=mask)

    def forward_step(self, batch):
        if self.training:
            corr, _, _, _ = self.forward(batch['src_img'], batch['trg_img'], batch['src_kps'], batch['n_pts'], batch['category_id'])  # B x N x N , src x trg
        else: 
            corr, object_mask, semantic_mask, coor_mask = self.forward(batch['src_img'], batch['trg_img'])
        loss = self.task.compute_loss(corr, batch)
        pred_trg_kps = self.task.compute_trg_kps(corr, batch)
        pred_flow = self.task.compute_flow(corr)

        # for visualization
        # flow = self.task.compute_flow(corr)
        # mask = self.task.compute_pseudo_label_mask(flow, kernel=2, threshold=0.05)
        # mask = mask & semantic_mask #& object_mask
        # src_kps, trg_kps, trg_mask = self.flow_to_kps(flow)
        # for i in range(src_kps.shape[0]):
        #     visualize_flow2(src_kps, trg_kps, mask, batch['src_img'], batch['trg_img'], batch['pair_name'], i)

        return OrderedDict(flow=pred_flow, pred_trg_kps=pred_trg_kps, total_loss=loss) 

    def compute_pseudo_loss(self, pred_flow, pseudo_flow, mask):
        loss = self.task.EPE(pred_flow, pseudo_flow, mask=mask)
        return loss

    def inference(self, batch, visualize=False):
        out = self.forward_step(batch)
        pred_trg_kps = out['pred_trg_kps']
        crop_size = 256
        img_size = batch['src_img'].shape[-1]
        if self.using_crop:
            for i in range(pred_trg_kps.shape[0]):
                trg_h, trg_w = batch['trg_imsize'][i]
                tmp_kps = pred_trg_kps[i, :, :batch['n_pts'][i]].clone()
                tmp_kps[0, :] = (tmp_kps[0, :] - batch['trg_left_pad'][i]) / batch['trg_scale'][i]
                tmp_kps[1, :] = (tmp_kps[1, :] - batch['trg_top_pad'][i]) / batch['trg_scale'][i]

                trg_pil = Image.fromarray(batch['org_trg_img'][i].cpu().numpy()[:trg_h, :trg_w, :])
                trg_pil, trg_kps, shift_x, shift_y, crop = process_imagev2(trg_pil, tmp_kps.clone(), crop_size=crop_size)
                if crop:
                    trg_img = transform(trg_pil, img_size)
                    trg_img = trg_img.unsqueeze(0)
                    src_img = batch['src_img'][i:(i+1), :, :, :].clone()
                    min_batch = {
                        'src_img': src_img,
                        'trg_img': trg_img,
                        'src_kps': batch['src_kps'][i:(i+1)],
                        'trg_kps': batch['trg_kps'][i:(i+1)],
                        'n_pts': batch['n_pts'][i:(i+1)],
                        'pair_name': batch['pair_name'][i:(i+1)],
                        # 'category_id': batch['category_id'][i:(i+1)]
                    }
                    min_out = self.forward_step(min_batch)
                    pred_kp = min_out['pred_trg_kps'] * (crop_size / img_size)
                    pred_kp[:, 0, :] = (pred_kp[:, 0, :] + shift_x) * batch['trg_scale'][i] + batch['trg_left_pad'][i]
                    pred_kp[:, 1, :] = (pred_kp[:, 1, :] + shift_y) * batch['trg_scale'][i] + batch['trg_top_pad'][i]
                    pred_trg_kps[i, :, :] = pred_kp[0]

        out['pred_trg_kps'] = pred_trg_kps
        # for i in range(pred_trg_kps.shape[0]):
        #     visualize_gt(batch, i)
        return out
    

def process_imagev2(img, kps, crop_size):
    box = get_bounding_box(kps)
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    max_side = max(box_width, box_height)
    if max_side >= crop_size:
        return img, kps, 0, 0, False

    img, kps, x1, y1 = center_crop(img, kps, (crop_size, crop_size))
    return img, kps, x1, y1, True

def transform(image, img_size):
    image = TF.resize(image, size=(img_size, img_size))
    mean= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)  
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = np.array(image).astype(float).transpose(2, 0, 1) / 255.0
    image = (image - mean) / std
    image = torch.tensor(image).float().cuda()  # to GPU
    return image

def get_bounding_box(kps):
    x_min = kps[0, :].min()
    x_max = kps[0, :].max()
    y_min = kps[1, :].min()
    y_max = kps[1, :].max()
    return [x_min, y_min, x_max, y_max]

def center_crop(img, kps, crop_size):
    '''
    crop_size: h x w
    '''
    img_size = img.size  # w x h
    center = get_center_point(kps)

    offset_x = crop_size[1] // 2
    offset_y = crop_size[0] // 2
    x1, y1 = center[0] - offset_x, center[1] - offset_y
    x2, y2 = center[0] + offset_x, center[1] + offset_y
    if x2 > img_size[0]:
        x1 = x1 - (x2 - img_size[0])
        x2 = img_size[0]
    if y2 > img_size[1]:
        y1 = y1 - (y2 - img_size[1])
        y2 = img_size[1]
    if x1 < 0:
        x2 = x2 + abs(x1)
        x1 = 0
    if y1 < 0:
        y2 = y2 + abs(y1)
        y1 = 0

    img = img.crop((x1, y1, x2, y2))
    kps[0, :] = kps[0, :] - x1
    kps[1, :] = kps[1, :] - y1

    return img, kps, x1, y1

def get_center_point(src_kps):
    x = (src_kps[0, :].max() + src_kps[0, :].min()) / 2
    y = (src_kps[1, :].max() + src_kps[1, :].min()) / 2
    return (x.long().item(), y.long().item())

def is_crop(kps, crop_size, threshold):
    kps = kps.clone()
    box = get_bounding_box(kps)

    # judge the crop size contains all keypoints
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    if box_width / crop_size[1] > threshold or box_height / crop_size[0] > threshold:
        return False

    return True
