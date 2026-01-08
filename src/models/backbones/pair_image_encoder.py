import torch
import torch.nn as nn 

from mmengine.registry import MODELS


@MODELS.register_module()
class PairImageEncoder(nn.Module):
    def __init__(self, backbone_cfg) -> None:
        super().__init__()
        self.image_encoder = MODELS.build(backbone_cfg)

    def forward(self, image1, image2):
        imgs = torch.cat([image1, image2], dim=0)
        img_feats = self.image_encoder.patch_embedding(imgs)

        B, N, C = img_feats.shape
        per_batch = B // 2
        for i, block in enumerate(self.image_encoder.blocks):
            img_feats = block(img_feats)

        return [img_feats[:per_batch, 1:, :]], [img_feats[per_batch:, 1:, :]]

