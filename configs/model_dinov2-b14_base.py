# config model
import os.path as osp
home_directory = osp.expanduser('~')

model = dict(
    type='CorrespondenceModel',
    pair_image_encoder_cfg=dict(
        type='PairImageEncoder',  
        backbone_cfg=dict(
            type='DinoVisionTransformer',
            img_size=518,
            patch_size=14,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            init_values=1.0,
            block_chunks=0,
            checkpoint_path=osp.join(home_directory, '.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth')
        ), 
    ),
    num_prototypes=20,
    tau=0.1, 
    t=0.96, 
    gamma=0.99
)
