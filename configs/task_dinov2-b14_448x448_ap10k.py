_base_ = [
    './base.py', './model_dinov2-b14_base.py',
]

dataset_type = 'AP10kDataset'
data_root = './data'
batch_size = 6
num_workers = 6
target_size = (448, 448)  
data_rate=1.0

max_epochs=10

eval_type = 'intra-species'  # `intra-species`, `cross-species`, `cross-family`

model = dict(
    type='CorrespondenceModel',
    img_size=target_size[0],
    num_prototypes=60,
    using_crop=False
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        eval_type='intra-species', 
        split='trn',
        target_size=target_size, 
        data_rate=data_rate
    )
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        eval_type=eval_type, 
        split='val',
        target_size=target_size, 
    )
)
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        eval_type=eval_type, 
        split='test',
        target_size=target_size,
    )
)

optimizer = dict(type='Adam', lr=1e-4)
metric = dict(type='CorrespondenceMetric', alpha=0.1, img_size=target_size)