_base_ = [
    './base.py', './model_dinov2-b14_base.py',
]

max_epochs = 10

dataset_type = 'SPairDataset'
data_root = './data'
batch_size = 6
num_workers = 6
target_size = (448, 448)
data_rate = 1.0

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
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
        split='test',
        target_size=target_size,
    )
)

optimizer = dict(type='Adam', lr=1e-3)
metric = dict(type='CorrespondenceMetric', alpha=0.1, img_size=target_size)


