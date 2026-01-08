_base_ = [
    './base.py', './model_dinov2-b14_base.py',
]

max_epochs = -1

dataset_type = 'PFWillowDataset'
data_root = './data'
batch_size = 16
num_workers = 16
target_size = (448, 448)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        target_size=target_size,
    )
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
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
        target_size=target_size,
    )
)

# config schedules
optimizer = dict(type='Adam', lr=1e-3)
metric = dict(type='CorrespondenceMetric', alpha=0.1, img_size=target_size)