import copy
import numpy as np
import pandas as pd
import os.path as osp

import torch
from torch.utils.data import Dataset

from mmengine.registry import DATASETS
from mmengine.dataset import Compose

from .piplines import Compose, LoadImage, \
    Normalize, ToTensor, TestTransform, PadKeyPoints

@DATASETS.register_module()
class PFWillowDataset(Dataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, data_root, target_size):
        r"""PF-WILLOW dataset constructor"""
        super(PFWillowDataset, self).__init__()

        dataset_dir = osp.join(data_root, 'PF-WILLOW')
        self.data_path = osp.join(dataset_dir, 'test_pairs.csv')
        self.img_dir = dataset_dir

        self.cls = ['car(G)', 'car(M)', 'car(S)', 'duck(S)',
                    'motorbike(G)', 'motorbike(M)', 'motorbike(S)',
                    'winebottle(M)', 'winebottle(wC)', 'winebottle(woC)']
        
        self.data_list = self.load_data_list()

        self.tst_transform = Compose([
            LoadImage(),
            TestTransform(target_size=target_size, crop_size=256),
            # ResizeTransform(target_size=target_size),
            PadKeyPoints(max_num=40),
            ToTensor(),
            Normalize()
        ])

    def load_data_list(self):
        train_data = pd.read_csv(self.data_path)

        data_list = []
        for _, row in train_data.iterrows():
            src_name = osp.basename(row['imageA'])[:-4]
            trg_name = osp.basename(row['imageB'])[:-4]
            category = row['imageA'].split('/')[1]
            src_kps = torch.tensor(row[2:22].values.reshape(2, 10).astype(np.float32))
            trg_kps = torch.tensor(row[22:].values.reshape(2, 10).astype(np.float32))

            sample = {
                'category': category,
                'category_id': self.cls.index(category),
                'src_name': src_name,
                'trg_name': trg_name,
                'src_img_path': osp.join(self.img_dir, category, src_name + '.png'),
                'trg_img_path': osp.join(self.img_dir, category, trg_name + '.png'),
                'pair_name': f'{src_name}-{trg_name}:{category}',
                'src_kps': src_kps,
                'trg_kps': trg_kps,
                'n_pts': 10,
            }
            data_list.append(sample)

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        batch = copy.deepcopy(self.data_list[idx])
        batch = self.tst_transform(batch)
        batch['pckthres'] = self.get_pckthres(batch)

        return batch

    def get_pckthres(self, batch):
        return max(batch['src_kps'].max(1)[0] - batch['src_kps'][:, :batch['n_pts']].min(1)[0]).clone()
        