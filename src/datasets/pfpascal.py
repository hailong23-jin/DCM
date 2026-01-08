r"""PF-PASCAL dataset"""
import copy
import scipy.io as sio
import pandas as pd
import numpy as np
import os.path as osp
import random
import torch
from torch.utils.data import Dataset

from mmengine import DATASETS
from mmengine.dataset import Compose

from .piplines import Compose, LoadImage, RandomCrop, NormalAug, \
    Normalize, ToTensor, TestTransform, PadKeyPoints, StrongAug, \
        Resize


def read_from_mat(path):  # kps bbox
    data = sio.loadmat(path)

    kps = []
    kps_idx = []
    for idx, kp in enumerate(data['kps']):
        if True in np.isnan(kp):
            continue
        else:
            kps.append(kp)
            kps_idx.append(idx)

    kps = np.stack(kps, axis=1)
    kps_idx = np.array(kps_idx)
    bbox = np.array(data['bbox'][0]).astype(np.float64)

    return kps, kps_idx, bbox


@DATASETS.register_module()
class PFPascalDataset(Dataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, data_root, split, target_size, data_rate=1.0, demo_sample=-1):
        r"""PF-PASCAL dataset constructor"""
        super(PFPascalDataset, self).__init__()
        self.split = split
        self.demo_sample = demo_sample
        dataset_dir = osp.join(data_root, 'PF-PASCAL')
        self.data_path = osp.join(dataset_dir, split + '_pairs.csv')
        self.img_dir = osp.join(dataset_dir, 'JPEGImages')
        self.ann_dir = osp.join(dataset_dir, 'Annotations')
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        data_list, data_dict = self.load_data_list()
        if split == 'trn':
            self.unlabeled_data = self.construct_unlabeled_data(data_list)
            self.data_list = self.construct_labeled_data(data_dict, alpha=data_rate)
        else:
            self.data_list = data_list

        self.trn_transform = Compose([
            LoadImage(),
            RandomCrop(target_size=target_size),
            NormalAug(),
            PadKeyPoints(max_num=40),
            ToTensor(),
            Normalize()
        ])
        self.tst_transform = Compose([
            LoadImage(),
            TestTransform(target_size=target_size, crop_size=256),
            # ResizeTransform(target_size=target_size),
            PadKeyPoints(max_num=40),
            ToTensor(),
            Normalize()
        ])
        self.strong_transform = Compose([
            LoadImage(),
            Resize(target_size=target_size),
            StrongAug(),
            ToTensor(),
            Normalize()
        ])
        self.weak_transform = Compose([
            LoadImage(),
            Resize(target_size=target_size),
            ToTensor(),
            Normalize()
        ])


    def load_data_list(self):
        split_data = pd.read_csv(self.data_path)
        src_img_list = np.array(split_data.iloc[:, 0])
        trg_img_list = np.array(split_data.iloc[:, 1])
        cls_ids = split_data.iloc[:, 2].values.astype('int') - 1
        if self.split == 'trn':
            flip = split_data.iloc[:, 3].values.astype('int')
        else:
            flip = [0] * len(src_img_list)

        data_list = []
        data_dict = dict()
        for src_path, trg_path, category_id, is_flip in zip(src_img_list, trg_img_list, cls_ids, flip):
            category = self.classes[category_id]
            src_name = osp.basename(src_path)[:-4]
            trg_name = osp.basename(trg_path)[:-4]
            src_ann_path = osp.join(self.ann_dir, category, src_name + '.mat')
            trg_ann_path = osp.join(self.ann_dir, category, trg_name + '.mat')
            src_kps, kps_ids, src_bbox = read_from_mat(src_ann_path)
            trg_kps, _, trg_bbox = read_from_mat(trg_ann_path)
            sample = {
                'category': category,
                'category_id': category_id,
                'src_name': src_name,
                'trg_name': trg_name,
                'src_img_path': osp.join(self.img_dir, src_name + '.jpg'),
                'trg_img_path': osp.join(self.img_dir, trg_name + '.jpg'),
                'pair_name': f'{src_name}-{trg_name}:{category}',
                'src_kps': src_kps,
                'trg_kps': trg_kps,
                'kps_ids': kps_ids,
                'n_pts': src_kps.shape[-1],
                'src_bbox': src_bbox,
                'trg_bbox': trg_bbox,
                'is_flip': int(is_flip)
            }
            data_list.append(sample)
            if category not in data_dict:
                data_dict[category] = [sample]
            else:
                data_dict[category] += [sample]

        return data_list, data_dict

    def construct_unlabeled_data(self, data_list):
        data_dict = dict()
        for sample in data_list:
            category = sample['category']
            if category not in data_dict:
                data_dict[category] = [sample['src_img_path'], sample['trg_img_path']]
            else:
                data_dict[category] += [sample['src_img_path'], sample['trg_img_path']]

        for k, v in data_dict.items():
            data_dict[k] = list(set(v))

        return data_dict

    def construct_labeled_data(self, data_dict, alpha):
        random.seed(123)
        data_list = []
        for key, val in data_dict.items():
            num = int(alpha * len(val))
            data_list += random.sample(val, num)
        return data_list

    def get_unlabeled_data(self, category):
        img_list = self.unlabeled_data[category]
        src_path, trg_path = random.sample(img_list, k=2)
        src_name = osp.basename(src_path)[:-4]
        trg_name = osp.basename(trg_path)[:-4]
        batch = {
            'category': category,
            'category_id': self.classes.index(category),
            'src_img_path': src_path,
            'trg_img_path': trg_path,
            'src_name': src_name,
            'trg_name': trg_name,
            'pair_name': f"{src_name}-{trg_name}:{category}"
        }
        return batch

    def __len__(self):
        if self.demo_sample > 0:
            return self.demo_sample
        else:
            return len(self.data_list)

    def __getitem__(self, idx):
        batch = copy.deepcopy(self.data_list[idx])

        if self.split == 'trn':
            batch = self.trn_transform(batch)
        else:
            batch = self.tst_transform(batch)

        batch['pckthres'] = self.get_pckthres(batch['trg_img'])

        # unlabeled data for unsupervised training
        if self.split == 'trn':
            unlabeled_data = self.get_unlabeled_data(batch['category'])
            batch_strong = self.strong_transform(copy.deepcopy(unlabeled_data))
            batch_weak = self.weak_transform(copy.deepcopy(unlabeled_data))
            batch['src_img_strong'] = batch_strong['src_img']
            batch['trg_img_strong'] = batch_strong['trg_img']
            batch['src_img_weak'] = batch_weak['src_img']
            batch['trg_img_weak'] = batch_weak['trg_img']
            batch['unlabeled_cls_id'] = unlabeled_data['category_id']
            batch['unlabeled_pair_name'] = unlabeled_data['pair_name']

        return batch
    
    def get_pckthres(self, trg_img):
        # threshold: PCK@img
        img_size = trg_img.size()
        pckthres = torch.tensor(max(img_size[1], img_size[2]))
        return pckthres