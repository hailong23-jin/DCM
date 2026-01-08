r"""SPair-71k dataset"""
import os
import json
import copy
import random
import numpy as np
import os.path as osp

from torch.utils.data import Dataset

from .piplines import Compose, LoadImage, RandomCrop, NormalAug, \
    Normalize, ToTensor, TestTransform, PadKeyPoints, StrongAug, \
        Resize, ResizeTransform, RandomRotation

from mmengine import DATASETS

def read_from_json(ann_file):
    data = json.load(open(ann_file))
    src_kps = np.array(data['src_kps']).T
    trg_kps = np.array(data['trg_kps']).T
    n_pts = src_kps.shape[-1]
    src_bbox = np.array(data['src_bndbox']).astype(np.float32)
    trg_bbox = np.array(data['trg_bndbox']).astype(np.float32)
    kps_ids = np.array(data['kps_ids']).astype(int)
    return src_kps, trg_kps, n_pts, src_bbox, trg_bbox, kps_ids


@DATASETS.register_module()
class SPairDataset(Dataset):
    def __init__(self, data_root, split, target_size, data_rate=1.0, demo_sample=-1):
        super(SPairDataset, self).__init__()
        self.demo_sample = demo_sample
        self.split = split
        dataset_dir = osp.join(data_root, 'SPair-71k')
        self.data_path = osp.join(dataset_dir, 'Layout/large', split + '.txt')  # split
        self.img_dir = osp.join(dataset_dir, 'JPEGImages')
        self.ann_dir = osp.join(dataset_dir, 'PairAnnotation', split) # split
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                        'car', 'cat', 'chair', 'cow', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']

        data_list, data_dict = self.load_data_list()
        if split == 'trn':
            # construct unlabeled and labeled data from training dataset
            self.unlabeled_data = self.construct_unlabeled_data(data_list)
            self.data_list = self.construct_labeled_data(data_dict, alpha=data_rate)
        else:
            # all data is used to test
            self.data_list = data_list

        self.trn_transform = Compose([
            LoadImage(),
            RandomCrop(target_size=target_size),
            # RandomRotation(),  # for further improvement
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

    def __len__(self):
        if self.demo_sample > 0:
            return self.demo_sample
        else:
            return len(self.data_list)

    def load_data_list(self):
        split_data = open(self.data_path).read().split('\n')
        split_data = split_data[:len(split_data) - 1]

        data_list = []
        data_dict = dict()
        for filename in split_data:  # 000009-2008_001546-2009_000327:aeroplane
            pair_name, category = filename.split(':')
            _, src_name, trg_name = pair_name.split('-')
            ann_file = osp.join(self.ann_dir, filename + '.json')  # # 000009-2008_001546-2009_000327:aeroplane.json
            src_kps, trg_kps, n_pts, src_bbox, trg_bbox, kps_ids = read_from_json(ann_file)
            sample = {
                'category': category,
                'category_id': self.classes.index(category),
                'src_name': src_name,
                'trg_name': trg_name,
                'src_img_path': osp.join(self.img_dir, category, src_name + '.jpg'),
                'trg_img_path': osp.join(self.img_dir, category, trg_name + '.jpg'),
                'pair_name': f'{src_name}-{trg_name}:{category}',
                'src_kps': src_kps,
                'trg_kps': trg_kps,
                'n_pts': n_pts,
                'src_bbox': src_bbox,
                'trg_bbox': trg_bbox,
                'kps_ids': kps_ids
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

        # remove duplicate paths
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

    def __getitem__(self, idx):
        batch = copy.deepcopy(self.data_list[idx])

        if self.split == 'trn':
            batch = self.trn_transform(batch)
        else:
            batch = self.tst_transform(batch)

        batch['pckthres'] = self.get_pckthres(batch['trg_bbox'])

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

    def get_pckthres(self, bbox):
        # threshold: PCK@bbox
        bbox_w = (bbox[2] - bbox[0])
        bbox_h = (bbox[3] - bbox[1])
        pckthres = max(bbox_w, bbox_h)
        return pckthres

