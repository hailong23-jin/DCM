r"""SPair-71k dataset"""
import json
import os
import os.path as osp
import torch
import random
import copy
from glob import glob
import numpy as np
from tqdm import tqdm

from mmengine import DATASETS

from .piplines import Compose, LoadImage, RandomCrop, NormalAug, \
    Normalize, ToTensor, TestTransform, PadKeyPoints, StrongAug, \
        Resize, ResizeTransform


@DATASETS.register_module()
class AP10kDataset(torch.nn.Module):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, data_root, eval_type, split, target_size, data_rate=1.0):
        r"""SPair-71k dataset constructor"""
        super().__init__()
        self.split = split
        self.target_size = target_size
        dataset_dir = osp.join(data_root, 'ap-10k')
        self.img_ann_dir = osp.join(dataset_dir, 'ImageAnnotation')
        self.pair_ann_dir = osp.join(dataset_dir, 'PairAnnotation')
        self.img_dir = osp.join(dataset_dir, 'JPEGImages')

        categories, split = self.get_categories(eval_type, split)
        self.classes = categories
        
        data_list, data_dict = self.load_data_list(categories, split)
        if split == 'trn':
            self.unlabeled_data = self.construct_unlabeled_data(data_list)
            self.data_list = self.construct_labeled_data(data_dict, alpha=data_rate)
        else:
            self.data_list = data_list

        self.trn_transform = Compose([
            LoadImage(using_org_img=False),
            RandomCrop(target_size=target_size, p=0),
            NormalAug(),
            PadKeyPoints(max_num=40),
            ToTensor(),
            Normalize()
        ])
        self.tst_transform = Compose([
            LoadImage(using_org_img=False),
            ResizeTransform(target_size=target_size),
            PadKeyPoints(max_num=40),
            ToTensor(),
            Normalize()
        ])
        self.strong_transform = Compose([
            LoadImage(using_org_img=False),
            Resize(target_size=target_size),
            StrongAug(),
            ToTensor(),
            Normalize()
        ])
        self.weak_transform = Compose([
            LoadImage(using_org_img=False),
            Resize(target_size=target_size),
            ToTensor(),
            Normalize()
        ])

    def get_categories(self, eval_type, split):
        categories = []
        subfolders = os.listdir(self.img_ann_dir)
        # Handle AP10K_EVAL test settings
        if eval_type == 'intra-species':
            categories = [folder for subfolder in subfolders for folder in os.listdir(os.path.join(self.img_ann_dir, subfolder))]
        elif eval_type == 'cross-species':
            categories = [subfolder for subfolder in subfolders if len(os.listdir(os.path.join(self.img_ann_dir, subfolder))) > 1]
            split += '_cross_species'
        elif eval_type == 'cross-family':
            categories = ['all']
            split += '_cross_family'
        categories = sorted(categories)

        return categories, split
    
    def load_data_list(self, categories, split):
        pairs = []
        for category in categories:
            pairs += sorted(glob(f'{self.pair_ann_dir}/{split}/*:{category}.json'))
        data_list = []
        data_dict = dict()
        for pair in tqdm(pairs, ncols=80):
            with open(pair) as f:
                data = json.load(f)
            category = pair.split(':')[1][:-5]
            source_json_path = data["src_json_path"]
            target_json_path = data["trg_json_path"]
            src_img_path = source_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')
            trg_img_path = target_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')

            with open(source_json_path) as f:
                src_file = json.load(f)
            with open(target_json_path) as f:
                trg_file = json.load(f)

            source_bbox = np.asarray(src_file["bbox"]).astype(float)  # l t w h
            target_bbox = np.asarray(trg_file["bbox"]).astype(float)

            source_kps = torch.tensor(src_file["keypoints"]).view(-1, 3).float()
            target_kps = torch.tensor(trg_file["keypoints"]).view(-1, 3).float()
            used_kps, = torch.where(source_kps[:, 2] * target_kps[:, 2]>0)
            source_kps = source_kps[used_kps, :2]
            target_kps = target_kps[used_kps, :2]

            sample = {
                    'category': category,
                    'category_id': self.classes.index(category),
                    'src_img_path': src_img_path,
                    'trg_img_path': trg_img_path,
                    'pair_name': osp.basename(pair)[:-5],
                    'src_kps': source_kps.T,
                    'trg_kps': target_kps.T,
                    'n_pts': target_kps.shape[0],
                    'src_bbox': source_bbox,
                    'trg_bbox': target_bbox,
                }
            
            if category not in data_dict:
                data_dict[category] = [sample]
            else:
                data_dict[category] += [sample]
            data_list.append(sample)

        return data_list, data_dict

    def __len__(self):
        return len(self.data_list)

    def construct_unlabeled_data(self, data_list):
        data_dict = dict()
        for sample in data_list:
            category = sample['category']
            path1 = sample['src_img_path']
            path2 = sample['trg_img_path']
            if category not in data_dict:
                data_dict[category] = [path1, path2]
            else:
                data_dict[category] += [path1, path2]

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
        batch = {
            'category': category,
            'category_id': self.classes.index(category),
            'src_img_path': src_path,
            'trg_img_path': trg_path
        }
        return batch


    def __getitem__(self, idx):  #
        batch = copy.deepcopy(self.data_list[idx])

        if self.split == 'trn':
            batch = self.trn_transform(batch)
        else:
            batch = self.tst_transform(batch)

        batch['pckthres'] = max(batch['trg_bbox'][3], batch['trg_bbox'][2])

        if self.split == 'trn':
            unlabeled_data = self.get_unlabeled_data(batch['category'])
            batch_strong = self.strong_transform(copy.deepcopy(unlabeled_data))
            batch_weak = self.weak_transform(copy.deepcopy(unlabeled_data))
            batch['src_img_strong'] = batch_strong['src_img']
            batch['trg_img_strong'] = batch_strong['trg_img']
            batch['src_img_weak'] = batch_weak['src_img']
            batch['trg_img_weak'] = batch_weak['trg_img']
            batch['unlabeled_cls_id'] = unlabeled_data['category_id']

        return batch
    
