

import copy
import os.path as osp
from tqdm import tqdm
from prettytable import PrettyTable
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmengine.config import Config, ConfigDict
from mmengine.registry import RUNNERS, MODELS, METRICS, DATASETS
from mmengine.logging import MMLogger

from src.utils import init_random_seed, set_random_seed, to_cuda, Timer

import time

@RUNNERS.register_module()
class Runner:
    def __init__(self,
            model: Union[nn.Module, Dict],
            work_dir: str,
            train_dataloader: Optional[Dict] = None,
            val_dataloader: Optional[Dict] = None,
            test_dataloader: Optional[Dict] = None,
            optimizer: Optional[Dict] = None, 
            metric: Optional[Dict] = None, 
            log_level: str = 'INFO',
            resume: bool = False,
            cfg: Optional[Union[Dict, Config, ConfigDict]] = None
        ):
        self.work_dir = work_dir
        self.max_epochs = cfg.max_epochs
        self.interval = cfg.interval
        self.start_epoch = 0

        self.logger = self.build_logger(log_level, log_name=cfg.log_name)
        self.metric_cfg = copy.deepcopy(metric)
        self.cfg = copy.deepcopy(cfg)

        # build model and 
        self.model = self.build_model(model, cfg.get('model_path', None))
        self.ema_model = self.build_model(model, cfg.get('model_path', None))
        with torch.no_grad():
            for t_params, s_params in zip(self.ema_model.parameters(), self.model.parameters()):
                t_params.data = s_params.data
            
            for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers()):
                ema_buffer.data = buffer

        for param in self.ema_model.parameters():
            param.requires_grad = False
            param.detach_()

        self.optimizer = self.build_optimizer(self.model, optimizer)
        self.train_dataloader = self.build_dataloader(train_dataloader)
        self.val_dataloader = self.build_dataloader(val_dataloader)
        self.test_dataloader = self.build_dataloader(test_dataloader)

        self.logger.info(f'work_dir: {work_dir}')
        self.logger.info(f'train batch size: {self.train_dataloader.batch_size}')
        self.logger.info(f'test batch size: {self.test_dataloader.batch_size}')
        self.logger.info(f'train data size: {len(self.train_dataloader)}')
        self.logger.info(f'val data size: {len(self.val_dataloader)}')
        self.logger.info(f'test data size: {len(self.test_dataloader)}')
        
        self.best_pck = 0.0
        set_random_seed(123)
        
    @classmethod
    def from_cfg(cls, cfg: Union[Dict, Config, ConfigDict]):
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model = cfg['model'],
            work_dir = cfg['work_dir'],
            train_dataloader = cfg.get('train_dataloader'),
            val_dataloader = cfg.get('val_dataloader'),
            test_dataloader = cfg.get('test_dataloader'),
            optimizer = cfg.get('optimizer'),
            metric = cfg.get('metric'),
            log_level = cfg.get('log_level'),
            resume = cfg.get('resume'),
            cfg = cfg,
        )
        return runner
    
    def build_model(self, model_cfg: Dict, model_path: str=None):
        model_cfg = copy.deepcopy(model_cfg)
        model = MODELS.build(model_cfg).cuda()
        if model_path is not None:
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt, strict=False)
        return model
        
    def build_optimizer(self, model, optimizer_cfg):
        optimizer_cfg = copy.deepcopy(optimizer_cfg)
        optim_type = optimizer_cfg.pop('type')
        optimizer_cfg['params'] = model.parameters()

        if optim_type == 'SGD':
            optimizer = torch.optim.SGD(**optimizer_cfg)
        elif optim_type == 'Adam':
            optimizer = torch.optim.Adam(**optimizer_cfg)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(**optimizer_cfg)
        else: 
            raise ValueError('Do not support such type of the optimizer.')
        return optimizer
    
    def build_dataloader(self, dataloader: Optional[Dict] = None):
        if dataloader is None:
            return None
        
        dataloader_cfg = copy.deepcopy(dataloader)
        dataset_cfg = dataloader_cfg.pop('dataset')
        dataset = DATASETS.build(dataset_cfg)
        data_loader = DataLoader(dataset=dataset, **dataloader_cfg)
        return data_loader

    def build_logger(self,
                     log_level: Union[int, str] = 'INFO',
                     log_name: str = None,
                     **kwargs) -> MMLogger:
        
        log_file = osp.join(self.work_dir, f'{log_name}.txt')
        log_cfg = dict(log_level=log_level, log_file=log_file, **kwargs)
        log_cfg.setdefault('name', 'abc')
        log_cfg.setdefault('file_mode', 'a+')

        return MMLogger.get_instance(**log_cfg)  # type: ignore

    def resume(self):
        latest_ckpt_path = osp.join(self.work_dir, 'latest.pth')
        ckpt = torch.load(latest_ckpt_path)

        self.best_pck = ckpt['pck']
        self.start_epoch = ckpt['epoch'] + 1
        self.best_epoch = ckpt['best_epoch']

        self.model.load_state_dict(ckpt['model'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer'])

        self.logger.info(f'resume from: {latest_ckpt_path}')
        self.logger.info(f'best PCK: {self.best_pck}')
        self.logger.info(f'continue training...')

    def run(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            # train one epoch and evaluate the model
            results_train = self.train(epoch)
            results_val = self.val(self.ema_model)
            self.record_epoch(results_train, results_val)
            self.save_best_checkpoint(results_val, epoch)

        # print best results
        self.logger.info(f'Best PCK: {self.best_pck}')

    
    def train(self, epoch):
        
        # build metric
        metric = METRICS.build(self.metric_cfg)

        timer = Timer()
        timer.record()
        self.model.train()
        if self.ema_model:
            self.ema_model.eval()
        total_iter = len(self.train_dataloader)
        for idx, batch in enumerate(self.train_dataloader): 
            to_cuda(batch)

            outputs = self.model.forward_step(batch)
            if self.ema_model is not None and epoch > 0:
                out1 = self.model.forward_only_flow(batch['src_img_strong'], batch['trg_img_strong'])
                with torch.no_grad():
                    out2 = self.ema_model.forward_only_flow(batch['src_img_weak'], batch['trg_img_weak'], category_ids=batch['unlabeled_cls_id'])
                semi_loss = self.ema_model.compute_pseudo_loss(out1['flow'], out2['flow'], mask=out2['mask']) * 0.3

            loss = outputs['total_loss']
            if self.ema_model is not None and epoch > 0:
                loss = loss + semi_loss
                outputs['sup_loss'] = outputs['total_loss']
                outputs['semi_loss'] = semi_loss
                outputs['total_loss'] = outputs['sup_loss'] + outputs['semi_loss']

            # update parameters
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            results = metric.update_metrics(outputs, batch)

            if self.ema_model:
                update_ema_variables(self.model, self.ema_model, ema_decay=0.999)

            timer.record()
            time_info = timer.infer_eta_time(idx, total_iter)
            self.record_train_iter(results, time_info, epoch, idx, total_iter)

        results = metric.get_metrics()
        return results

    def val(self, model):
        # build metric
        metric = METRICS.build(self.metric_cfg)

        model.eval()
        for idx, batch in enumerate(self.val_dataloader):
            # forward model
            to_cuda(batch)
            with torch.no_grad():
                outputs = model.inference(batch)

            # compute metrics
            results = metric.update_metrics(outputs, batch)
            self.record_val_iter(results, idx, len(self.val_dataloader))

        results = metric.get_metrics()
        return results

    def inference(self):
        # set evaluation metric
        metric = METRICS.build(self.metric_cfg)

        self.ema_model.eval()
        for idx, batch in enumerate(self.test_dataloader):
            # forward model
            to_cuda(batch)
            with torch.no_grad():
                outputs = self.ema_model.inference(batch, visualize=False)

            # compute metrics
            results = metric.update_metrics(outputs, batch)
            self.record_val_iter(results, idx, len(self.test_dataloader))

        results = metric.get_metrics()
        self.record_infer(results)
        return results

    def record_train_iter(self, results, time_info, epoch, iter, max_iter):
        if (iter + 1) % self.interval != 0:
            return 
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        process = 'Epoch:[{:03d}/{}] Iter:[{:04d}/{}] '.format(epoch + 1, self.max_epochs, iter + 1, max_iter)
        eta = 'eta:{:02d}:{:02d}:{:02d} '.format(time_info['hours'], time_info['minutes'], time_info['seconds'])
        res = ' '.join([f'{key}: {val:.4f}' for key, val in results.items() if 'loss' in key])  # for loss, mIoU, acc, etc.
        res += f" pck: {results['pck']:.3f}"
        msg = process + eta + res + f' lr: {lr:.4e}'
        self.logger.info(msg)

    def record_val_iter(self, results, iter, max_iter):
        # if use gradient accumulation, skip current step
        if (iter + 1) % self.interval != 0:
            return 
        process = 'Iter:[{:04d}/{}] '.format(iter + 1, max_iter)
        res = ' '.join([f'{key}: {val:.4f}' for key, val in results.items() if 'loss' in key])
        res += f" pck: {results['pck']:.3f}"
        msg = process + res
        self.logger.info(msg)

    def record_epoch(self, results_train, results_val):
        table = PrettyTable(['mode'] + ['total_loss', 'pck'])
        table.add_row(['Training'] + [f"{results_train['total_loss']:.4f}", f"{results_train['pck']:.4f}"])
        table.add_row(['Validation'] + [f"{results_val['total_loss']:.4f}", f"{results_val['pck']:.4f}"])
        self.logger.info('\n' + table.get_string())

    def record_infer(self, results):
        table = PrettyTable(['total_loss', 'pck'])
        table.add_row([f"{results['total_loss']:.3f}", f"{results['pck']:.3f}"])
        self.logger.info('\n' + table.get_string())
        
        for key, val in results['category_pck'].items():
            self.logger.info(f'{key}\t{val:.1f}')

        alphas=[0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30]
        for alpha, pck in zip(alphas, results['pck_alpha']):
            self.logger.info(f'{alpha}: {pck:.1f}')

    def save_best_checkpoint(self, results_val, epoch):
        cur_pck = results_val['pck']
        if cur_pck > self.best_pck:
            self.best_pck = cur_pck
            save_path = osp.join(self.work_dir, 'best_model.pth')
            torch.save(self.model.state_dict(), save_path)
            torch.save(self.ema_model.state_dict(), save_path.replace('best_model', 'best_ema_model'))
            self.best_epoch = epoch
            self.logger.info('Save checkpoint')
            self.logger.info(f'Epoch:{epoch + 1} Best PCK:{cur_pck:.2f}')

        self.save_last_checkpoint(epoch)

    def save_last_checkpoint(self, epoch):
        latest_ckpt = {
            'epoch': epoch,  # current epoch
            'best_epoch': self.best_epoch,
            'pck': self.best_pck,
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        latest_ckpt_path = osp.join(self.work_dir, 'latest.pth')
        torch.save(latest_ckpt, latest_ckpt_path)


def update_ema_variables(model, ema_model, ema_decay):
    # update param
    model.eval()
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        if param.requires_grad:
            ema_param.data = ema_param.data * ema_decay + param.data * (1 - ema_decay)

    # update bn
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data = ema_buffer.data * ema_decay + buffer * (1 - ema_decay)

    model.train()
   

