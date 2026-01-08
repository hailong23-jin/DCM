import numpy as np 
from PIL import Image


class LoadImage:
    def __init__(self, using_org_img=True):
        self.using_org_img = using_org_img
    def __call__(self, batch):
        
        src_path = batch['src_img_path']
        trg_path = batch['trg_img_path']

        src_img = self.read_img(src_path)
        trg_img = self.read_img(trg_path)

        batch['src_img'] = src_img
        batch['trg_img'] = trg_img
        batch['src_imsize'] = np.array(src_img.shape[:2])  # [h, w]
        batch['trg_imsize'] = np.array(trg_img.shape[:2])
        if self.using_org_img:
            batch['org_trg_img'] = np.pad(trg_img, pad_width=((0, 1024 - trg_img.shape[0]), (0, 1024 - trg_img.shape[1]), (0, 0))) 

        return batch

    def read_img(self, path):
        return np.array(Image.open(path).convert('RGB'))
