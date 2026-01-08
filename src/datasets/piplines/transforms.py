import cv2
import numpy as np 
import torch
from torchvision import transforms
import albumentations as A
import random
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import math



class Resize:
    def __init__(self, target_size) -> None:
        self.target_size = target_size

    def __call__(self, batch):
        batch['src_img'] = cv2.resize(batch['src_img'], dsize=self.target_size)
        batch['trg_img'] = cv2.resize(batch['trg_img'], dsize=self.target_size)
        return batch


class RandomCrop:
    def __init__(self, target_size, p=0.5) -> None:
        self.target_size = target_size
        self.p = p

    def random_crop(self, img, kps, bbox):
        h, w, _ = img.shape
        kps = kps.T
        left = random.randint(0, bbox[0])
        top = random.randint(0, bbox[1])
        height = random.randint(bbox[3], h) - top
        width = random.randint(bbox[2], w) - left

        crop_img = img[top:(top+height), left:(left+width), :]
        
        crop_img = cv2.resize(crop_img, dsize=self.target_size)

        resized_kps = np.zeros_like(kps, dtype=np.float32)
        resized_kps[:, 0] = (kps[:, 0] - left) * (self.target_size[1] / width)
        resized_kps[:, 1] = (kps[:, 1] - top) * (self.target_size[0] / height)
        resized_kps = np.clip(resized_kps, 0, self.target_size[0] - 1)
        return crop_img, resized_kps.T

    def resize(self, img, kps):
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=self.target_size)
        kps[0, :] = kps[0, :] * (self.target_size[1] / w)
        kps[1, :] = kps[1, :] * (self.target_size[0] / h)
        return img, kps

    def __call__(self, batch):
        if random.uniform(0, 1) > self.p:
            batch['src_img'], batch['src_kps'] = self.resize(batch['src_img'], batch['src_kps'])
            batch['trg_img'], batch['trg_kps'] = self.resize(batch['trg_img'], batch['trg_kps'])
        else:
            batch['src_img'], batch['src_kps'] = self.random_crop(batch['src_img'], batch['src_kps'], batch['src_bbox'].copy().astype(int))
            batch['trg_img'], batch['trg_kps'] = self.random_crop(batch['trg_img'], batch['trg_kps'], batch['trg_bbox'].copy().astype(int))
        return batch


class RandomRotation:
    def __init__(self, p=0.5, target_size=448):
        self.p = p
        self.target_size = target_size

    def rotate_img_points(self, image, key_points, angle):
        center = ((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # 计算旋转后的图像宽度和高度
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])

        scale = 1.0
        new_width = int(scale * (image.shape[1] * cos + image.shape[0] * sin))
        new_height = int(scale * (image.shape[1] * sin + image.shape[0] * cos))

        # 调整旋转矩阵以确保中心点位于输出图像中心
        rotation_matrix[0, 2] += ((new_width - 1) / 2) - center[0]
        rotation_matrix[1, 2] += ((new_height - 1) / 2) - center[1]

        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

        rotated_key_points = key_points.copy()
        rotated_key_points[0, :] = key_points[0, :] * rotation_matrix[0, 0] + key_points[1, :] * rotation_matrix[0, 1] + rotation_matrix[0, 2]
        rotated_key_points[1, :] = key_points[0, :] * rotation_matrix[1, 0] + key_points[1, :] * rotation_matrix[1, 1] + rotation_matrix[1, 2]

        return rotated_image, rotated_key_points

    def resize(self, img, kps):
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=(self.target_size, self.target_size))
        kps[0, :] = kps[0, :] * (self.target_size / w)
        kps[1, :] = kps[1, :] * (self.target_size / h)
        return img, kps

    def __call__(self, batch):
        if random.uniform(0, 1) < self.p:
            angle = random.random() * 60 - 30

            rand_num = random.randint(0, 2)
            if rand_num == 0 or rand_num == 2:
                batch['src_img'], batch['src_kps'] = self.rotate_img_points(batch['src_img'], batch['src_kps'], angle)
            if rand_num == 1 or rand_num == 2:
                batch['trg_img'], batch['trg_kps'] = self.rotate_img_points(batch['trg_img'], batch['trg_kps'], angle)

        batch['src_img'], batch['src_kps'] = self.resize(batch['src_img'], batch['src_kps'])
        batch['trg_img'], batch['trg_kps'] = self.resize(batch['trg_img'], batch['trg_kps'])

        return batch

def img_aug_identity(img, scale=None):
    return img

def img_aug_autocontrast(img, scale=None):
    return ImageOps.autocontrast(img)

def img_aug_equalize(img, scale=None):
    return ImageOps.equalize(img)

def img_aug_invert(img, scale=None):
    return ImageOps.invert(img)

def img_aug_blur(img, scale=[0.1, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))

def img_aug_contrast(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    return ImageEnhance.Contrast(img).enhance(v)

def img_aug_brightness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    return ImageEnhance.Brightness(img).enhance(v)

def img_aug_color(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    return ImageEnhance.Color(img).enhance(v)

def img_aug_sharpness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    return ImageEnhance.Sharpness(img).enhance(v)

def img_aug_hue(img, scale=[0, 0.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v += min_v
    if np.random.random() < 0.5:
        hue_factor = -v
    else:
        hue_factor = v

    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img
    h, s, v = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)

    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")
    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img

def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.posterize(img, v)

def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.solarize(img, v)

class StrongAug:
    def __init__(self) -> None:
        self.transforms = [
            (img_aug_autocontrast, None),
            (img_aug_equalize, None),
            (img_aug_blur, [0.1, 2.0]),
            (img_aug_contrast, [0.05, 0.95]),
            (img_aug_brightness, [0.05, 0.95]),
            (img_aug_color, [0.05, 0.95]),
            (img_aug_sharpness, [0.05, 0.95]),
            (img_aug_posterize, [4, 8]),
            (img_aug_solarize, [1, 256]),
            (img_aug_hue, [0, 0.5])
        ]
        self.num_augs = 3

    def aug_img(self, img):
        img = Image.fromarray(img)
        max_num = np.random.randint(1, high=self.num_augs + 1)
        ops = random.choices(self.transforms, k=max_num)
        for op, scales in ops:
            img = op(img, scales)
        return np.array(img)

    def __call__(self, batch):
        batch['src_img'] = self.aug_img(batch['src_img'])
        batch['trg_img'] = self.aug_img(batch['trg_img'])
        return batch


class NormalAug:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.ToGray(p=0.1),
            A.Posterize(p=0.2),
            A.Equalize(p=0.2),
            A.augmentations.transforms.Sharpen(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Solarize(p=0.2),
            A.ColorJitter(p=0.2),
        ])

    def __call__(self, batch):
        batch['src_img'] = self.transform(image=batch['src_img'])['image']
        batch['trg_img'] = self.transform(image=batch['trg_img'])['image']
        return batch


class PadKeyPoints:
    def __init__(self, max_num):
        self.max_num = max_num

    def pad_kps(self, kps):
        pad_num = self.max_num - kps.shape[-1]
        pad_pts = np.ones((2, pad_num)) * -1
        kps = np.concatenate([kps, pad_pts], axis=1)
        return kps
        
    def pad_kps_ids(self, kps_ids):
        pad_num = self.max_num - kps_ids.shape[-1]
        pad_vals = np.ones(pad_num, dtype=int) * -1
        kps_ids = np.concatenate([kps_ids, pad_vals], axis=0)
        return kps_ids

    def __call__(self, batch):
        batch['src_kps'] = self.pad_kps(batch['src_kps'])
        batch['trg_kps'] = self.pad_kps(batch['trg_kps'])
        if 'kps_ids' in batch:
            batch['kps_ids'] = self.pad_kps_ids(batch['kps_ids'])
        return batch


class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, batch):
        batch['src_img'] = torch.from_numpy(batch['src_img']).float()
        batch['trg_img'] = torch.from_numpy(batch['trg_img']).float()
        if 'src_kps' in batch:
            batch['src_kps'] = torch.from_numpy(batch['src_kps']).float()
        if 'trg_kps' in batch:
            batch['trg_kps'] = torch.from_numpy(batch['trg_kps']).float()

        return batch


class Normalize:
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, batch):
        batch['src_img'] = self.normalize(batch['src_img'].permute(2, 0, 1) / 255.0)
        batch['trg_img'] = self.normalize(batch['trg_img'].permute(2, 0, 1) / 255.0)
        return batch

# --------------  for test -------------- #

def get_center_point(src_kps):
    x = (src_kps[0, :].max() + src_kps[0, :].min()) / 2.0
    y = (src_kps[1, :].max() + src_kps[1, :].min()) / 2.0
    # return int(x), int(y)
    return math.floor(x), math.floor(y)

def center_crop(img, kps, crop_size):
    '''
    img: np.array h x w x 3 
    crop_size: h x w
    '''
    img_h, img_w = img.shape[:2]
    center = get_center_point(kps)

    offset_x = crop_size[1] // 2
    offset_y = crop_size[0] // 2
    x1, y1 = center[0] - offset_x, center[1] - offset_y
    x2, y2 = center[0] + offset_x, center[1] + offset_y
    if x2 >= img_w:
        x1 = x1 - (x2 - img_w + 1)
        x2 = img_w - 1
    if y2 >= img_h:
        y1 = y1 - (y2 - img_h + 1)
        y2 = img_h - 1
    if x1 < 0:
        x2 = x2 + abs(x1)
        x1 = 0
    if y1 < 0:
        y2 = y2 + abs(y1)
        y1 = 0
    
    img = np.pad(img, pad_width=((0, max(0, crop_size[0] - img_h)), (0, max(0, crop_size[1] - img_w)), (0, 0)))
    img = img[y1:(y1+crop_size[0]), x1:(x1+crop_size[1])]
    
    kps[0, :] = kps[0, :] - x1
    kps[1, :] = kps[1, :] - y1

    return img, kps, x1, y1

def resize_kps(kps, target_size, img_size, scale=None):
    '''
    imside: 448
    img_size: h x w
    '''
    if scale is not None:
        kps = kps * scale
    else:
        kps[0, :] = kps[0, :] * (target_size[1] / img_size[1])
        kps[1, :] = kps[1, :] * (target_size[0] / img_size[0])
    return kps

def resize_bbox(bbox, target_size, img_size, scale=None):
    if scale is not None:
        bbox = bbox * scale
    else:
        bbox[0::2] *= target_size[1] / img_size[1]
        bbox[1::2] *= target_size[0] / img_size[0]
    return bbox

def get_bounding_box(kps):
    x_min = kps[0, :].min()
    x_max = kps[0, :].max()
    y_min = kps[1, :].min()
    y_max = kps[1, :].max()
    return [x_min, y_min, x_max, y_max]

class ResizeTransform:
    def __init__(self, target_size):
        self.target_size = target_size

    def resize(self, img):
        img = cv2.resize(img, dsize=self.target_size)
        return img

    def __call__(self, batch):
        batch['src_img'] = self.resize(batch['src_img'])
        batch['trg_img'] = self.resize(batch['trg_img'])
        batch['src_kps'] = resize_kps(batch['src_kps'], self.target_size, batch['src_imsize'])
        batch['trg_kps'] = resize_kps(batch['trg_kps'], self.target_size, batch['trg_imsize'])
        batch['src_bbox'] = resize_bbox(batch['src_bbox'], self.target_size, batch['src_imsize'])
        batch['trg_bbox'] = resize_bbox(batch['trg_bbox'], self.target_size, batch['trg_imsize'])
        return batch

class TestTransform:
    def __init__(self, target_size, crop_size):
        self.target_size = target_size
        self.crop_size = crop_size

    def process_imagev2(self, img, kps):
        box = get_bounding_box(kps)
        max_side = max(box[2] - box[0], box[3] - box[1])
        if max_side >= self.crop_size:
            return img, kps

        img, kps, x1, y1 = center_crop(img, kps, (self.crop_size, self.crop_size))
        return img, kps

    def preprocess_image(self, img):
        max_size = max(self.target_size)

        h, w = img.shape[:2]
        scale = min(max_size / w, max_size / h)
        img = cv2.resize(img, dsize=(round(w*scale), round(h*scale)))

        h, w = img.shape[:2]
        left_pad = (max_size - w) // 2
        right_pad = max_size - w - left_pad
        top_pad = (max_size - h) // 2
        bottom_pad = max_size - h - top_pad
        img = np.pad(img, pad_width=((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)))
        return img, scale, left_pad, top_pad
    
    def __call__(self, batch):
        src_img, src_kps = self.process_imagev2(batch['src_img'], batch['src_kps'])
        # src_img, src_kps = batch['src_img'], batch['src_kps']
        batch['src_img'], scale, left_pad, top_pad = self.preprocess_image(src_img)
        src_kps = resize_kps(src_kps, self.target_size, None, scale=scale)
        src_kps[0, :] = src_kps[0, :] + left_pad
        src_kps[1, :] = src_kps[1, :] + top_pad
        batch['src_kps'] = src_kps
        if 'src_bbox' in batch:
            batch['src_bbox'] = resize_bbox(batch['src_bbox'], None, None, scale=scale)

        batch['trg_img'], scale, left_pad, top_pad = self.preprocess_image(batch['trg_img'])
        trg_kps = resize_kps(batch['trg_kps'], self.target_size, None, scale=scale)
        trg_kps[0, :] = trg_kps[0, :] + left_pad
        trg_kps[1, :] = trg_kps[1, :] + top_pad
        batch['trg_kps'] = trg_kps
        batch['trg_scale'] = scale
        batch['trg_left_pad'] = left_pad
        batch['trg_top_pad'] = top_pad
        if 'trg_bbox' in batch:
            batch['trg_bbox'] = resize_bbox(batch['trg_bbox'], None, None, scale=scale)

        return batch








