import torch
import numpy as np 
import shutil

def find_object_region(mask: torch.Tensor, expand_value: int=0):
    h, w = mask.shape
    # Project mask onto x and y dimensions
    projection_x = torch.sum(mask, dim=0)
    projection_y = torch.sum(mask, dim=1)

    # Find non-zero indices
    x_indices = torch.nonzero(projection_x).squeeze(1)
    y_indices = torch.nonzero(projection_y).squeeze(1)

    # Calculate bounding box coordinates
    if x_indices.numel() == 0:
        x_min, x_max = 0, w - 1
        y_min, y_max = 0, h - 1
    else:
        x_min, x_max = x_indices.min().item(), x_indices.max().item()
        y_min, y_max = y_indices.min().item(), y_indices.max().item()

    # expand bounding box
    if x_min - expand_value > 0 and y_min - expand_value > 0:
        x_min = x_min - expand_value
        y_min = y_min - expand_value
    if x_max + expand_value < w and y_max + expand_value < h:
        x_max = x_max + expand_value
        y_max = y_max + expand_value

    # Return the bounding box coordinates
    return [x_min, y_min, x_max, y_max]

def find_object_region_batch(mask: torch.Tensor, expand_value: int=0):
    box_list = []
    for m in mask:
        box_list.append(find_object_region(m, expand_value))
    return torch.tensor(box_list).cuda()

def find_object_region_np(mask: np.ndarray):
    # Project mask onto x and y dimensions
    projection_x = np.sum(mask, axis=0)
    projection_y = np.sum(mask, axis=1)

    # Find non-zero indices
    x_indices = np.nonzero(projection_x)[0]
    y_indices = np.nonzero(projection_y)[0]

    # Calculate bounding box coordinates
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Return the bounding box coordinates
    return x_min, y_min, x_max, y_max


def voronoi_split(region_num, mask, sampler):
        """
        Parameters
        ----------
        region_num: int
            Background partition numbers
        mask: torch.Tensor
            [B, h, w], bool
        sampler: np.random.RandomState

        Returns
        -------
        bg_proto: torch.Tensor
            [B, c, k], where k is the number of background proxies
        """
        B, h, w = mask.shape

        unique_labels = torch.unique(mask)
        if True not in unique_labels:
            return torch.zeros_like(mask).to(mask.device)
        
        new_mask = []
        for b in range(B):
            # bg_protos = []
            mask_i = mask[b]     # [h, w]

            # Check if zero
            with torch.no_grad():
                if mask_i.sum() < region_num:
                    mask_i = mask[b].clone()    # don't change original mask
                    mask_i.view(-1)[:region_num] = True

            # Iteratively select farthest points as centers of background local regions
            all_centers = []
            first = True
            points = torch.stack(torch.where(mask_i), dim=1)     # [N, 2]
            for _ in range(region_num):
                if first:
                    i = sampler.choice(points.shape[0])
                    first = False
                else:
                    dist = points.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                    # choose the farthest point
                    i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
                pt = points[i]   # center y, x
                all_centers.append(pt)
        
            # Assign bg labels for bg pixels
            dist = points.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
            bg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)

            tmp_mask = torch.zeros(size=[h, w], dtype=torch.long).to(mask.device)
            for i in range(region_num):
                pos = points[bg_labels==i]
                tmp_mask[pos[:, 0], pos[:, 1]] = i + 1
            new_mask.append(tmp_mask)
        
        return torch.stack(new_mask, dim=0)


def save_code(src_dir, dst_dir):
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def do_low_rank(weight, k, debug=False, niter=2):
    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k)

    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")

    results = torch.svd_lowrank(weight,
                                q=desired_rank,
                                niter=niter)
    weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    weight_approx = torch.nn.Parameter(weight_approx)

    return weight_approx
