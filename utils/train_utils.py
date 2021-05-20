import torch
import torch.nn.functional as F
import numpy as np


def data_to_device(data, device):
    """Recursively transfers all tensors in dictionary to given device"""
    if isinstance(data, torch.Tensor):
        res = data.to(device)
    elif isinstance(data, dict):
        res = {k: data_to_device(v, device) for k, v in data.items()}
    else:
        res = data
    return res


def min_stack(data):
    """
    Stack batch of keypoints prediction into single tensor.
    For each instance keep number of keypoints minimal in the batch. Discard other low confidence keypoints.
    """
    kpts_num = np.array([x.shape[0] for x in data['keypoints']])
    min_kpts_to_keep = kpts_num.min()

    if np.all(kpts_num == min_kpts_to_keep):
        data_stacked = data
    else:
        # get scores and indices of keypoints to keep in each batch element
        indices_to_keep = [torch.topk(scores, min_kpts_to_keep, dim=0) for scores in data['scores']]

        data_stacked = {k: [] for k in data.keys()}
        for keypoints, descriptors, (scores, indices) in zip(data['keypoints'], data['descriptors'], indices_to_keep):
            data_stacked['scores'].append(scores)
            data_stacked['keypoints'].append(keypoints[indices])
            data_stacked['descriptors'].append(descriptors[:, indices])
    data_stacked = {k: torch.stack(v, dim=0) for k, v in data_stacked.items()}

    return data_stacked


def reproject_keypoints(kpts, transformation):
    """Reproject batch of keypoints given corresponding transformations"""
    transformation_type = transformation['type'][0]
    if transformation_type == 'perspective':
        H = transformation['H']
        return perspective_transform(kpts, H)
    elif transformation_type == '3d_reprojection':
        K0, K1 = transformation['K0'], transformation['K1']
        T, R = transformation['T'], transformation['R']
        depth0 = transformation['depth0']
        return reproject_3d(kpts, K0, K1, T, R, depth0)
    else:
        raise ValueError(f'Unknown transformation type {transformation_type}.')


def get_inverse_transformation(transformation):
    transformation_type = transformation['type'][0]
    if transformation_type == 'perspective':
        H = transformation['H']
        return {
            'type': transformation['type'],
            'H': torch.linalg.inv(H)
        }
    elif transformation_type == '3d_reprojection':
        K0, K1 = transformation['K0'], transformation['K1']
        T, R = transformation['T'], transformation['R']
        depth0, depth1 = transformation['depth0'], transformation['depth1']
        R_t = torch.transpose(R, 1, 2).contiguous()
        return {
            'type': transformation['type'],
            'K0': K1,
            'K1': K0,
            'R': R_t,
            'T': -torch.matmul(R_t, T.unsqueeze(-1)).squeeze(-1),
            'depth0': depth1,
            'depth1': depth0,
        }
    else:
        raise ValueError(f'Unknown transformation type {transformation_type}.')


def perspective_transform(kpts, H, eps=1e-8):
    """Transform batch of keypoints given batch of homography matrices"""
    batch_size, num_kpts, _ = kpts.size()
    kpts = torch.cat([kpts, torch.ones(batch_size, num_kpts, 1, device=kpts.device)], dim=2)
    Ht = torch.transpose(H, 1, 2).contiguous()
    kpts_transformed = torch.matmul(kpts, Ht)
    kpts_transformed = kpts_transformed[..., :2] / (kpts_transformed[..., 2].unsqueeze(-1) + eps)
    mask = torch.ones(batch_size, num_kpts, dtype=torch.bool)  # all keypoints are valid
    return kpts_transformed, mask


def reproject_3d(kpts, K0, K1, T, R, depth0, eps=1e-8):
    """Transform batch of keypoints given batch of relative poses and depth maps"""
    batch_size, num_kpts, _ = kpts.size()
    kpts_hom = torch.cat([kpts, torch.ones(batch_size, num_kpts, 1, device=kpts.device)], dim=2)

    K0_inv = torch.linalg.inv(K0)
    K0_inv_t = torch.transpose(K0_inv, 1, 2).contiguous()
    K1_t = torch.transpose(K1, 1, 2).contiguous()
    R_t = torch.transpose(R, 1, 2).contiguous()

    # transform to ray space
    kpts_transformed = torch.matmul(kpts_hom, K0_inv_t)

    depth_idx = kpts.type(torch.int64)
    depth = depth0[
        torch.arange(batch_size, device=kpts.device).unsqueeze(-1),
        depth_idx[..., 1],
        depth_idx[..., 0]
    ]
    mask = depth != 0  # mask for values with missing depth information
    # multiply by corresponding depth
    kpts_transformed = kpts_transformed * depth.unsqueeze(-1)
    # apply (R, T)
    kpts_transformed = torch.matmul(kpts_transformed, R_t) + T.unsqueeze(1)
    kpts_transformed = torch.matmul(kpts_transformed, K1_t)
    kpts_transformed = kpts_transformed[..., :2] / (kpts_transformed[..., 2].unsqueeze(-1) + eps)
    return kpts_transformed, mask


def pairwise_cosine_dist(x1, x2):
    """
    Return pairwise half of cosine distance in range [0, 1].
    dist = (1 - cos(theta)) / 2
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return 0.25 * torch.cdist(x1, x2).pow(2)


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1
