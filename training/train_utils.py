import torch
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
        K1, K2 = transformation['K1'], transformation['K2']
        T, R = transformation['T'], transformation['R']
        depth = transformation['depth0']
        return reproject_3d(kpts, K1, K2, T, R, depth)
    else:
        raise ValueError(f'Unknown transformation type {transformation_type}.')


def perspective_transform(kpts, H, eps=1e-8):
    """Transform batch of keypoints given batch of homography matrices"""
    batch_size, num_kpts, _ = kpts.size()
    kpts = torch.cat([kpts, torch.ones(batch_size, num_kpts, 1, device=kpts.device)], dim=2)
    Ht = torch.transpose(H, 1, 2).contiguous()
    kpts_transformed = torch.matmul(kpts, Ht)
    kpts_transformed = kpts_transformed[..., :2] / (kpts_transformed[..., 2].unsqueeze(-1) + eps)
    return kpts_transformed


def reproject_3d(kpts, K1, K2, T, R, depth):
    """Transform batch of keypoints given batch of relative poses and depth maps"""
    pass


if __name__ == '__main__':
    import cv2

    kpts = np.random.randn(1, 5, 2)
    H = np.random.randn(3, 3)
    transformed_cv = cv2.perspectiveTransform(kpts, H)
    print(kpts, transformed_cv, sep='\n')

    kpts_torch = torch.FloatTensor(kpts)
    H = torch.FloatTensor(H).unsqueeze(0)
    transformed_my = perspective_transform(kpts_torch, H)
    print(transformed_my.numpy())
