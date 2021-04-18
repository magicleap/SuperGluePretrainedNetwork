"""This module contains datasets that use SuperPoint as keypoints detector and descriptor"""

import torch
import glob
import cv2
import numpy as np
import deepdish as dd
from pathlib import Path
from itertools import chain
from collections import OrderedDict


def array_to_tensor(img_array):
    return torch.FloatTensor(img_array / 255.).unsqueeze(0)


class MegaDepthWarpingDataset(torch.utils.data.Dataset):
    """
    MegaDepth dataset that creates images pair by warping single image.
    """

    def __init__(self, root_path, scenes_list, target_size, color_aug_transform=None):
        self.root_path = Path(root_path)
        self.images_list = [  # iter through all scenes and concatenate the results into one list
            *chain(*[glob.glob(
                str(self.root_path / scene / 'dense*' / 'imgs' / '*')
            ) for scene in scenes_list])
        ]
        self.target_size = tuple(target_size)
        self.color_aug_transform = color_aug_transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = self.images_list[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # augment image with aug_transform
        if self.color_aug_transform is not None:
            image = self.color_aug_transform(image=image)['image']
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.target_size != -1:
            image = cv2.resize(image, self.target_size)

        # warp image with random perspective transformation
        height, width = image.shape
        corners = np.array([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]], dtype=np.float32)
        warp_offset = np.random.randint(-300, 300, size=(4, 2)).astype(np.float32)

        H = cv2.getPerspectiveTransform(corners, corners + warp_offset)
        warped = cv2.warpPerspective(src=image, M=H, dsize=(width, height))
        transformation = {
            'type': 'perspective',
            'H': torch.FloatTensor(H)
        }

        return {'image0': array_to_tensor(image), 'image1': array_to_tensor(warped), 'transformation': transformation}


class MegaDepthPairsDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, scenes_list, target_size, color_aug_transform=None):
        self.root_path = Path(root_path)
        self.target_size = tuple(target_size)
        self.color_aug_transform = color_aug_transform

        pairs_metadata_files = {scene: self.root_path / 'Undistorted-SfM' / scene / 'sparse-txt' / 'pairs.txt' for scene
                                in scenes_list}
        self.image_pairs = OrderedDict()
        for scene, pairs_path in pairs_metadata_files.items():
            with open(pairs_path) as f:
                pairs_metadata = f.readlines()
                pairs_metadata = list(map(lambda x: x.rstrip(), pairs_metadata))
            self.image_pairs[scene] = pairs_metadata
        self.scene_pairs_numbers = OrderedDict([(k, len(v)) for k, v in self.image_pairs.items()])

    def __len__(self):
        return sum(self.scene_pairs_numbers.values())

    def __getitem__(self, idx):
        for s, pairs_num in self.scene_pairs_numbers.items():
            if idx < pairs_num:
                scene, scene_idx = s, idx
                break
            else:
                idx -= pairs_num
        metadata = self.image_pairs[scene][scene_idx]
        img0_name, img1_name, K0, K1, R, T = self.parse_pairs_line(metadata)

        # read and transform images
        images = []
        for img_name, K in ((img0_name, K0), (img1_name, K1)):
            image = cv2.imread(str(self.root_path / 'phoenix/S6/zl548/MegaDepth_v1' / scene / 'dense0/imgs' / img_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.color_aug_transform is not None:
                image = self.color_aug_transform(image=image)['image']
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            depth = dd.io.load(str(
                self.root_path / 'phoenix/S6/zl548/MegaDepth_v1' / scene / 'dense0/depths' / (img_name[:-3] + 'h5')))[
                'depth']

            if self.target_size != -1:
                size = image.shape[:2][::-1]
                scales = np.diag([self.target_size[0] / size[0], self.target_size[1] / size[1], 1.0]).astype(np.float32)

                K = np.dot(scales, K)
                image = cv2.resize(image, self.target_size)
                depth = cv2.resize(depth, self.target_size)

            images.append((image, depth, K))

        (image0, depth0, K0), (image1, depth1, K1) = images

        transformation = {
            'type': '3d_reprojection',
            'K0': torch.from_numpy(K0),
            'K1': torch.from_numpy(K1),
            'R': torch.from_numpy(R),
            'T': torch.from_numpy(T),
            'depth0': torch.from_numpy(depth0),
            'depth1': torch.from_numpy(depth1),
        }

        return {'image0': array_to_tensor(image0), 'image1': array_to_tensor(image1), 'transformation': transformation}

    @staticmethod
    def parse_pairs_line(line):
        img0_name, img1_name, _, _, *camera_params, _ = line.split(' ')
        camera_params = list(map(lambda x: float(x), camera_params))
        K0, K1, RT = camera_params[:9], camera_params[9:18], camera_params[18:]
        K0 = np.array(K0).astype(np.float32).reshape(3, 3)
        K1 = np.array(K1).astype(np.float32).reshape(3, 3)
        RT = np.array(RT).astype(np.float32).reshape(4, 4)
        R, T = RT[:3, :3], RT[:3, 3]
        return img0_name, img1_name, K0, K1, R, T


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    with open('../assets/megadepth_validation_scenes.txt') as f:
        scenes_list = f.readlines()
    scenes_list = [s.rstrip() for s in scenes_list]

    ds = MegaDepthWarpingDataset(
        root_path='/datasets/extra_space2/ostap/MegaDepth/phoenix/S6/zl548/MegaDepth_v1',
        scenes_list=scenes_list,
        target_size=(352, 352)
    )
    data = ds[150]
    img1, img2 = data['image0'].numpy()[0], data['image1'].numpy()[0]
    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(img1)
    axes[1].imshow(img2)
    plt.show()
