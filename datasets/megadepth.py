"""This module contains datasets that use SuperPoint as keypoints detector and descriptor"""

import torch
import glob
import cv2
import numpy as np
from pathlib import Path
from itertools import chain


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
        self.target_size = target_size
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
        image = cv2.resize(image, self.target_size)

        # warp image with random perspective transformation
        height, width = image.shape
        corners = np.array([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]], dtype=np.float32)
        warp_offset = np.random.randint(-100, 100, size=(4, 2)).astype(np.float32)

        H = cv2.getPerspectiveTransform(corners, corners + warp_offset)
        warped = cv2.warpPerspective(src=image, M=H, dsize=(width, height))
        transformation = {
            'type': 'perspective',
            'H': torch.FloatTensor(H)
        }

        return {'image0': array_to_tensor(image), 'image1': array_to_tensor(warped), 'transformation': transformation}


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


