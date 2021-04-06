import torch
import torch.nn as nn

from models.superpoint import SuperPoint
from train.train_utils import data_to_device


class SuperPointMatchesGenerator(nn.Module):
    """
    Class for generating ground truth matches using image pair and transformation between them.
    Keypoints are detected and described using SuperPoint model.
    """

    def __init__(self, superpoint_config):
        super(SuperPointMatchesGenerator, self).__init__()
        self.superpoint = SuperPoint(superpoint_config)

    def forward(self, data):
        pred0 = self.superpoint({'image': data['image0']})
        pred1 = self.superpoint({'image': data['image1']})

        pred0, pred1 = self.min_stack(pred0), self.min_stack(pred1)
        print(pred0['scores'].shape, pred1['scores'].shape)
        # TODO: establish ground truth correspondences given transformation

    @staticmethod
    def min_stack(data):
        """
        Stack batch of keypoints prediction into single tensor.
        For each instance keep number of keypoints minimal in the batch. Discard other low confidence keypoints.
        """
        kpts_to_keep = min(data['keypoints'], key=lambda x: x.shape[0]).shape[0]
        # get scores and indices of keypoints to keep in each batch element
        indices_to_keep = [torch.topk(scores, kpts_to_keep, dim=0) for scores in data['scores']]

        data_stacked = {k: [] for k in data.keys()}
        for keypoints, descriptors, (scores, indices) in zip(data['keypoints'], data['descriptors'], indices_to_keep):
            data_stacked['scores'].append(scores)
            data_stacked['keypoints'].append(keypoints[indices])
            data_stacked['descriptors'].append(descriptors[:, indices])
        data_stacked = {k: torch.stack(v, dim=0) for k, v in data_stacked.items()}

        return data_stacked


if __name__ == '__main__':
    from datasets.megadepth import MegaDepthWarpingDataset

    device = torch.device('cuda:0')

    matches_generator = SuperPointMatchesGenerator(
        superpoint_config=dict(
            max_keypoints=2048
        )
    )
    matches_generator.eval().to(device)

    with open('../assets/megadepth_validation_scenes.txt') as f:
        scenes_list = f.readlines()
    scenes_list = [s.rstrip() for s in scenes_list]

    ds = MegaDepthWarpingDataset(
        root_path='/datasets/extra_space2/ostap/MegaDepth/phoenix/S6/zl548/MegaDepth_v1',
        scenes_list=scenes_list,
        target_size=(768, 768)
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=12, num_workers=12, shuffle=False)
    data = next(iter(dl))
    data = data_to_device(data, device)

    with torch.no_grad():
        matches_generator(data)
