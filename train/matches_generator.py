import torch
import torch.nn as nn

from models.superpoint import SuperPoint
from train.train_utils import data_to_device, min_stack, reproject_keypoints


class SuperPointMatchesGenerator(nn.Module):
    """
    Class for generating ground truth matches using image pair and transformation between them.
    Keypoints are detected and described using SuperPoint model.
    """

    def __init__(self, config):
        super(SuperPointMatchesGenerator, self).__init__()
        self.superpoint = SuperPoint(config)
        self.gt_positive_threshold = config['gt_positive_threshold']

    def forward(self, data):
        pred0 = self.superpoint({'image': data['image0']})
        pred1 = self.superpoint({'image': data['image1']})

        with torch.no_grad():
            pred0, pred1 = min_stack(pred0), min_stack(pred1)

            # establish ground truth correspondences given transformation
            kpts0, kpts1 = pred0['keypoints'], pred1['keypoints']
            desc0, desc1 = pred0['descriptors'], pred1['descriptors']
            scores0, scores1 = pred0['scores'], pred1['scores']
            transformation = data['transformation']
            num0, num1 = kpts0.size(1), kpts1.size(1)

            kpts0_transformed = reproject_keypoints(kpts0, transformation)
            reprojection_error = torch.cdist(kpts0_transformed, kpts1, p=2)  # batch_size x num0 x num1

            min_dist0, gt_matches0 = reprojection_error.min(2)  # batch_size x num0
            min_dist1, gt_matches1 = reprojection_error.min(1)  # batch_size x num1
            # remove matches that don't satisfy cross-check
            device = gt_matches0.device
            cross_check_inconsistent = torch.arange(num0, device=device).unsqueeze(0) != \
                                       torch.gather(gt_matches1, dim=1, index=gt_matches0)
            gt_matches0[cross_check_inconsistent] = -1
            # remove matches with large distance
            distance_inconsistent = min_dist0 > self.gt_positive_threshold
            gt_matches0[distance_inconsistent] = -1

        return {
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'descriptors0': desc0,
            'descriptors1': desc1,
            'scores0': scores0,
            'scores1': scores1,
            'image0': data['image0'],
            'image1': data['image1'],
            'gt_matches0': gt_matches0,
        }


if __name__ == '__main__':
    from datasets.megadepth import MegaDepthWarpingDataset

    device = torch.device('cuda:0')

    matches_generator = SuperPointMatchesGenerator(
        config=dict(
            max_keypoints=2048,
            keypoint_threshold=0.005,
            gt_positive_threshold=3
        )
    )
    matches_generator.eval().to(device)

    with open('../assets/megadepth_validation_scenes.txt') as f:
        scenes_list = f.readlines()
    scenes_list = [s.rstrip() for s in scenes_list]

    ds = MegaDepthWarpingDataset(
        root_path='/datasets/extra_space2/ostap/MegaDepth/phoenix/S6/zl548/MegaDepth_v1',
        scenes_list=scenes_list,
        target_size=(512, 512)
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=12, num_workers=12, shuffle=False)
    data = next(iter(dl))
    data = data_to_device(data, device)

    with torch.no_grad():
        matches_generator(data)
